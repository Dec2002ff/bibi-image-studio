"""Поправки к Gradio: повторная загрузка того же файла и очистка временных файлов.

Каждая поправка проверяется вместе с контролем — исходной функцией Gradio,
на которой тот же опыт показывает дефект. Без контроля зелёный тест не
отличал бы «поправка работает» от «опыт ничего не ловит».
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import types
from datetime import datetime, timedelta

import pytest

pytest.importorskip("gradio")

from gradio import route_utils, routes

from fooocus_qwen.ui import gradio_fixes

# Исходные функции — из модуля поправок: он запоминает их до подмены, а
# интерфейс в других тестах мог уже поставить поправки.
ORIGINAL_MOVE = gradio_fixes.ORIGINAL_MOVE
ORIGINAL_DELETE = gradio_fixes.ORIGINAL_DELETE


def test_install_replaces_the_functions_where_gradio_calls_them():
    gradio_fixes.install()
    assert routes.move_uploaded_files_to_cache is gradio_fixes.move_uploaded_files_to_cache
    assert route_utils.delete_files_created_by_app is gradio_fixes.delete_files_created_by_app


# --- повторная загрузка ---


def test_the_same_content_is_not_copied_over_itself(tmp_path):
    """Каталог загрузки назван хешем содержимого: файл с тем же именем — тот же файл."""
    dest = tmp_path / "sha" / "photo.jpg"
    dest.parent.mkdir()
    dest.write_bytes(b"x" * 1000)
    before = dest.stat().st_mtime_ns
    upload = tmp_path / "upload.tmp"
    upload.write_bytes(b"x" * 1000)
    gradio_fixes.move_uploaded_files_to_cache([str(upload)], [str(dest)])
    assert not upload.exists(), "временная копия убрана"
    assert dest.stat().st_mtime_ns == before, "файл назначения не тронут"


def test_a_new_file_is_moved_as_before(tmp_path):
    dest = tmp_path / "new.jpg"
    upload = tmp_path / "upload.tmp"
    upload.write_bytes(b"data")
    gradio_fixes.move_uploaded_files_to_cache([str(upload)], [str(dest)])
    assert dest.read_bytes() == b"data" and not upload.exists()


def test_a_truncated_leftover_is_replaced(tmp_path):
    """Обрывок прежней неудачной копии (другой размер) заменяется целым файлом."""
    dest = tmp_path / "photo.jpg"
    dest.write_bytes(b"x" * 10)
    upload = tmp_path / "upload.tmp"
    upload.write_bytes(b"x" * 1000)
    gradio_fixes.move_uploaded_files_to_cache([str(upload)], [str(dest)])
    assert dest.stat().st_size == 1000


def _read_while_replacing(folder, move) -> dict:
    """Опыт из разбора дефекта: файл заменяется повторной загрузкой, а его читают."""
    folder.mkdir()
    payload = os.urandom(12 * 1024 * 1024)
    dest = folder / "dest.bin"
    dest.write_bytes(payload)
    counts = {"ok": 0, "denied": 0}
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                with open(dest, "rb") as handle:
                    handle.read(16)
                counts["ok"] += 1
            except PermissionError:
                counts["denied"] += 1

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for index in range(12):
            upload = folder / f"upload{index}.tmp"
            upload.write_bytes(payload)
            try:
                os.rename(upload, dest)  # так пробует Gradio; под Windows — отказ
            except OSError:
                move([str(upload)], [str(dest)])
    finally:
        stop.set()
        thread.join()
    return counts


@pytest.mark.skipif(sys.platform != "win32", reason="блокировка CopyFile2 — только Windows")
def test_reading_during_a_repeated_upload_is_never_denied(tmp_path):
    control = _read_while_replacing(tmp_path / "control", ORIGINAL_MOVE)
    if control["denied"] == 0:
        pytest.skip("на этой машине исходный Gradio блокировку не показал — опыт ничего не различит")
    fixed = _read_while_replacing(tmp_path / "fixed", gradio_fixes.move_uploaded_files_to_cache)
    assert fixed["denied"] == 0, f"с поправкой: {fixed}; без неё: {control}"


# --- очистка временных файлов ---


def _blocks_with(path):
    return types.SimpleNamespace(blocks={}, temp_file_sets=[{str(path)}])


def _two_days_later(monkeypatch, module):
    real = datetime

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return real.now(tz) + timedelta(days=2)

    monkeypatch.setattr(module, "datetime", Later)


def test_a_two_day_old_file_is_deleted_at_a_one_day_age(tmp_path, monkeypatch):
    old = tmp_path / "old.png"
    old.write_bytes(b"x")
    # Контроль: исходная функция Gradio сравнивает .seconds (секунды внутри
    # суток) и файл двухсуточной давности при возрасте в сутки не удаляет.
    _two_days_later(monkeypatch, route_utils)
    ORIGINAL_DELETE(_blocks_with(old), 86400)
    assert old.exists(), "контроль: без поправки файл остаётся"

    _two_days_later(monkeypatch, gradio_fixes)
    blocks = _blocks_with(old)
    gradio_fixes.delete_files_created_by_app(blocks, 86400)
    assert not old.exists() and blocks.temp_file_sets == [set()]


def test_a_fresh_file_is_kept(tmp_path):
    fresh = tmp_path / "fresh.png"
    fresh.write_bytes(b"x")
    gradio_fixes.delete_files_created_by_app(_blocks_with(fresh), 86400)
    assert fresh.exists()


def test_a_busy_file_does_not_stop_the_cleanup(tmp_path, monkeypatch):
    """Занятый файл (Windows: открыт браузером или антивирусом) — пропускается."""
    busy, other = tmp_path / "busy.png", tmp_path / "other.png"
    busy.write_bytes(b"x")
    other.write_bytes(b"x")
    real_remove = os.remove

    def remove(path):
        if str(path) == str(busy):
            raise PermissionError(13, "busy")
        real_remove(path)

    monkeypatch.setattr(gradio_fixes.os, "remove", remove)
    blocks = types.SimpleNamespace(blocks={}, temp_file_sets=[{str(busy), str(other)}])
    gradio_fixes.delete_files_created_by_app(blocks, None)
    assert busy.exists() and not other.exists()
    assert blocks.temp_file_sets == [{str(busy)}], "занятый останется в списке до следующего прохода"


def test_shutil_is_what_gradio_uses():
    """Исходная функция всё ещё копирует через shutil.move — то, что обходит поправка."""
    import inspect

    assert "shutil.move" in inspect.getsource(ORIGINAL_MOVE)
    assert shutil.move is not None
