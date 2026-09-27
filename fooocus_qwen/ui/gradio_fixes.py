"""Две поправки к Gradio 6.5.1 (версия закреплена в requirements.txt).

**Повторная загрузка того же файла роняла событие.** Gradio кладёт загрузку
в каталог, названный хешем её содержимого (``uploaded_file_dir/<sha>/<имя>``),
и переименовывает временный файл туда. Под Windows ``os.rename`` поверх
существующего файла падает, и Gradio, уже вернув браузеру путь, копирует
файл поверх фоном (``shutil.move``). В Python 3.12 под Windows копирование
идёт через ``CopyFile2``, который держит файл назначения без права чтения, —
а браузер тем временем шлёт событие поля, и Gradio, открывая этот же файл,
получает ``PermissionError`` (у пользователя: картинка 15.5 МБ, повторно
положенная в ячейку референса). Опыт: пока ``shutil.move`` 20 раз копирует
15 МБ поверх файла, его чтение падает 5863 раза против 504 удачных.

Копировать при этом нечего: каталог назван хешем содержимого, так что файл с
тем же именем в нём — побайтно тот же. Поправка оставляет его на месте и
убирает временную копию.

**Очистка временных файлов не работала.** ``delete_cache=(частота, возраст)``
сравнивает возраст как ``(сейчас - создание).seconds > возраст``, а
``.seconds`` у интервала — только секунды внутри суток (0–86399). С
возрастом сутки (как у нас) условие не выполнялось никогда, и каталог
загрузок рос без конца. Поправка сравнивает полную длительность.

Обе ставятся один раз, до запуска сервера (``install``), и проверяют, что
заменяют именно то, что ожидают: сменится Gradio — поправка сообщит об этом
в журнал и ничего не тронет, а не подменит чужой код молча.
"""

from __future__ import annotations

import inspect
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from gradio import route_utils, routes

LOGGER = logging.getLogger(__name__)
_installed = False

# Исходные функции Gradio — до подмены. Нужны проверкам (контроль: на них
# дефект виден) и не должны зависеть от того, собирался ли уже интерфейс.
ORIGINAL_MOVE = route_utils.move_uploaded_files_to_cache
ORIGINAL_DELETE = route_utils.delete_files_created_by_app


def move_uploaded_files_to_cache(files: list[str], destinations: list[str]) -> None:
    """Как у Gradio, но одинаковое содержимое не копируется поверх себя."""
    for file, dest in zip(files, destinations, strict=False):
        if os.path.exists(dest) and os.path.getsize(dest) == os.path.getsize(file):
            try:
                os.remove(file)
            except OSError:
                LOGGER.warning("Could not remove the temporary upload %s", file)
            continue
        shutil.move(file, dest)


def delete_files_created_by_app(blocks, age: int | None) -> None:
    """Как у Gradio, но возраст файла — полная длительность, а не секунды внутри суток."""
    dont_delete = set()
    for component in blocks.blocks.values():
        dont_delete.update(getattr(component, "keep_in_cache", set()))
    for temp_set in blocks.temp_file_sets:
        to_remove = set()
        for file in list(temp_set):
            if file in dont_delete:
                continue
            try:
                created = datetime.fromtimestamp(Path(file).lstat().st_ctime)
                if age is None or (datetime.now() - created).total_seconds() > age:
                    os.remove(file)
                    to_remove.add(file)
            except FileNotFoundError:
                to_remove.add(file)
            except OSError as error:
                # Файл занят (открыт браузером или антивирусом) — удалится в
                # следующий проход, а не уронит весь цикл очистки.
                LOGGER.debug("Temporary file %s not removed yet: %s", file, error)
        temp_set -= to_remove


def install() -> None:
    """Подменяет обе функции в модулях Gradio, которые их вызывают."""
    global _installed
    if _installed:
        return
    expected = {
        ORIGINAL_MOVE: "shutil.move(file, dest)",
        ORIGINAL_DELETE: ".seconds > age",
    }
    for function, marker in expected.items():
        if marker not in inspect.getsource(function):
            LOGGER.warning("Gradio changed %s; leaving it as is (fix not applied)", function.__name__)
            return
    route_utils.move_uploaded_files_to_cache = move_uploaded_files_to_cache
    routes.move_uploaded_files_to_cache = move_uploaded_files_to_cache
    route_utils.delete_files_created_by_app = delete_files_created_by_app
    _installed = True
