"""Библиотека поз: каталог openposes.com и позы, добавленные пользователем.

Поза — четыре файла с общим именем:

* ``<имя>.json`` — точки OpenPose (``skeleton.parse``);
* ``<имя>.png`` — скелет, тот самый, что ложится в ячейку референса;
* ``<имя>.jpg`` — плитка: иллюстрация героини в этой позе (1024×1024);
* ``<имя>.thumb.jpg`` — уменьшенная плитка для окна выбора.

**Каталог** (``config.POSE_LIBRARY_DIR``, ``resources/poses/catalog``) —
позы openposes.com с одной моделью, Эммой Уотсон; лежит в репозитории, как
стили и системные промты. Собран ``tools/fetch_poses.py``: архив скелетов и
точек ``poses.zip`` и плитки ``poses/emma_watson/jpg/<имя>.jpg`` из их
хранилища; тем же инструментом каталог обновляется.

**Свои позы** (``config.user_pose_dir()``, ``user/outputs/poses``) —
распознанные на фотографиях, данные пользователя, рядом с его генерациями.
Скелет сохраняется сразу, плитка — когда её нарисует Qwen-Image: до этого
в окне стоит сам скелет, и позой уже можно пользоваться.

**Имена и перерисованные обложки** — тоже данные пользователя, в
``user/outputs/poses/meta``: имена всех поз — ``titles.json``, обложки поз
каталога — ``covers/``, позы в схематичном виде — ``schematic.json``.
Каталог — часть поставки и в репозитории, поэтому его файлы не
переписываются: правки лежат поверх него. Свои позы перерисовывают
собственную обложку.

**Схематичный вид** — когда обложка не удалась (модель не держит позу), её
можно не показывать: вместо неё в окне стоит сам скелет. Обложка при этом
не удаляется — флажок снят, и она снова на месте.
"""

from __future__ import annotations

import io
import json
import logging
import time
import urllib.request
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import skeleton

LOGGER = logging.getLogger(__name__)

STORAGE = "https://openposes-storage.s3.ca-central-1.amazonaws.com"
ARCHIVE_URL = f"{STORAGE}/poses.zip"
MODEL = "emma_watson"
TILE_URL = STORAGE + "/poses/" + MODEL + "/jpg/{name}.jpg"

THUMB_SIDE = 320
CUSTOM_PREFIX = "custom_"
META_DIR = "meta"
TITLES_FILE = "titles.json"
SCHEMATIC_FILE = "schematic.json"
COVERS_DIR = "covers"
MAX_TITLE = 60


@dataclass(frozen=True)
class PoseEntry:
    name: str
    folder: Path
    custom: bool
    # Имя, заданное человеком (``titles.json``), или пустое — тогда в окне
    # стоит имя по умолчанию (см. ui/reference_tools.display_title).
    title: str = ""
    # Каталог перерисованных обложек поз каталога; у своих поз — None.
    covers: Path | None = None
    # Показывать скелет вместо обложки (``schematic.json``).
    schematic: bool = False

    @property
    def keypoints(self) -> Path:
        return self.folder / f"{self.name}.json"

    @property
    def skeleton(self) -> Path:
        return self.folder / f"{self.name}.png"

    @property
    def tile(self) -> Path:
        return self.folder / f"{self.name}.jpg"

    @property
    def thumb(self) -> Path:
        return self.folder / f"{self.name}.thumb.jpg"

    @property
    def cover_tile(self) -> Path:
        """Куда ложится перерисованная обложка: своя поза — на место своей
        плитки, поза каталога — в ``meta/covers`` поверх каталога."""
        return self.covers / f"{self.name}.jpg" if self.covers else self.tile

    @property
    def cover_thumb(self) -> Path:
        return self.covers / f"{self.name}.thumb.jpg" if self.covers else self.thumb

    def preview(self) -> Path:
        """Что показать в окне выбора: скелет в схематичном виде, иначе
        перерисованную обложку, плитку, а пока их нет — скелет."""
        if self.schematic:
            return self.skeleton
        for path in (self.cover_thumb, self.thumb):
            if path.exists():
                return path
        return self.skeleton


def _entries(folder: Path, custom: bool) -> list[PoseEntry]:
    if not folder.is_dir():
        return []
    names = sorted(path.stem for path in folder.glob("*.json"))
    entries = [PoseEntry(name, folder, custom) for name in names]
    return [entry for entry in entries if entry.skeleton.exists()]


def list_poses(catalog: Path, user: Path) -> list[PoseEntry]:
    """Каталог по имени, затем свои позы в порядке добавления — с именами и обложками."""
    titles = load_titles(user)
    schematic = load_schematic(user)
    covers = user / META_DIR / COVERS_DIR
    entries = [
        PoseEntry(entry.name, entry.folder, False, titles.get(entry.name, ""), covers, entry.name in schematic)
        for entry in _entries(catalog, custom=False)
    ]
    entries += [
        PoseEntry(entry.name, entry.folder, True, titles.get(entry.name, ""), None, entry.name in schematic)
        for entry in _entries(user, custom=True)
    ]
    return entries


def _titles_file(user: Path) -> Path:
    return user / META_DIR / TITLES_FILE


def load_titles(user: Path) -> dict[str, str]:
    """Имена поз, заданные человеком. Испорченный файл — «имён нет», а не сбой окна."""
    try:
        data = json.loads(_titles_file(user).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(key): str(value) for key, value in data.items()} if isinstance(data, dict) else {}


def set_title(user: Path, name: str, title: str) -> str:
    """Задаёт имя позе; пустое — вернуть имя по умолчанию. Возвращает записанное."""
    clean = " ".join(title.split())[:MAX_TITLE]
    titles = load_titles(user)
    if clean:
        titles[name] = clean
    else:
        titles.pop(name, None)
    _write_json(_titles_file(user), titles)
    return clean


def _write_json(path: Path, data) -> None:
    """Запись файла правок целиком: через ``.part``, чтобы сбой не оставил половину."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    partial.replace(path)


def _schematic_file(user: Path) -> Path:
    return user / META_DIR / SCHEMATIC_FILE


def load_schematic(user: Path) -> set[str]:
    """Позы в схематичном виде. Испорченный файл — «таких нет», а не сбой окна."""
    try:
        data = json.loads(_schematic_file(user).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {str(name) for name in data} if isinstance(data, list) else set()


def set_schematic(user: Path, name: str, on: bool) -> None:
    """Схематичный вид позы: вместо обложки — скелет; обложка не удаляется."""
    names = load_schematic(user)
    if on == (name in names):
        return
    names = names | {name} if on else names - {name}
    _write_json(_schematic_file(user), sorted(names))


def save_thumb(tile: Image.Image, destination: Path) -> None:
    thumb = tile.convert("RGB")
    thumb.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.Resampling.LANCZOS)
    thumb.save(destination, quality=88)


def _download(url: str, timeout: float = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Fooocus-Qwen-Image"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_catalog(
    catalog: Path,
    archive: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
    downloader: Callable[[str], bytes] = _download,
) -> int:
    """Скачивает недостающее в каталог; возвращает число поз в каталоге.

    ``archive`` — уже скачанный ``poses.zip`` (не качать его заново). Плитки
    качаются параллельно и только недостающие: прерванная загрузка
    продолжается, а не начинается сначала.
    """
    catalog.mkdir(parents=True, exist_ok=True)
    if not any(catalog.glob("*.json")):
        data = archive.read_bytes() if archive else downloader(ARCHIVE_URL)
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            for member in bundle.namelist():
                path = Path(member)
                if path.suffix in (".json", ".png") and path.parent == Path("."):
                    (catalog / path.name).write_bytes(bundle.read(member))

    entries = _entries(catalog, custom=False)
    missing = [entry for entry in entries if not entry.thumb.exists()]

    def one(entry: PoseEntry) -> None:
        if not entry.tile.exists():
            data = downloader(TILE_URL.format(name=entry.name))
            partial = entry.tile.with_suffix(".part")
            partial.write_bytes(data)
            partial.replace(entry.tile)
        save_thumb(Image.open(entry.tile), entry.thumb)

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(one, missing):
            done += 1
            if progress:
                progress(done, len(missing))
    return len(entries)


def add_custom(user: Path, pose: skeleton.Pose, title: str = "") -> PoseEntry:
    """Сохраняет распознанную позу (и имя, если задано); плитки у неё пока нет."""
    user.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name, suffix = f"{CUSTOM_PREFIX}{stamp}", 1
    while (user / f"{name}.json").exists():
        suffix += 1
        name = f"{CUSTOM_PREFIX}{stamp}_{suffix}"
    entry = PoseEntry(name, user, custom=True)
    skeleton.render(pose).save(entry.skeleton)
    entry.keypoints.write_text(pose.to_json(), encoding="utf-8")
    if title.strip():
        return PoseEntry(name, user, True, set_title(user, name, title))
    return entry


def set_tile(entry: PoseEntry, tile: Image.Image) -> None:
    """Обложка позы: своя — на место плитки, каталога — поверх каталога."""
    target, thumb = entry.cover_tile, entry.cover_thumb
    target.parent.mkdir(parents=True, exist_ok=True)
    tile.convert("RGB").save(target, quality=92)
    save_thumb(tile, thumb)
