"""Выбор качества: пресеты понятными словами, от быстрого к лучшему.

Имена пресетов (``LowQuality``, ``TurboDraft``…) — идентификаторы: они
пишутся в метаданные картинок и в сохранённые промты и не меняются. На
экране же нужно другое — что выбор даёт: сколько шагов, какой кадр, для
чего он. Числа берутся из самих пресетов (``engine/presets.py``), так что
подпись не разойдётся с тем, что считается.

Порядок — по скорости: сначала дистилляты, затем полная модель по
возрастанию. ``presets.NAMES`` остаётся прежним — это порядок ключа
``--preset`` и метаданных.
"""

from __future__ import annotations

from ..engine import presets

ORDER: tuple[str, ...] = ("TurboDraft", "Turbo", "Turbo4", "LowQuality", "MiddleQuality", "MaxQuality")

# (ru, en): как назвать и, если нужно, зачем выбирать.
_TITLES: dict[str, tuple[str, str]] = {
    "TurboDraft": ("Черновик", "Draft"),
    "Turbo": ("Turbo", "Turbo"),
    "Turbo4": ("Turbo4", "Turbo4"),
    "LowQuality": ("Низкое", "Low"),
    "MiddleQuality": ("Среднее", "Medium"),
    "MaxQuality": ("Максимальное, 2K", "Maximum, 2K"),
}
_NOTES: dict[str, tuple[str, str]] = {
    "TurboDraft": ("для перебора промтов и сидов", "for trying prompts and seeds"),
    "Turbo": ("быстрый дистиллят", "fast distilled model"),
    "Turbo4": ("быстрее всего на 8 ГБ, правка слабее", "fastest on 8 GB, weaker at editing"),
}


def _steps(count: int, lang: str) -> str:
    if lang != "ru":
        return f"{count} steps"
    tail = count % 100
    if 11 <= tail <= 14 or count % 10 in (0, 5, 6, 7, 8, 9):
        return f"{count} шагов"
    return f"{count} шаг" if count % 10 == 1 else f"{count} шага"


def label(name: str, lang: str) -> str:
    """«Среднее — 28 шагов, 1536 px»; у дистиллятов — ещё зачем."""
    preset = presets.get(name)
    index = 0 if lang == "ru" else 1
    text = f"{_TITLES[name][index]} — {_steps(preset.num_inference_steps, lang)}, {preset.output_resolution} px"
    note = _NOTES.get(name)
    return f"{text}; {note[index]}" if note else text


def choices(lang: str) -> list[tuple[str, str]]:
    """Варианты для ``gr.Radio``: подпись на языке интерфейса, значение — имя пресета."""
    return [(label(name, lang), name) for name in ORDER]
