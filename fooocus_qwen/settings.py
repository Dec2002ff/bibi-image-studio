"""Настройки производительности: точность трансформера, профиль памяти, внимание.

Выбор делается при установке и меняется во вкладке «Настройки», поэтому
живёт в файле, а не в ключах командной строки: ключ пришлось бы помнить при
каждом запуске, а файл запоминает сам. Лежит в ``user/`` — это выбор
человека за машиной, а не проекта, и в репозиторий он не попадает.

Испорченный или устаревший файл — не повод ронять запуск: непонятное поле
заменяется значением по умолчанию, а причина уходит в журнал.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from . import config

LOGGER = logging.getLogger(__name__)

PRECISION_BF16 = "bf16"
PRECISION_INT8 = "int8"
# Варианты GGUF (Unsloth) — от точного к компактному. Q4_K_M — для 8 ГБ:
# с ним трансформер, VAE и активации кадра 1024² помещаются в карту
# (``docs/research/2026-10-02-8-gb.md``); Q4_K_S и Q3_K_M — запас на случай
# нехватки, Q5_K_M–Q8_0 — для 10–16 ГБ.
GGUF_VARIANTS: tuple[str, ...] = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q3_K_M")
PRECISION_GGUF_DEFAULT = "Q4_K_M"
PRECISIONS: tuple[str, ...] = (PRECISION_BF16, PRECISION_INT8, *GGUF_VARIANTS)

# Профиль памяти — как раскладываются веса (``engine/residency.py``):
# «high» — трансформер и bf16-энкодер меняются на карте местами (24 ГБ);
# «low» — трансформер не покидает карту, энкодер в int8 подаётся по блоку,
# крупный KV-кэш живёт в оперативной памяти (6–16 ГБ). «auto» — по объёму
# видеопамяти: от ``HIGH_PROFILE_MIN_GIB`` — «high», меньше — «low».
MEMORY_AUTO = "auto"
MEMORY_HIGH = "high"
MEMORY_LOW = "low"
MEMORY_PROFILES: tuple[str, ...] = (MEMORY_AUTO, MEMORY_HIGH, MEMORY_LOW)
HIGH_PROFILE_MIN_GIB = 20.0


def is_gguf(precision: str) -> bool:
    return precision in GGUF_VARIANTS


def resolve_profile(profile: str, vram_gib: float | None) -> str:
    """«auto» превращается в «high» или «low» по объёму видеопамяти.

    Без видеокарты (``vram_gib is None``) — «low»: это ближе к правде о машине,
    чем допущение 24 ГБ, и ничего не ломает там, где карта всё же есть.
    """
    if profile in (MEMORY_HIGH, MEMORY_LOW):
        return profile
    if vram_gib is not None and vram_gib >= HIGH_PROFILE_MIN_GIB:
        return MEMORY_HIGH
    return MEMORY_LOW


def recommended_precision(vram_gib: float | None) -> str:
    """Точность, которую установка предлагает по умолчанию для этой карты.

    Пороги — по весу трансформера плюс VAE, активациям кадра 1024² и
    контексту CUDA: bf16 нужны 24 ГБ, INT8 — от 12, Q8_0 — от 10, остальным
    — Q4_K_M.
    """
    if vram_gib is None or vram_gib < 10:
        return PRECISION_GGUF_DEFAULT
    if vram_gib < 12:
        return "Q8_0"
    if vram_gib < HIGH_PROFILE_MIN_GIB:
        return PRECISION_INT8
    return PRECISION_BF16


@dataclass(frozen=True)
class Settings:
    """Выбор пользователя.

    ``precision`` — в чём хранится трансформер: ``bf16`` (исходные веса,
    13.3 ГиБ видеопамяти), ``int8`` (7.3 ГиБ, веса Unsloth; качество —
    LPIPS 0.064 к bf16 по их замеру) или вариант GGUF (``Q4_K_M`` — 3.9 ГиБ).
    ``sage_attention`` — считать внимание SageAttention, если пакет
    установлен: −15…25 % времени шага
    (``docs/research/2026-09-24-uskorenie-turbo-sage-int8.md``).
    ``memory_profile`` — раскладка весов по памяти (см. ``MEMORY_PROFILES``).
    """

    precision: str = PRECISION_BF16
    sage_attention: bool = False
    memory_profile: str = MEMORY_AUTO


def load(path: Path | None = None) -> Settings:
    """Читает настройки; отсутствующий файл — значения по умолчанию."""
    path = path or config.SETTINGS_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Settings()
    except (OSError, ValueError) as error:
        LOGGER.warning("Settings file %s is unreadable (%s), using defaults", path, error)
        return Settings()
    if not isinstance(raw, dict):
        LOGGER.warning("Settings file %s is not an object, using defaults", path)
        return Settings()

    settings = Settings()
    precision = raw.get("precision", settings.precision)
    if precision in PRECISIONS:
        settings = replace(settings, precision=precision)
    else:
        LOGGER.warning("Unknown precision %r in %s, using %s", precision, path, settings.precision)
    sage = raw.get("sage_attention", settings.sage_attention)
    if isinstance(sage, bool):
        settings = replace(settings, sage_attention=sage)
    profile = raw.get("memory_profile", settings.memory_profile)
    if profile in MEMORY_PROFILES:
        settings = replace(settings, memory_profile=profile)
    else:
        LOGGER.warning("Unknown memory profile %r in %s, using %s", profile, path, settings.memory_profile)
    return settings


def save(settings: Settings, path: Path | None = None) -> None:
    """Записывает настройки целиком, через временный файл.

    Запись через замену: оборванная на середине запись оставила бы
    полуфайл, и следующий запуск молча вернулся бы к значениям по умолчанию.
    """
    if settings.precision not in PRECISIONS:
        raise ValueError(f"unknown precision: {settings.precision!r}")
    if settings.memory_profile not in MEMORY_PROFILES:
        raise ValueError(f"unknown memory profile: {settings.memory_profile!r}")
    path = path or config.SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def update(path: Path | None = None, **changes) -> Settings:
    """Меняет часть настроек и сохраняет."""
    settings = replace(load(path), **changes)
    save(settings, path)
    return settings
