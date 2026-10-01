"""Что за видеокарта: объём её памяти решает профиль размещения весов."""

from __future__ import annotations


def vram_gib(device: int = 0) -> float | None:
    """Полный объём видеопамяти в ГиБ или ``None``, если CUDA нет."""
    try:
        import torch
    except ImportError:  # pragma: no cover — без torch движок не работает
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(device).total_memory / 2**30
