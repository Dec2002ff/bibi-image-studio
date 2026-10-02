"""Bibi Image Studio product-layer primitives.

This package is intentionally independent from torch/Gradio so profile data can
be validated and tested before a GPU or model weights are available.
"""

from .profiles import (
    CharacterProfile,
    ProfileError,
    StyleDefaults,
    StyleProfile,
    load_character_profiles,
    load_style_profiles,
)

__all__ = [
    "CharacterProfile",
    "ProfileError",
    "StyleDefaults",
    "StyleProfile",
    "load_character_profiles",
    "load_style_profiles",
]
