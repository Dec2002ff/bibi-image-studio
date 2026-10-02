"""Bibi Image Studio product-layer primitives.

This package is intentionally independent from torch/Gradio so profile data and
backend contracts can be validated before a GPU or model weights are available.
"""

from .backend import GenerationRequest, GenerationResult, ImageBackend, LoraSpec
from .manifest import ModelEntry, ModelManifest
from .profiles import (
    CharacterProfile,
    ProfileError,
    StyleDefaults,
    StyleProfile,
    load_character_profiles,
    load_style_profiles,
)
from .storage import CharacterProfileStore, StyleProfileStore

__all__ = [
    "CharacterProfile",
    "CharacterProfileStore",
    "GenerationRequest",
    "GenerationResult",
    "ImageBackend",
    "LoraSpec",
    "ModelEntry",
    "ModelManifest",
    "ProfileError",
    "StyleDefaults",
    "StyleProfile",
    "StyleProfileStore",
    "load_character_profiles",
    "load_style_profiles",
]
