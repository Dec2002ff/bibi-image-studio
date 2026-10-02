"""Serializable profile models used by Bibi Image Studio.

These models deliberately use only the Python standard library.  The product
layer should be able to read a user's style and character library before torch,
diffusers, Gradio, or any model weights are loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ProfileError(ValueError):
    """Raised when a profile document is malformed."""


def _text(value: Any, field_name: str, *, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ProfileError(f"{field_name} must be a string")
    value = value.strip()
    if required and not value:
        raise ProfileError(f"{field_name} is required")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ProfileError(f"{field_name} must be a list of strings")
    return [item.strip() for item in value if item.strip()]


def _dict_list(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ProfileError(f"{field_name} must be a list of objects")
    return [dict(item) for item in value]


def _read_document(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ProfileError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ProfileError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ProfileError("profile document root must be an object")
    return raw


@dataclass(frozen=True)
class StyleDefaults:
    aspect_ratio: str | None = None
    quality: str | None = None
    true_cfg_scale: float | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> "StyleDefaults":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ProfileError("defaults must be an object")

        aspect = raw.get("aspect_ratio")
        quality = raw.get("quality")
        cfg = raw.get("true_cfg_scale")

        if aspect is not None and not isinstance(aspect, str):
            raise ProfileError("defaults.aspect_ratio must be a string")
        if quality is not None and not isinstance(quality, str):
            raise ProfileError("defaults.quality must be a string")
        if cfg is not None and not isinstance(cfg, (int, float)):
            raise ProfileError("defaults.true_cfg_scale must be a number")

        return cls(
            aspect_ratio=aspect.strip() if isinstance(aspect, str) and aspect.strip() else None,
            quality=quality.strip() if isinstance(quality, str) and quality.strip() else None,
            true_cfg_scale=float(cfg) if cfg is not None else None,
        )


@dataclass(frozen=True)
class StyleProfile:
    id: str
    name: str
    description: str = ""
    enabled: bool = True
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    negative_prompt: str = ""
    reference_images: list[str] = field(default_factory=list)
    loras: list[dict[str, Any]] = field(default_factory=list)
    defaults: StyleDefaults = field(default_factory=StyleDefaults)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Any) -> "StyleProfile":
        if not isinstance(raw, dict):
            raise ProfileError("style profile must be an object")
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ProfileError("enabled must be a boolean")
        return cls(
            id=_text(raw.get("id"), "id", required=True),
            name=_text(raw.get("name"), "name", required=True),
            description=_text(raw.get("description"), "description"),
            enabled=enabled,
            prompt_prefix=_text(raw.get("prompt_prefix"), "prompt_prefix"),
            prompt_suffix=_text(raw.get("prompt_suffix"), "prompt_suffix"),
            negative_prompt=_text(raw.get("negative_prompt"), "negative_prompt"),
            reference_images=_string_list(raw.get("reference_images"), "reference_images"),
            loras=_dict_list(raw.get("loras"), "loras"),
            defaults=StyleDefaults.from_dict(raw.get("defaults")),
            tags=_string_list(raw.get("tags"), "tags"),
        )

    def apply_prompt(self, prompt: str) -> str:
        """Compose a user prompt without silently dropping any part."""
        pieces = [self.prompt_prefix.strip(), prompt.strip(), self.prompt_suffix.strip()]
        return ", ".join(piece for piece in pieces if piece)


@dataclass(frozen=True)
class CharacterProfile:
    id: str
    name: str
    description: str = ""
    reference_images: list[str] = field(default_factory=list)
    identity_strength: float = 0.85
    default_style_profile: str | None = None
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Any) -> "CharacterProfile":
        if not isinstance(raw, dict):
            raise ProfileError("character profile must be an object")

        strength = raw.get("identity_strength", 0.85)
        if not isinstance(strength, (int, float)):
            raise ProfileError("identity_strength must be a number")
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ProfileError("identity_strength must be between 0 and 1")

        default_style = raw.get("default_style_profile")
        if default_style is not None and not isinstance(default_style, str):
            raise ProfileError("default_style_profile must be a string or null")

        return cls(
            id=_text(raw.get("id"), "id", required=True),
            name=_text(raw.get("name"), "name", required=True),
            description=_text(raw.get("description"), "description"),
            reference_images=_string_list(raw.get("reference_images"), "reference_images"),
            identity_strength=strength,
            default_style_profile=(
                default_style.strip()
                if isinstance(default_style, str) and default_style.strip()
                else None
            ),
            notes=_text(raw.get("notes"), "notes"),
            tags=_string_list(raw.get("tags"), "tags"),
        )


def _profiles_from_document(
    path: Path,
    parser: type[StyleProfile] | type[CharacterProfile],
) -> list[StyleProfile] | list[CharacterProfile]:
    raw = _read_document(path)
    version = raw.get("schema_version")
    if version != 1:
        raise ProfileError(f"unsupported schema_version: {version!r}")

    profiles = raw.get("profiles")
    if not isinstance(profiles, list):
        raise ProfileError("profiles must be a list")

    parsed = [parser.from_dict(item) for item in profiles]
    ids = [item.id for item in parsed]
    if len(ids) != len(set(ids)):
        raise ProfileError("profile ids must be unique")
    return parsed


def load_style_profiles(path: Path) -> list[StyleProfile]:
    return list(_profiles_from_document(path, StyleProfile))


def load_character_profiles(path: Path) -> list[CharacterProfile]:
    return list(_profiles_from_document(path, CharacterProfile))
