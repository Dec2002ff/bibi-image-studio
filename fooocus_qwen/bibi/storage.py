"""Small JSON stores for Bibi style and character profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .profiles import (
    CharacterProfile,
    ProfileError,
    StyleProfile,
    load_character_profiles,
    load_style_profiles,
)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _style_to_dict(profile: StyleProfile) -> dict[str, Any]:
    return {
        "id": profile.id,
        "name": profile.name,
        "description": profile.description,
        "enabled": profile.enabled,
        "prompt_prefix": profile.prompt_prefix,
        "prompt_suffix": profile.prompt_suffix,
        "negative_prompt": profile.negative_prompt,
        "reference_images": list(profile.reference_images),
        "loras": list(profile.loras),
        "defaults": {
            "aspect_ratio": profile.defaults.aspect_ratio,
            "quality": profile.defaults.quality,
            "true_cfg_scale": profile.defaults.true_cfg_scale,
        },
        "tags": list(profile.tags),
    }


def _character_to_dict(profile: CharacterProfile) -> dict[str, Any]:
    return {
        "id": profile.id,
        "name": profile.name,
        "description": profile.description,
        "reference_images": list(profile.reference_images),
        "identity_strength": profile.identity_strength,
        "default_style_profile": profile.default_style_profile,
        "notes": profile.notes,
        "tags": list(profile.tags),
    }


class StyleProfileStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[StyleProfile]:
        if not self.path.exists():
            return []
        return load_style_profiles(self.path)

    def save(self, profiles: list[StyleProfile]) -> None:
        ids = [profile.id for profile in profiles]
        if len(ids) != len(set(ids)):
            raise ProfileError("profile ids must be unique")
        _atomic_write(
            self.path,
            {"schema_version": 1, "profiles": [_style_to_dict(p) for p in profiles]},
        )

    def upsert(self, profile: StyleProfile) -> None:
        current = {item.id: item for item in self.load()}
        current[profile.id] = profile
        self.save(list(current.values()))

    def delete(self, profile_id: str) -> bool:
        profiles = self.load()
        kept = [item for item in profiles if item.id != profile_id]
        if len(kept) == len(profiles):
            return False
        self.save(kept)
        return True


class CharacterProfileStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[CharacterProfile]:
        if not self.path.exists():
            return []
        return load_character_profiles(self.path)

    def save(self, profiles: list[CharacterProfile]) -> None:
        ids = [profile.id for profile in profiles]
        if len(ids) != len(set(ids)):
            raise ProfileError("profile ids must be unique")
        _atomic_write(
            self.path,
            {"schema_version": 1, "profiles": [_character_to_dict(p) for p in profiles]},
        )

    def upsert(self, profile: CharacterProfile) -> None:
        current = {item.id: item for item in self.load()}
        current[profile.id] = profile
        self.save(list(current.values()))

    def delete(self, profile_id: str) -> bool:
        profiles = self.load()
        kept = [item for item in profiles if item.id != profile_id]
        if len(kept) == len(profiles):
            return False
        self.save(kept)
        return True
