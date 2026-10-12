import json

import pytest

from fooocus_qwen.bibi.profiles import (
    CharacterProfile,
    ProfileError,
    StyleProfile,
    load_character_profiles,
    load_style_profiles,
)


def test_style_profile_composes_prompt():
    profile = StyleProfile.from_dict(
        {
            "id": "cinematic",
            "name": "Cinematic",
            "prompt_prefix": "soft rim light",
            "prompt_suffix": "35mm film",
        }
    )
    assert profile.apply_prompt("portrait") == "soft rim light, portrait, 35mm film"


def test_character_strength_is_bounded():
    with pytest.raises(ProfileError, match="between 0 and 1"):
        CharacterProfile.from_dict(
            {"id": "hero", "name": "Hero", "identity_strength": 1.5}
        )


def test_style_document_requires_unique_ids(tmp_path):
    path = tmp_path / "styles.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profiles": [
                    {"id": "same", "name": "One"},
                    {"id": "same", "name": "Two"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProfileError, match="unique"):
        load_style_profiles(path)


def test_load_character_profiles(tmp_path):
    path = tmp_path / "characters.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profiles": [
                    {
                        "id": "meng-siming",
                        "name": "孟司嫇",
                        "reference_images": ["front.png", "side.png"],
                        "identity_strength": 0.9,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    profiles = load_character_profiles(path)
    assert profiles[0].name == "孟司嫇"
    assert profiles[0].reference_images == ["front.png", "side.png"]
    assert profiles[0].identity_strength == 0.9
