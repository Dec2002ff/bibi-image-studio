from pathlib import Path

import pytest

from fooocus_qwen.bibi.backend import GenerationRequest, LoraSpec
from fooocus_qwen.bibi.manifest import ModelManifest
from fooocus_qwen.bibi.profiles import CharacterProfile, StyleProfile
from fooocus_qwen.bibi.storage import CharacterProfileStore, StyleProfileStore


def test_generation_request_limits_references():
    with pytest.raises(ValueError, match="at most 10"):
        GenerationRequest(prompt="x", references=[str(i) for i in range(11)])


def test_generation_request_knows_edit_mode():
    assert not GenerationRequest(prompt="x").is_edit
    assert GenerationRequest(prompt="x", references=["a.png"]).is_edit


def test_lora_weight_validation():
    with pytest.raises(ValueError, match="between -4 and 4"):
        LoraSpec(path="a.safetensors", weight=8)


def test_style_store_round_trip(tmp_path):
    path = tmp_path / "styles.json"
    store = StyleProfileStore(path)
    profile = StyleProfile.from_dict(
        {"id": "film", "name": "Film", "prompt_suffix": "35mm film"}
    )
    store.upsert(profile)
    loaded = store.load()
    assert loaded[0].id == "film"
    assert loaded[0].prompt_suffix == "35mm film"
    assert store.delete("film") is True
    assert store.load() == []


def test_character_store_round_trip(tmp_path):
    path = tmp_path / "characters.json"
    store = CharacterProfileStore(path)
    profile = CharacterProfile.from_dict(
        {"id": "hero", "name": "Hero", "reference_images": ["front.png"]}
    )
    store.upsert(profile)
    assert store.load()[0].reference_images == ["front.png"]


def test_model_manifest_loads_repository_manifest():
    root = Path(__file__).resolve().parents[1]
    manifest = ModelManifest.load(root / "config" / "model_manifest.json")
    qwen = manifest.by_id("qwen-image-2.1")
    assert qwen is not None
    assert qwen.required is True
