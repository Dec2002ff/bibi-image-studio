"""Профиль памяти «low»: сжатие, поблочная подача, KV-кэш в ОЗУ, GGUF, настройки.

Всё на крошечных модулях и без настоящих весов. Настоящая модель на карте 8
ГБ проверяется опытами (``tools/experiments/lowvram_*.py``).
"""

from __future__ import annotations

import json
import types

import pytest
import torch
from torch import nn

from fooocus_qwen import config
from fooocus_qwen import settings as settings_module
from fooocus_qwen.engine import gguf, quant, streaming, text_encoder
from fooocus_qwen.engine.residency import STREAM, SWAP, DeviceModule, ResidencyManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- сжатие int8 -----------------------------------------------------------------


def test_grouped_int8_round_trip_is_close_and_beats_rowwise():
    torch.manual_seed(0)
    weight = torch.randn(64, 256)
    weight[:, 3] *= 40  # выброс в одном входе: худший случай для масштаба на строку
    q, scale = quant.quantize(weight, group=128)
    assert q.dtype == torch.int8 and scale.shape == (64, 2)
    restored = quant.dequantize(q, scale, torch.float32)
    grouped = ((restored - weight).norm() / weight.norm()).item()
    rq, rscale = quant.quantize(weight, group=256)
    rowwise = ((quant.dequantize(rq, rscale, torch.float32) - weight).norm() / weight.norm()).item()
    assert grouped < 0.05
    assert grouped < rowwise


def test_int8_linear_matches_the_float_layer():
    torch.manual_seed(1)
    reference = nn.Linear(256, 32, bias=False)
    layer = quant.Int8Linear(256, 32)
    q, scale = quant.quantize(reference.weight.data)
    layer.weight.data, layer.scale.data = q, scale
    x = torch.randn(5, 256)
    assert torch.allclose(layer(x), reference(x), rtol=0.02, atol=0.02)


def test_int8_embedding_dequantizes_only_the_asked_rows():
    torch.manual_seed(2)
    table = torch.randn(10, 128)
    embedding = quant.Int8Embedding(10, 128, dtype=torch.float32)
    embedding.weight.data, embedding.scale.data = quant.quantize(table)
    ids = torch.tensor([[1, 7], [7, 0]])
    rows = embedding(ids)
    assert rows.shape == (2, 2, 128)
    assert torch.allclose(rows, table[ids], atol=0.03)


def test_assign_reports_missing_and_unexpected_names():
    with torch.device("meta"):
        module = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    problems = quant.assign(module, {"0.weight": torch.ones(4, 4), "2.weight": torch.ones(1)})
    assert "unexpected 2.weight" in problems and "missing 1.weight" in problems
    assert module[0].weight.device.type == "cpu"


# --- сборка int8-энкодера: свежесть копии ------------------------------------------


def _fake_source(directory, payload=b"x"):
    directory.mkdir()
    (directory / "a.safetensors").write_bytes(payload)
    (directory / text_encoder.INDEX).write_text(json.dumps({"weight_map": {"w": "a.safetensors"}}), encoding="utf-8")


def _fake_target(directory, source):
    directory.mkdir()
    (directory / "a.safetensors").write_bytes(b"q")
    manifest = {
        "format": text_encoder.FORMAT, "version": text_encoder.VERSION,
        "source": text_encoder._source_shards(source), "shards": ["a.safetensors"],
    }
    (directory / text_encoder.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


def test_a_built_copy_is_current_until_the_source_changes(tmp_path):
    source, target = tmp_path / "src", tmp_path / "dst"
    _fake_source(source)
    _fake_target(target, source)
    assert text_encoder.is_current(target, source)
    (source / "a.safetensors").write_bytes(b"another size")
    assert not text_encoder.is_current(target, source)


def test_a_copy_of_an_old_scheme_is_rebuilt(tmp_path):
    source, target = tmp_path / "src", tmp_path / "dst"
    _fake_source(source)
    _fake_target(target, source)
    manifest = json.loads((target / text_encoder.MANIFEST).read_text(encoding="utf-8"))
    manifest["version"] = text_encoder.VERSION - 1
    (target / text_encoder.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    assert not text_encoder.is_current(target, source)


def test_the_copy_works_without_the_bf16_shards(tmp_path):
    """bf16-шарды можно удалить ради места: сверять не с чем, копия рабочая."""
    source, target = tmp_path / "src", tmp_path / "dst"
    _fake_source(source)
    _fake_target(target, source)
    assert text_encoder.is_current(target, tmp_path / "deleted")


def test_without_bf16_weights_the_build_explains_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch-model"):
        text_encoder.ensure(tmp_path / "none", tmp_path / "dst")


def test_conversion_quantizes_linears_and_drops_the_head(tmp_path):
    from safetensors.torch import load_file, save_file

    source, target = tmp_path / "src", tmp_path / "dst"
    source.mkdir()
    tensors = {
        "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(8, 128, dtype=torch.bfloat16),
        "model.language_model.embed_tokens.weight": torch.randn(4, 128, dtype=torch.bfloat16),
        "model.language_model.norm.weight": torch.ones(128, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(4, 128, dtype=torch.bfloat16),
    }
    save_file(tensors, str(source / "s.safetensors"))
    (source / text_encoder.INDEX).write_text(json.dumps({"weight_map": dict.fromkeys(tensors, "s.safetensors")}))
    text_encoder.convert(source, target, device="cpu", out=lambda _line: None)
    written = load_file(str(target / "s.safetensors"))
    assert written["model.language_model.layers.0.mlp.down_proj.weight"].dtype == torch.int8
    assert written["model.language_model.layers.0.mlp.down_proj.scale"].shape == (8, 1)
    assert written["model.language_model.embed_tokens.weight"].dtype == torch.int8
    assert written["model.language_model.norm.weight"].dtype == torch.bfloat16
    assert "lm_head.weight" not in written
    assert text_encoder.is_current(target, source)


# --- поблочная подача ----------------------------------------------------------------


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.seen: list[str] = []

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return x


def test_streaming_lifts_each_block_only_for_its_call():
    model = Tiny()
    expected = model(torch.ones(1, 4)).detach()
    model.seen.clear()
    staged = streaming.StreamedModule(model, model.blocks, DEVICE)
    staged.to_device()
    for block in model.blocks:
        # Хук после хука подъёма: видит то, что видит forward блока.
        block.register_forward_pre_hook(lambda module, _args: model.seen.append(module.weight.device.type))
    assert model.embed.weight.device.type == DEVICE
    assert all(block.weight.device.type == "cpu" for block in model.blocks)
    result = model(torch.ones(1, 4, device=DEVICE))
    assert model.seen == [DEVICE] * 3
    assert all(block.weight.device.type == "cpu" for block in model.blocks)
    staged.to_host()
    assert model.embed.weight.device.type == "cpu"
    assert torch.allclose(result.cpu(), expected, atol=1e-5)


def test_streaming_never_copies_host_tensors():
    """Веса энкодера — страницы файла; копия на хосте удвоила бы оперативную память."""
    model = Tiny()
    pointer = model.blocks[1].weight.data_ptr()
    staged = streaming.StreamedModule(model, model.blocks, DEVICE)
    staged.to_device()
    model(torch.ones(1, 4, device=DEVICE))
    staged.to_host()
    assert model.blocks[1].weight.data_ptr() == pointer


def test_a_failure_inside_a_block_leaves_nothing_on_the_device():
    model = Tiny()
    staged = streaming.StreamedModule(model, model.blocks, DEVICE)

    def boom(_module, _args, _output):
        raise RuntimeError("out of memory")

    model.blocks[1].register_forward_hook(boom)
    staged.to_device()
    with pytest.raises(RuntimeError):
        model(torch.ones(1, 4, device=DEVICE))
    staged.to_host()
    assert all(p.device.type == "cpu" for p in model.parameters())
    model.blocks[1]._forward_hooks.clear()
    model(torch.ones(1, 4))  # хуков подачи не осталось — модуль считает на хосте


def test_host_modules_stay_on_the_host_and_still_compute():
    torch.manual_seed(3)
    table = torch.randn(10, 128)
    embedding = quant.Int8Embedding(10, 128, dtype=torch.float32)
    embedding.weight.data, embedding.scale.data = quant.quantize(table)
    model = nn.Module()
    model.embed = embedding
    model.blocks = nn.ModuleList([nn.Linear(128, 128)])
    staged = streaming.StreamedModule(model, model.blocks, DEVICE, host_modules=[embedding])
    staged.to_device()
    assert embedding.weight.device.type == "cpu"
    rows = embedding(torch.tensor([1, 2], device=DEVICE))
    assert rows.device.type == DEVICE
    staged.to_host()


def test_blocks_must_belong_to_the_module():
    with pytest.raises(ValueError):
        streaming.StreamedModule(Tiny(), [nn.Linear(4, 4)], DEVICE)


# --- KV-кэш в оперативной памяти ------------------------------------------------------


class FakeTransformer(nn.Module):
    def forward(self, x, kv_cache=None):
        layer = kv_cache.get_layer(0)
        if layer.k is None:
            layer.store(x, x * 2)
        return layer.get()


def test_a_large_kv_cache_goes_to_the_host_and_comes_back():
    transformer = FakeTransformer()
    handle = streaming.offload_kv_cache(transformer, torch.device(DEVICE), threshold_bytes=0)
    original = types.SimpleNamespace(layer_caches=[object()] * 2)
    x = torch.ones(3, device=DEVICE)
    k, v = transformer(x, kv_cache=original)
    replacement = original._studio_host
    assert replacement.offloaded
    assert replacement.layer_caches[0].k.device.type == "cpu"
    assert k.device.type == DEVICE and torch.equal(v.cpu(), torch.full((3,), 2.0))
    # Второй шаг того же вызова пайплайна — та же замена, а не новая пустая.
    transformer(x, kv_cache=original)
    assert original._studio_host is replacement
    handle.remove()


def test_a_small_kv_cache_stays_on_the_device():
    transformer = FakeTransformer()
    streaming.offload_kv_cache(transformer, torch.device(DEVICE), threshold_bytes=2**30)
    original = types.SimpleNamespace(layer_caches=[object()])
    transformer(torch.ones(3, device=DEVICE), kv_cache=original)
    assert not original._studio_host.offloaded
    assert original._studio_host.layer_caches[0].k.device.type == DEVICE


# --- политика STREAM ----------------------------------------------------------------


def _stream_layout(monkeypatch):
    monkeypatch.setattr(streaming, "text_encoder_blocks", lambda encoder: list(encoder.blocks))
    monkeypatch.setattr(streaming, "text_encoder_host_modules", lambda encoder: [])


def _pipe():
    encoder = Tiny()
    return types.SimpleNamespace(transformer=nn.Linear(4, 4), text_encoder=encoder, vae=nn.Linear(4, 4))


def test_stream_policy_keeps_the_transformer_and_streams_the_encoder(monkeypatch):
    pipe = _pipe()
    _stream_layout(monkeypatch)
    manager = ResidencyManager(pipe, DEVICE, pin_memory=False, policy=STREAM)
    manager.start()
    assert pipe.transformer.weight.device.type == DEVICE
    assert pipe.vae.weight.device.type == DEVICE
    assert pipe.text_encoder.embed.weight.device.type == "cpu"
    with manager.text_encoder_resident():
        # Трансформер не выгоняется: на 8 ГБ места хватает по слою энкодера.
        assert pipe.transformer.weight.device.type == DEVICE
        assert pipe.text_encoder.embed.weight.device.type == DEVICE
        if DEVICE == "cuda":
            # VAE уступает место энкодеру и возвращается после.
            assert pipe.vae.weight.device.type == "cpu"
        pipe.text_encoder(torch.ones(1, 4, device=DEVICE))
    assert pipe.text_encoder.embed.weight.device.type == "cpu"
    assert pipe.vae.weight.device.type == DEVICE
    assert manager.stats()["swaps"] == 1


def test_stream_policy_restages_on_the_device(monkeypatch):
    pipe = _pipe()
    _stream_layout(monkeypatch)
    manager = ResidencyManager(pipe, DEVICE, pin_memory=False, policy=STREAM)
    manager.start()

    def attach(transformer):
        transformer.extra = nn.Linear(4, 4)

    manager.restage_transformer(attach)
    assert pipe.transformer.extra.weight.device.type == DEVICE


def test_device_module_is_always_resident():
    module = DeviceModule(nn.Linear(4, 4), DEVICE)
    module.to_device()
    module.to_host()
    assert module.resident and module.module.weight.device.type == DEVICE
    assert module.nbytes == (16 + 4) * 4


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError):
        ResidencyManager(_pipe(), DEVICE, policy="magic")
    assert SWAP != STREAM


# --- GGUF ----------------------------------------------------------------------------


def test_comfy_names_become_diffusers_names():
    assert gguf.diffusers_name("model.diffusion_model.img_in.weight") == "img_in.weight"
    assert gguf.diffusers_name("img_in.weight") == "img_in.weight"


def test_gate_up_splits_by_rows_gate_first():
    stem = "transformer_blocks.3."
    fused = torch.arange(8).reshape(4, 2)
    parts = gguf.split_gate_up(stem + gguf.FUSED_GATE_UP, fused)
    assert torch.equal(parts[stem + gguf.GATE], fused[:2])
    assert torch.equal(parts[stem + gguf.PROJ], fused[2:])


def test_gate_up_with_odd_rows_is_refused():
    with pytest.raises(ValueError):
        gguf.split_gate_up("x." + gguf.FUSED_GATE_UP, torch.zeros(3, 2))


# --- настройки и профиль ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile", "vram", "expected"),
    [("auto", 8.0, "low"), ("auto", 24.0, "high"), ("auto", None, "low"), ("high", 8.0, "high"), ("low", 24.0, "low")],
)
def test_profile_resolution(profile, vram, expected):
    assert settings_module.resolve_profile(profile, vram) == expected


def test_gguf_precisions_round_trip_and_bad_profiles_fall_back():
    settings_module.save(settings_module.Settings("Q4_K_M", False, "low"))
    assert settings_module.load() == settings_module.Settings("Q4_K_M", False, "low")
    assert settings_module.is_gguf("Q4_K_M") and not settings_module.is_gguf("int8")
    config.SETTINGS_FILE.write_text(json.dumps({"precision": "Q4_K_M", "memory_profile": "tiny"}), encoding="utf-8")
    assert settings_module.load().memory_profile == settings_module.MEMORY_AUTO
    with pytest.raises(ValueError):
        settings_module.save(settings_module.Settings("bf16", False, "tiny"))


def test_the_strong_card_path_is_untouched():
    """Профиль «high» и bf16 остаются тем, чем были: расширение, а не замена."""
    assert "bf16" in settings_module.PRECISIONS and "int8" in settings_module.PRECISIONS
    assert settings_module.Settings().precision == "bf16"
    assert settings_module.recommended_precision(24.0) == "bf16"
    assert settings_module.resolve_profile("auto", 24.0) == "high"


def test_read_only_mapping_reads_what_safetensors_wrote(tmp_path):
    from safetensors.torch import save_file

    tensors = {
        "a": torch.randn(3, 5, dtype=torch.bfloat16),
        "b": torch.randint(-127, 127, (4, 8), dtype=torch.int8),
        "c": torch.rand(7, dtype=torch.float32),
        "empty": torch.empty(0, 4),
    }
    save_file(tensors, str(tmp_path / "t.safetensors"))
    loaded = text_encoder.read_mapped(tmp_path / "t.safetensors")
    assert set(loaded) == set(tensors)
    for name, tensor in tensors.items():
        assert loaded[name].dtype == tensor.dtype and torch.equal(loaded[name], tensor)



# --- подмена трансформера (Turbo4) ------------------------------------------------------


def test_swap_policy_parks_the_main_transformer_and_brings_it_back(monkeypatch):
    pipe = _pipe()
    main = pipe.transformer
    manager = ResidencyManager(pipe, DEVICE, pin_memory=False, policy=SWAP)
    manager.start()
    parked = manager.replace_transformer(lambda device: nn.Linear(4, 4))
    assert pipe.transformer is not main and pipe.transformer.weight.device.type == DEVICE
    assert parked is not None and main.weight.device.type == "cpu"
    loaded = []
    manager.replace_transformer(lambda device: loaded.append(device) or nn.Linear(4, 4), parked=parked)
    assert pipe.transformer is main and main.weight.device.type == DEVICE
    assert loaded == []  # с хоста, без чтения с диска


def test_stream_policy_releases_the_main_transformer_and_reloads_it(monkeypatch):
    pipe = _pipe()
    _stream_layout(monkeypatch)
    manager = ResidencyManager(pipe, DEVICE, pin_memory=False, policy=STREAM)
    manager.start()
    devices = []

    def build(device):
        devices.append(device)
        return nn.Linear(4, 4).to(device)

    parked = manager.replace_transformer(build)
    assert parked is None and devices == [torch.device(DEVICE)]
    manager.replace_transformer(build)
    assert len(devices) == 2 and pipe.transformer.weight.device.type == DEVICE
    with manager.text_encoder_resident():
        assert pipe.transformer.weight.device.type == DEVICE


class _Recorder:
    def __init__(self):
        self.calls = []

    def activate(self, enabled):
        self.calls.append(("activate", enabled))

    def reset(self):
        self.calls.append(("reset",))


def test_turbo4_switches_scheduler_and_resets_the_adapter_after_a_reload(monkeypatch, tmp_path):
    from fooocus_qwen.engine import fetch, turbo

    gguf_dir, turbo_dir = tmp_path / "gguf", tmp_path / "turbo"
    (gguf_dir).mkdir()
    (gguf_dir / fetch.TURBO4_FILE).write_bytes(b"x")
    (turbo_dir / "scheduler").mkdir(parents=True)
    (turbo_dir / fetch.TURBO_SCHEDULER).write_text("{}", encoding="utf-8")
    monkeypatch.setattr("diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained", lambda *a, **k: "turbo-scheduler")
    monkeypatch.setattr(gguf, "load_transformer", lambda *a, **k: nn.Linear(4, 4))

    pipe = _pipe()
    pipe.scheduler = "base-scheduler"
    _stream_layout(monkeypatch)
    manager = ResidencyManager(pipe, DEVICE, pin_memory=False, policy=STREAM)
    manager.start()
    prepared, adapter = [], _Recorder()
    switch = turbo.Turbo4Transformer(
        pipe, manager, gguf_dir / fetch.TURBO4_FILE, turbo_dir, tmp_path,
        base_loader=lambda device: nn.Linear(4, 4).to(device), prepare=prepared.append, turbo=adapter,
    )
    switch.activate(True)
    assert switch.active and pipe.scheduler == "turbo-scheduler"
    assert adapter.calls == [("activate", False)]
    switch.activate(True)  # повтор — ничего не меняет
    switch.activate(False)
    assert not switch.active and pipe.scheduler == "base-scheduler"
    assert adapter.calls[-1] == ("reset",)
    assert len(prepared) == 2


def test_turbo4_without_weights_says_so(tmp_path):
    from fooocus_qwen.engine import turbo

    switch = turbo.Turbo4Transformer(None, None, tmp_path / "none.gguf", tmp_path, tmp_path, None, None)
    assert not switch.weights_present()
    with pytest.raises(FileNotFoundError):
        switch.activate(True)
