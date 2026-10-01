r"""Опыт: профиль «low» на карте 8 ГБ — время и память по режимам работы.

Всё идёт через ``Studio`` — ту же дорогу, что у интерфейса, — с настройками
из ``user/settings.json`` (точность, профиль памяти, SageAttention). Режимы:

* ``t2i``      — текст в картинку, LowQuality (1024², 16 шагов);
* ``t2i-25``   — то же, 25 шагов (рекомендация ComfyUI для полного качества);
* ``turbo``    — пресет Turbo (адаптер r128, 6 шагов);
* ``draft``    — TurboDraft (768², 6 шагов);
* ``edit``     — правка без маски с исходником 1024²: KV-кэш префикса 2+ ГиБ,
  в профиле «low» он уезжает в оперативную память;
* ``refs2``    — два референса (масштаб 768 по правилу числа условий);
* ``middle``   — MiddleQuality (1536², 28 шагов) — помещается ли вовсе.

Меры: секунды на кадр (кодирование промта входит при первом прогоне на промт —
он здесь всегда первый: у каждого режима свой промт), секунды на шаг
денойзинга, пик выделенной видеопамяти, зарезервированная, объём KV-кэша и
куда он лёг. Картинки — в ``logs/lowvram/`` для осмотра глазами.

Запуск (необязательный аргумент — имена режимов через запятую):
    .venv\Scripts\python tools\experiments\lowvram_profile.py [t2i,edit]
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402

use_utf8_console()

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import presets  # noqa: E402
from fooocus_qwen.engine.generator import GenerationRequest  # noqa: E402
from fooocus_qwen.engine.streaming import HostKVCache  # noqa: E402
from fooocus_qwen.imaging import aspect  # noqa: E402
from fooocus_qwen.ui.state import Studio  # noqa: E402

OUT = ROOT / "logs" / "lowvram"
SOURCE = OUT / "t2i_low_q4km.png"


def cases():
    source = Image.open(SOURCE).convert("RGB") if SOURCE.is_file() else Image.new("RGB", (1024, 1024), "gray")
    pose = Image.open(ROOT / "resources" / "poses" / "add_pose.png").convert("RGB")
    return {
        "t2i": GenerationRequest(
            prompt="a cozy reading nook by a rainy window, warm lamp light, film photo", preset=presets.get("LowQuality"),
        ),
        "t2i-25": GenerationRequest(
            prompt="a lighthouse on black rocks at dusk, crashing waves, long exposure",
            preset=presets.QualityPreset("LowQuality25", output_resolution=1024, num_inference_steps=25),
        ),
        "turbo": GenerationRequest(prompt="a bowl of ramen, top view, studio light", preset=presets.get("Turbo")),
        "draft": GenerationRequest(prompt="a paper boat on a puddle, macro", preset=presets.get("TurboDraft")),
        "edit": GenerationRequest(
            prompt="make it a snowy winter evening, keep the tram and the headline", preset=presets.get("LowQuality"),
            source=source, aspect=aspect.FOLLOW_REFERENCE,
        ),
        "refs2": GenerationRequest(
            prompt="the tram from <image1> drawn in the style of <image2>", preset=presets.get("LowQuality"),
            references=(source, pose),
        ),
        "middle": GenerationRequest(
            prompt="an alpine lake with a wooden pier, morning mist", preset=presets.get("MiddleQuality"),
        ),
    }


def main() -> None:
    wanted = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    OUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("LOWVRAM_CAP") == "1":
        # Потолок аллокатора — физически свободная память: при нехватке он
        # освобождает свой кэш, а не просит у драйвера общую память системы.
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction((free - 256 * 2**20) / total)
        print(f"allocator capped at {(free - 256 * 2**20) / 2**30:.2f} GiB")
    studio = Studio(config.AppConfig(preload=False))
    generator = studio.generator

    kv_seen: list[HostKVCache] = []

    def watch(_module, _args, kwargs):
        cache = kwargs.get("kv_cache")
        if isinstance(cache, HostKVCache) and (not kv_seen or kv_seen[-1] is not cache):
            kv_seen.append(cache)

    # Пик кодирования промта отдельно от пика денойзинга: у них разные
    # хозяева памяти (энкодер по блоку против активаций трансформера).
    encode_peaks: list[float] = []
    residency = generator._residency
    original_resident = residency.text_encoder_resident

    @contextlib.contextmanager
    def measured():
        torch.cuda.reset_peak_memory_stats()
        with original_resident():
            yield
        encode_peaks.append(torch.cuda.max_memory_allocated() / 2**30)
        torch.cuda.reset_peak_memory_stats()

    residency.text_encoder_resident = measured

    # После подмены (её хук стоит раньше) трансформер видит HostKVCache.
    generator.pipe.transformer.register_forward_pre_hook(watch, with_kwargs=True)

    print(f"{'case':7s} {'s/img':>7s} {'s/step':>7s} {'enc':>6s} {'denoise':>7s} {'resvd':>6s}  kv")
    for name, request in cases().items():
        if wanted and name not in wanted:
            continue
        studio.weights_for(request.preset, "en")
        steps = []

        def progress(_index, step, _total, _steps=steps):
            _steps.append(time.perf_counter())

        encode_peaks.clear()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        results, failure = studio.run_generation(request, "en", progress=progress)
        seconds = time.perf_counter() - started
        if failure:
            print(f"{name:7s} FAILED: {failure}")
            continue
        per_step = (steps[-1] - steps[0]) / max(1, len(steps) - 1)
        peak = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        kv = "-"
        if kv_seen and kv_seen[-1].layer_caches[0].k is not None:
            cache = kv_seen[-1]
            layer = cache.layer_caches[0]
            size = (layer.k.nbytes + layer.v.nbytes) * len(cache.layer_caches) / 2**30
            kv = f"{size:.2f} GiB on {'host' if cache.offloaded else 'device'}"
        results[0].image.save(OUT / f"profile_{name}.png")
        encode = f"{encode_peaks[0]:6.2f}" if encode_peaks else "  hit "
        print(f"{name:7s} {seconds:7.1f} {per_step:7.2f} {encode} {peak:7.2f} {reserved:6.2f}  {kv}")


if __name__ == "__main__":
    main()
