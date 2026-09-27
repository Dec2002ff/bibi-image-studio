r"""Опыт: какие рычаги скорости что дают на этой карте.

Профиль (`tools/profile_pipeline.py`) показал, куда уходит время правки по
маске на среднем пресете: 82.6 % — шаги трансформера, 14.7 % —
декодирование VAE, всё остальное вместе меньше двух процентов. Значит
оптимизировать имеет смысл ровно две вещи, и здесь перебираются рычаги для
каждой.

Рычаги проверяются **по одному**, от общей отправной точки, и каждый
меряется отдельно на денойзинге и на декодировании: они упираются в разное
(внимание против свёрток), и рычаг, полезный одному, другому может быть
безразличен.

Чего здесь нет и почему:

- ``torch.compile`` — на Windows инструктору нужен Triton, а его в
  окружении нет и ставить его ради этого пришлось бы отдельной сборкой;
- SageAttention и flash-attn — не установлены; на Ampere (SM 8.6) у
  SageAttention работает только путь int8/fp16, которому тоже нужен Triton.

Остаются родные механизмы PyTorch и два флага, которые просто выключены по
умолчанию.

Запуск (из корня проекта):
    .venv\Scripts\python tools\experiments\speed_levers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fooocus_qwen.logging_setup import use_utf8_console

use_utf8_console()

import argparse
import json
import time

import torch
from PIL import Image

from fooocus_qwen import config, logging_setup
from fooocus_qwen.engine import loader

OUT = config.LOG_DIR / "speed-levers"
PROMPT = "a wooden desk with a brass lamp and a cup of tea, warm afternoon light"


def timed(function, *args, **kwargs) -> tuple[object, float]:
    """Вызов с синхронизацией: без неё ядра возвращаются раньше, чем считают."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function(*args, **kwargs)
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


@torch.no_grad()
def decode(pipe, latents, height: int, width: int):
    """Хвост пайплайна для готового латента (строки 828-842)."""
    unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    unpacked = unpacked.to(pipe.vae.dtype)
    shape = (1, pipe.vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(shape).to(unpacked.device, unpacked.dtype)
    std = torch.tensor(pipe.vae.config.latents_std).view(shape).to(unpacked.device, unpacked.dtype)
    decoded = pipe.vae.decode(unpacked * std + mean, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def set_tiling(pipe, tile: int, stride: int) -> None:
    pipe.vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile)
    pipe.vae.tile_sample_stride_height = stride
    pipe.vae.tile_sample_stride_width = stride


def baseline(pipe) -> None:
    """Отправная точка: то, что стоит в проекте сегодня."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True  # значение по умолчанию у torch
    try:
        pipe.transformer.set_attention_backend("native")
    except Exception:  # noqa: BLE001 — механизм может быть недоступен
        pass
    set_tiling(pipe, 512, 256)


def levers(pipe) -> list[tuple[str, object]]:
    """Список рычагов: имя и функция настройки поверх отправной точки."""

    def attention(name: str):
        def apply(p):
            p.transformer.set_attention_backend(name)
        return apply

    def flag(**values):
        def apply(_p):
            for key, value in values.items():
                owner, attribute = key.rsplit(".", 1)
                target = torch.backends
                for part in owner.split("."):
                    target = getattr(target, part)
                setattr(target, attribute, value)
        return apply

    def tiling(tile: int, stride: int):
        def apply(p):
            set_tiling(p, tile, stride)
        return apply

    return [
        ("baseline", lambda _p: None),
        ("cudnn.benchmark", flag(**{"cudnn.benchmark": True})),
        ("tf32 in matmul", flag(**{"cuda.matmul.allow_tf32": True})),
        ("attention _native_flash", attention("_native_flash")),
        ("attention _native_efficient", attention("_native_efficient")),
        ("attention _native_cudnn", attention("_native_cudnn")),
        ("VAE tile 1024/512", tiling(1024, 512)),
        ("VAE tile 768/384", tiling(768, 384)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Speed levers one at a time")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=2, help="the first run is a warm-up")
    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument("--with-source", action="store_true",
                        help="add a condition image: an edit has a sequence twice as long")
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    pipe, _residency, _cache = loader.load(config.MODEL_DIR)
    device = pipe._execution_device

    condition = None
    if args.with_source:
        condition = [Image.open(config.LOG_DIR / "smoke" / "source.png").convert("RGB")]

    def denoise():
        return pipe(
            prompt=PROMPT,
            image=condition,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.steps,
            output_resolution=1024 if condition else args.resolution,
            output_type="latent",
            generator=torch.Generator(device=device).manual_seed(7),
        ).images

    rows: dict[str, dict] = {}
    for name, apply in levers(pipe):
        baseline(pipe)
        apply(pipe)

        latents = None
        denoise_times, decode_times = [], []
        for attempt in range(args.repeat + 1):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            latents, spent = timed(denoise)
            _image, decode_spent = timed(decode, pipe, latents, args.resolution, args.resolution)
            if attempt:  # прогревочный прогон отбрасываем
                denoise_times.append(spent)
                decode_times.append(decode_spent)

        rows[name] = {
            "seconds_per_step": round(min(denoise_times) / args.steps, 3),
            "decode_s": round(min(decode_times), 2),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        print(f"{name}: {rows[name]}", flush=True)
        (OUT / "scores.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    base = rows["baseline"]
    print("\n=== summary (minimum over steady-state runs) ===")
    print(f"{'lever':>28} | {'s/step':>6} | {'gain':>8} | {'decode, s':>9} | "
          f"{'gain':>8} | {'peak GiB':>8}")
    for name, row in rows.items():
        step_gain = 100 * (base["seconds_per_step"] - row["seconds_per_step"]) / base["seconds_per_step"]
        decode_gain = 100 * (base["decode_s"] - row["decode_s"]) / base["decode_s"]
        print(f"{name:>28} | {row['seconds_per_step']:>6} | {step_gain:>7.1f}% | "
              f"{row['decode_s']:>9} | {decode_gain:>7.1f}% | {row['peak_gib']:>8}")
    print(f"\njson: {OUT / 'scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
