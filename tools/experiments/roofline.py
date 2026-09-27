r"""Опыт: далеко ли шаг денойзинга от предела карты.

Разложение по ядрам (`kernel_profile.py`) показало, что шаг состоит из двух
вещей: линейные слои (два GEMM-ядра, около 59 % времени) и внимание (около
32 %). Прежде чем что-то оптимизировать, надо знать, есть ли там запас
вообще: если ядра уже идут на скорости, близкой к пиковой для bf16, то
никакая перестановка вызовов не поможет — помогут только меньше операций
(меньше токенов, меньше шагов) или другая точность.

Здесь считается достигнутая производительность в TFLOPS и сравнивается с
измеренным пиком самой карты на тех же операциях. Пик берётся не из
справочника, а меряется здесь же: справочные числа для RTX 3090 приводят
то со разреженностью, то без, и спорить с ними бессмысленно.

Формулы. Для внимания на слой: 2·S_q·S_k·H·D для QKᵀ плюс столько же для
умножения на V. Для линейных слоёв: 2·N·T, где N — число параметров
трансформера, T — число токенов.

Запуск (из корня проекта):
    .venv\Scripts\python tools\experiments\roofline.py
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

from fooocus_qwen import config, logging_setup

OUT = config.LOG_DIR / "roofline"


def measure_peak(size: int = 8192, repeats: int = 30) -> float:
    """Пиковые TFLOPS карты на большом матричном умножении в bf16."""
    a = torch.randn(size, size, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(size, size, dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        a @ b
    torch.cuda.synchronize()
    spent = time.perf_counter() - started
    flops = 2 * size ** 3 * repeats
    return flops / spent / 1e12


def measure_attention_peak(seq: int, heads: int, dim: int, repeats: int = 20) -> float:
    """TFLOPS на том же внимании, что считает модель, но изолированно."""
    q = torch.randn(1, heads, seq, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, heads, seq, dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, heads, seq, dim, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    spent = time.perf_counter() - started
    flops = 4 * seq * seq * heads * dim * repeats
    return flops / spent / 1e12


def main() -> int:
    parser = argparse.ArgumentParser(description="Achieved share of the GPU peak")
    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    peak_gemm = measure_peak()
    print(f"GPU peak on bf16 GEMM: {peak_gemm:.1f} TFLOPS", flush=True)

    from fooocus_qwen.engine import loader

    pipe, _residency, _cache = loader.load(config.MODEL_DIR)
    device = pipe._execution_device
    transformer = pipe.transformer

    parameters = sum(p.numel() for p in transformer.parameters())
    layers = len(transformer.transformer_blocks)
    print(f"transformer: {parameters / 1e9:.2f}B parameters, {layers} layers", flush=True)

    # Форма внимания снимается с настоящего прогона, а не предполагается.
    import diffusers.models.transformers.transformer_qwenimage21 as module

    shapes: list[tuple[int, int, int, int]] = []
    original = module.dispatch_attention_fn

    def spy(query, key, value, **kwargs):
        if query.shape[1] > 64:  # интересен только длинный, целевой вызов
            shapes.append((query.shape[1], key.shape[1], query.shape[2], query.shape[3]))
        return original(query, key, value, **kwargs)

    module.dispatch_attention_fn = spy

    def run(steps: int):
        return pipe(
            prompt="a wooden desk with a brass lamp",
            height=args.resolution, width=args.resolution,
            num_inference_steps=steps, output_resolution=args.resolution,
            output_type="latent",
            generator=torch.Generator(device=device).manual_seed(7),
        )

    run(2)  # прогрев
    shapes.clear()
    torch.cuda.synchronize()
    started = time.perf_counter()
    run(args.steps)
    torch.cuda.synchronize()
    spent = time.perf_counter() - started
    module.dispatch_attention_fn = original

    per_step = spent / args.steps
    seq_q, seq_k, heads, dim = shapes[-1]
    tokens = seq_q

    attention_flops = 4 * seq_q * seq_k * heads * dim * layers
    linear_flops = 2 * parameters * tokens
    total_flops = attention_flops + linear_flops

    peak_attention = measure_attention_peak(seq_q, heads, dim)

    report = {
        "seconds_per_step": round(per_step, 3),
        "tokens": tokens,
        "attention_shape": {"q": seq_q, "k": seq_k, "heads": heads, "head_dim": dim},
        "attention_tflop": round(attention_flops / 1e12, 1),
        "linear_tflop": round(linear_flops / 1e12, 1),
        "achieved_tflops": round(total_flops / per_step / 1e12, 1),
        "peak_gemm_tflops": round(peak_gemm, 1),
        "peak_attention_tflops": round(peak_attention, 1),
    }
    report["peak_share_%"] = round(100 * report["achieved_tflops"] / peak_gemm, 1)
    (OUT / f"{args.resolution}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== frame {args.resolution}x{args.resolution} ===")
    for key, value in report.items():
        print(f"{key:>26}: {value}")
    print()
    if report["peak_share_%"] > 70:
        print("The step is compute-bound and runs close to the GPU peak: reordering")
        print("calls has nothing left to speed up. What remains is fewer tokens,")
        print("fewer steps or a different precision.")
    else:
        print("Far from the peak; it is worth looking for overheads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
