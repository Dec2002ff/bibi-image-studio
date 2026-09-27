r"""Опыт: сколько стоит несращённый адаптер Turbo — на подключении и на каждом шаге.

Повод — совет из сообщества (Reddit, 2026-09-27): «сплавьте LoRA с весами
заранее, офлайн». Там речь о Mac, где ``fuse_lora()`` на старте занимал
35–45 с. У нас адаптер подключается один раз, при первом выборе Turbo, и
держится отдельно от весов (включается и выключается на лету). Вопросы:

1. сколько стоит первое подключение (``TurboAdapter._load``);
2. сколько стоит сам несращённый адаптер на каждом шаге: та же генерация
   Turbo — 6 шагов, те же узлы и разрешение — с включённой и выключенной
   LoRA. Разница — это ровно то, что сэкономило бы слияние.

Качество с выключенной LoRA не важно (6 шагов без дистиллята — каша):
меряется только время.

Запуск (нужны веса модели и Turbo):
    .venv\Scripts\python tools\experiments\lora_overhead.py --runs 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fooocus_qwen import config, settings  # noqa: E402
from fooocus_qwen.engine import presets  # noqa: E402
from fooocus_qwen.engine.generator import GenerationRequest  # noqa: E402
from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(description="Cost of the unfused Turbo LoRA: attach time and per-step overhead")
    parser.add_argument("--runs", type=int, default=3, help="timed runs per variant (after one warm-up)")
    parser.add_argument("--size", default="1:1", help="aspect ratio (the Turbo preset is 1024 px)")
    args = parser.parse_args()

    from fooocus_qwen.ui.state import Studio

    studio = Studio(config.AppConfig(lang="en"))
    chosen = settings.load()
    print(f"precision: {chosen.precision}, SageAttention: {chosen.sage_attention}")
    started = time.perf_counter()
    engine = studio.generator
    print(f"model load: {time.perf_counter() - started:.1f} s")

    preset = presets.get("Turbo")
    turbo = engine._turbo
    started = time.perf_counter()
    turbo.activate(True)
    print(f"first Turbo attach (load_lora_weights + restage): {time.perf_counter() - started:.1f} s")

    def request(seed: int) -> GenerationRequest:
        return GenerationRequest(prompt="a red fox in a snowy birch forest", prompt_original="x",
                                 preset=preset, aspect=args.size, seed=seed)

    def timed(lora: bool) -> list[float]:
        # Генератор сам включает адаптер для пресета Turbo; для замера «без
        # LoRA» он выключается после этого, прямо перед вызовом пайплайна.
        original = turbo.activate

        def activate(enabled: bool) -> None:
            original(enabled)
            if enabled and not lora:
                engine._pipe.disable_lora()

        turbo.activate = activate
        try:
            engine.generate(request(0))  # прогрев: кэш промта, ядра
            times = []
            for seed in range(1, args.runs + 1):
                started = time.perf_counter()
                engine.generate(request(seed))
                times.append(time.perf_counter() - started)
            return times
        finally:
            turbo.activate = original
            engine._pipe.enable_lora()

    with_lora = timed(True)
    without_lora = timed(False)
    a, b = statistics.median(with_lora), statistics.median(without_lora)
    print(f"Turbo with LoRA:    median {a:.2f} s  {[round(x, 2) for x in with_lora]}")
    print(f"Turbo without LoRA: median {b:.2f} s  {[round(x, 2) for x in without_lora]}")
    print(f"LoRA overhead: {a - b:+.2f} s per image ({(a - b) / b * 100:+.1f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
