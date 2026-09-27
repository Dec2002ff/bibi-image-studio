r"""Опыт: стоит ли закрепление памяти тех секунд, что оно отнимает у запуска.

В журнале заказчика видно: «Готовлю копии весов на хосте» — двадцать четыре
секунды из сорока девяти, то есть половина запуска. Это ``pin_memory()`` на
29.6 ГБ (трансформер 13.3 и энкодер 16.3): веса после ``from_pretrained``
уже лежат в оперативной памяти, и закрепление делает их **полную копию** в
странице, которую нельзя выгрузить.

Взамен закрепление ускоряет пересылку на карту: DMA идёт напрямую, без
промежуточного буфера. Пересылка случается при каждом промахе кэша
эмбеддингов — дважды на промах (туда и обратно, для обеих моделей).

Опыт считает обе стороны сделки: сколько секунд закрепление отнимает у
запуска и сколько возвращает на каждой перестановке. Точка безубыточности —
частное этих чисел, то есть число промахов кэша, после которого закрепление
окупается.

Запуск (из корня проекта):
    .venv\Scripts\python tools\experiments\pinning_cost.py --pin
    .venv\Scripts\python tools\experiments\pinning_cost.py --no-pin

Разными процессами: держать в одном два набора копий весов по 29.6 ГБ
негде.
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

OUT = config.LOG_DIR / "pinning"
# Файл замеров с английскими ключами. Прежний scores.json (русские ключи)
# не дочитывается: его строки уронили бы опыт на KeyError.
SCORES = "scores.en.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Cost and benefit of pinned memory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pin", dest="pin", action="store_true")
    group.add_argument("--no-pin", dest="pin", action="store_false")
    parser.add_argument("--swaps", type=int, default=3, help="how many swaps to measure")
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    from diffusers import QwenImage21Pipeline  # noqa: F401 — прогрев импортов

    from fooocus_qwen.engine.pipeline import QwenImage21StudioPipeline, assert_contract
    from fooocus_qwen.engine.residency import ResidencyManager

    assert_contract()

    started = time.perf_counter()
    pipe = QwenImage21StudioPipeline.from_pretrained(str(config.MODEL_DIR), dtype=torch.bfloat16)
    read_seconds = time.perf_counter() - started

    residency = ResidencyManager(pipe, device="cuda", pin_memory=args.pin)
    started = time.perf_counter()
    residency.start()
    stage_seconds = time.perf_counter() - started

    # Перестановка: ровно то, что делает промах кэша эмбеддингов.
    swap_times = []
    for _ in range(args.swaps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with residency.text_encoder_resident():
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        swap_times.append(time.perf_counter() - started)

    row = {
        "pinned": args.pin,
        "weights_read_s": round(read_seconds, 1),
        "copies_staging_s": round(stage_seconds, 1),
        "swap_s": round(min(swap_times), 2),
        "swaps": [round(t, 2) for t in swap_times],
    }

    path = OUT / SCORES
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rows["pinned" if args.pin else "unpinned"] = row
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{row}")

    if len(rows) == 2:
        pinned, plain = rows["pinned"], rows["unpinned"]
        saved = plain["copies_staging_s"] - pinned["copies_staging_s"]  # < 0: закрепление дороже
        gain = plain["swap_s"] - pinned["swap_s"]
        print("\n=== trade-off ===")
        print(f"startup: {pinned['copies_staging_s']} s pinned vs "
              f"{plain['copies_staging_s']} s unpinned")
        print(f"swap: {pinned['swap_s']} s vs {plain['swap_s']} s")
        if gain > 0:
            print(f"pays off after {abs(saved) / gain:.0f} cache misses")
        else:
            print("pinning does not speed up the swap; nothing to pay off")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
