r"""Опыт: что декодер VAE делает на краях плитки и почему это видно как полосы.

Полосы в кадрах заказчика стоят на ``x = 256k - 10`` — шаг ровно равен шагу
плиток (``VAE_TILE_STRIDE``), но сами линии сдвинуты внутрь предыдущего блока
на десяток пикселей. Прямая граница плиток при этом чистая. Объяснение может
быть только одно: в результат просачивается **край** соседней плитки, где
декодер работает с дополнением и врёт.

Линейное смешивание `diffusers` даёт краю плитки вес ``1 - x/stride``: в самом
последнем столбце это 0.4 %, а за десять столбцов до него — уже 4 %. Если
краевой артефакт велик, четырёх процентов хватает на видимую цветную линию.

Здесь это проверяется напрямую: одна и та же область латента декодируется
дважды — в составе целого кадра и как отдельная плитка, — и разность
показывает профиль вранья по столбцам.

Запуск (из корня проекта):
    .venv\Scripts\python tools\experiments\vae_tile_edge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fooocus_qwen.logging_setup import use_utf8_console

use_utf8_console()

import argparse

import numpy as np
import torch
from PIL import Image

from fooocus_qwen import config

SOURCE = config.OUTPUT_DIR / "2026-09-22" / "19-35-06.png"


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="VAE tile edge artifact")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--tile", type=int, default=512, help="tile size in frame pixels")
    args = parser.parse_args()

    from diffusers import AutoencoderKLQwenImage21

    print("loading VAE…", flush=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(
        config.MODEL_DIR, subfolder="vae", torch_dtype=torch.bfloat16
    ).to("cuda")
    vae.eval()

    image = Image.open(args.source).convert("RGB")
    # Кадр обрезается до целого числа плиток, чтобы неполная последняя плитка
    # не подмешивала в опыт свой отдельный эффект.
    width = height = args.tile * 2
    image = image.crop((0, 0, width, height))
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    # VAE 2.1 принимает четыре канала: маска приходит моделью как альфа
    # (docs/research/2026-09-22-maska-kak-alfa.md). Здесь она везде
    # непрозрачна — опыт про края плиток, а не про маску.
    opaque = np.ones_like(array[..., :1])
    array = np.concatenate([array, opaque], axis=2)
    x = torch.from_numpy(array).permute(2, 0, 1)[None, :, None].to("cuda", torch.bfloat16)

    print(f"encoding {width}x{height}…", flush=True)
    vae.disable_tiling()
    latent = vae.encode(x).latent_dist.mode()

    print("decoding whole…", flush=True)
    whole = vae.decode(latent, return_dict=False)[0][:, :, 0]
    whole = _to_numpy(whole)

    ratio = vae.spatial_compression_ratio
    tile_latent = args.tile // ratio
    print(f"decoding a single tile {args.tile}x{args.tile} (latent {tile_latent})…", flush=True)
    piece = vae.decode(latent[:, :, :, :tile_latent, :tile_latent], return_dict=False)[0][:, :, 0]
    piece = _to_numpy(piece)

    # Разность плитки и того же места целого кадра: чистый краевой артефакт.
    diff = np.abs(piece - whole[: args.tile, : args.tile]).mean(axis=2)
    by_column = diff.mean(axis=0)
    middle = float(np.median(by_column[args.tile // 4 : args.tile * 3 // 4]))

    print(f"\ntile middle (error background): {middle:.3f} levels of 255")
    print("\nlast columns of the tile, the edge in question:")
    for offset in (0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64):
        column = args.tile - 1 - offset
        value = by_column[column]
        # Вес, с которым этот столбец входит в результат при шаге 256.
        weight = max(0.0, 1.0 - (256 - 1 - offset) / 256) if offset < 256 else 0.0
        print(
            f"  edge−{offset:<3d} (x={column:3d}): off by {value:7.3f}"
            f"  (×{value / max(middle, 1e-6):5.1f} of background), blend weight {weight * 100:4.1f} %"
            f"  → contribution {value * weight:6.3f}"
        )

    print("\nfirst columns of the tile (the other edge, for symmetry):")
    for column in (0, 1, 2, 4, 8, 16, 32):
        print(f"  x={column:3d}: off by {by_column[column]:7.3f} (×{by_column[column] / max(middle, 1e-6):5.1f})")

    # Вторая ось: артефакт обязан быть и на нижнем крае — обрезать придётся оба.
    by_row = diff.mean(axis=1)
    middle_row = float(np.median(by_row[args.tile // 4 : args.tile * 3 // 4]))
    print()
    print(f"bottom edge of the tile (row background {middle_row:.3f}):")
    for offset in (0, 2, 4, 8, 16, 24, 32, 48):
        row = args.tile - 1 - offset
        print(f"  edge−{offset:<3d} (y={row:3d}): off by {by_row[row]:7.3f} (×{by_row[row] / max(middle_row, 1e-6):5.1f})")
    print("top edge:", ", ".join(f"y={r}: {by_row[r]:.3f}" for r in (0, 1, 2, 4, 8)))

    trusted = _trusted_margin(by_column, middle)
    print(
        f"\nConclusion: about {trusted} columns at the tile edge cannot be trusted; "
        f"beyond them the error drops below three times the background."
    )
    return 0


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    out = tensor.float().clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (out[..., :3] + 1.0) * 127.5


def _trusted_margin(by_column: np.ndarray, middle: float) -> int:
    """Сколько столбцов от края врут заметно сильнее середины плитки."""
    limit = 3 * middle
    for offset in range(len(by_column) // 2):
        if by_column[len(by_column) - 1 - offset] < limit:
            return offset
    return len(by_column) // 2


if __name__ == "__main__":
    raise SystemExit(main())
