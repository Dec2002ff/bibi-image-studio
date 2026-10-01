r"""Опыт: плитка VAE для профиля «low» — пик памяти и время декодирования.

На 8 ГБ пик денойзинга даёт не трансформер, а декодирование VAE: плитки 512
добавляют 1.7 ГиБ к 4.55 резидентным. Здесь — то же декодирование на одном
латенте кадра 1024² и 1536² при плитках 512/384/256 (шаг — половина плитки,
перекрытие не меньше отбрасываемого края ``vae_tiling.TRIM``). Меры: прирост
пика над весами VAE, время, и расхождение с плиткой 512 (средняя абсолютная
разница в уровнях из 255) — швы здешний декодер убирает при любой плитке, но
это стоит проверить, а не предположить.

Запуск:
    .venv\Scripts\python tools\experiments\lowvram_vae_tile.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402

use_utf8_console()

import torch  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import vae_tiling  # noqa: E402


def main() -> None:
    from diffusers import AutoencoderKLQwenImage21

    vae = AutoencoderKLQwenImage21.from_pretrained(str(config.MODEL_DIR / "vae"), torch_dtype=torch.bfloat16).to("cuda")
    vae_tiling.install(vae)
    base = torch.cuda.memory_allocated()
    torch.manual_seed(0)
    print(f"{'frame':6s} {'tile':>5s} {'peak +GiB':>9s} {'s':>6s} {'|diff| vs 512':>14s}")
    for side in (1024, 1536):
        latent = torch.randn(1, vae.config.z_dim, 1, side // 16, side // 16, device="cuda", dtype=torch.bfloat16)
        reference = None
        for tile in (512, 384, 256):
            vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile)
            vae.tile_sample_stride_height = vae.tile_sample_stride_width = tile // 2
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                image = vae.decode(latent, return_dict=False)[0].float().clamp(-1, 1)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            peak = (torch.cuda.max_memory_allocated() - base) / 2**30
            image = ((image + 1) * 127.5).cpu()
            diff = "-" if reference is None else f"{(image - reference).abs().mean().item():.3f}"
            reference = image if reference is None else reference
            print(f"{side:6d} {tile:5d} {peak:9.2f} {seconds:6.2f} {diff:>14s}")


if __name__ == "__main__":
    main()
