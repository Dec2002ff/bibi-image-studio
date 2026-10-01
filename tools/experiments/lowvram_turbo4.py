r"""Опыт: 4-шаговый дистиллят Abiray (GGUF Q4_K_M) против Turbo и LowQuality на 8 ГБ.

``Abiray/Qwen-Image-2.1-viggle-4-steps-turbo-GGUF`` — ранний (v0.1) 4-шаговый
дистиллят Viggle, влитый в трансформер и сжатый в GGUF. Правила карточки:
4 или 8 шагов, CFG 1, без негатива, Euler, расписание «simple». Здесь
«simple» — равномерные узлы ``linspace(1, 1/n, n)``, к которым пайплайн сам
применяет сдвиг по разрешению, и планировщик turbo Viggle (у штатного
``shift_terminal: 0.02`` портит последний шаг).

Варианты на одних промтах и сидах:

* ``low16``  — LowQuality, полная модель, 16 шагов (эталон качества);
* ``turbo6`` — пресет Turbo: дистиллят v0.2.1 адаптером r128, 6 шагов;
* ``t4x4``   — Abiray Q4_K_M, 4 шага;
* ``t4x8``   — Abiray Q4_K_M, 8 шагов.

Меры: секунды на кадр (второй прогон того же промта — без кодирования),
пик видеопамяти. Кадры — в ``logs/turbo4/`` для осмотра глазами.

Запуск:
    .venv\Scripts\python tools\experiments\lowvram_turbo4.py
"""

from __future__ import annotations

import gc
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
from fooocus_qwen.engine import loader, presets, residency  # noqa: E402
from fooocus_qwen.engine.generator import GenerationRequest  # noqa: E402
from fooocus_qwen.ui.state import Studio  # noqa: E402

OUT = ROOT / "logs" / "turbo4"
TURBO4 = config.GGUF_DIR / "qwen_image_2.1_turbo_Q4_K_M.gguf"
SEED = 7
PROMPTS = {
    "poster": 'A vintage travel poster for Lisbon. Bold yellow headline "LISBOA 1934", a tram climbing a steep '
    "street, pastel houses, art deco typography",
    "portrait": "studio portrait of an old fisherman mending a net, warm rim light, 85mm, detailed skin",
    "food": "a bowl of ramen with a soft egg and chashu, top view, studio light",
}
EDIT_PROMPT = "make it a snowy winter evening, keep the tram and the headline"


def source() -> Image.Image | None:
    path = ROOT / "logs" / "lowvram" / "t2i_low_q4km.png"
    return Image.open(path).convert("RGB") if path.is_file() else None


def timed(run) -> tuple[Image.Image, float, float]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    image = run()
    torch.cuda.synchronize()
    return image, time.perf_counter() - started, torch.cuda.max_memory_allocated() / 2**30


def studio_variants(rows: list[str]) -> None:
    studio = Studio(config.AppConfig(preload=False))
    for variant, preset in (("low16", "LowQuality"), ("turbo6", "Turbo")):
        studio.weights_for(presets.get(preset), "en")
        jobs = [(name, GenerationRequest(prompt=prompt, preset=presets.get(preset), seed=SEED))
                for name, prompt in PROMPTS.items()]
        if source() is not None:
            jobs.append(("edit", GenerationRequest(prompt=EDIT_PROMPT, preset=presets.get(preset), seed=SEED,
                                                   source=source())))
        for name, request in jobs:
            for attempt in ("new", "cached"):
                image, seconds, peak = timed(lambda r=request: studio.run_generation(r, "en")[0][0].image)
                rows.append(f"{variant:7s} {name:9s} {attempt:6s} {seconds:7.1f} {peak:6.2f}")
                print(rows[-1], flush=True)
            image.save(OUT / f"{variant}_{name}.png")
    studio.unload()


def turbo4_variants(rows: list[str]) -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler

    pipe, _residency, _cache = loader.load(
        config.MODEL_DIR, gguf_file=TURBO4, text_encoder_dir=config.TE_INT8_DIR,
        policy=residency.STREAM, sage_attention=True, pin_memory=False,
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(config.TURBO_DIR), subfolder="scheduler")
    pipe.set_progress_bar_config(disable=True)
    jobs = [(name, prompt, None) for name, prompt in PROMPTS.items()]
    if source() is not None:
        jobs.append(("edit", EDIT_PROMPT, source()))
    for steps in (4, 8):
        sigmas = [1.0 - index / steps for index in range(steps)]
        for name, prompt, image_in in jobs:
            arguments = dict(prompt=prompt, num_inference_steps=steps, sigmas=sigmas, true_cfg_scale=1.0)
            if image_in is None:
                arguments.update(height=1024, width=1024)
            else:
                arguments.update(image=[image_in], output_resolution=1024)
            for attempt in ("new", "cached"):
                def run(a=arguments, p=pipe):
                    generator = torch.Generator("cuda").manual_seed(SEED)
                    return p(generator=generator, **a).images[0]

                image, seconds, peak = timed(run)
                rows.append(f"t4x{steps:<4d} {name:9s} {attempt:6s} {seconds:7.1f} {peak:6.2f}")
                print(rows[-1], flush=True)
            image.save(OUT / f"t4x{steps}_{name}.png")
    del pipe
    gc.collect()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    print(f"{'variant':7s} {'prompt':9s} {'prompt':6s} {'s/img':>7s} {'peak':>6s}", flush=True)
    studio_variants(rows)
    torch.cuda.empty_cache()
    turbo4_variants(rows)
    (OUT / "results.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
