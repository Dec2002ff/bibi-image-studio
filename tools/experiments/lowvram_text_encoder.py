r"""Опыт: текстовый энкодер в INT8, поданный на карту по блоку, против bf16.

Вопрос — сколько теряют эмбеддинги промта, если энкодер (Qwen3-VL 8B, 16.3
ГБ) сжать до int8 по строкам (``engine/text_encoder.py``) и подавать на
видеокарту по одному блоку (``engine/streaming.py``). Эталон — тот же
энкодер в bf16, поданный тем же способом: целиком в 8 ГБ он не помещается,
а поблочная подача не меняет арифметику.

Меры, на каждом промте:

* косинус эмбеддингов по токенам — средний и худший;
* относительная ошибка ``‖e_int8 − e_bf16‖ / ‖e_bf16‖`` по всему промту;
* время кодирования и пик видеопамяти.

Промты — короткий, длинный с надписью (текст в картинке — первое, что
портится, когда эмбеддинги неточны) и промт правки с картинкой (визуальная
часть энкодера и сотни токенов изображения).

Запуск:
    .venv\Scripts\python tools\experiments\lowvram_text_encoder.py
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
from PIL import Image  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import quant, streaming, text_encoder  # noqa: E402

PROMPTS = [
    ("short", "a red fox in the snow", None),
    (
        "text",
        'A vintage travel poster for Lisbon. Bold yellow headline "LISBOA 1934", a tram climbing a steep street, '
        "pastel houses, art deco typography, small caption at the bottom reading \"Ride the 28\".",
        None,
    ),
    ("edit", "make the sky a warm sunset, keep everything else", "resources/poses/add_pose.png"),
]


def bf16_encoder(config_dir: Path):
    """bf16-энкодер, веса — отображение файлов (оперативную память не занимают)."""
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    model_config = AutoConfig.from_pretrained(str(config_dir))
    with init_empty_weights(include_buffers=False):
        model = Qwen3VLForConditionalGeneration._from_config(model_config, dtype=torch.bfloat16)
    model.lm_head = quant.Passthrough()
    state = {}
    for shard in sorted(Path(config_dir).glob("model-*.safetensors")):
        state.update(load_file(str(shard)))
    state.pop("lm_head.weight", None)
    problems = quant.assign(model, state)
    assert not problems, problems[:3]
    return model.eval()


def blocks_of(model):
    return [*model.model.language_model.layers, *model.model.visual.blocks]


def encode_all(model, processor):
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=model, processor=processor, transformer=None)
    staged = streaming.StreamedModule(model, blocks_of(model), "cuda")
    results = {}
    for name, prompt, image_path in PROMPTS:
        image = [Image.open(ROOT / image_path).convert("RGB").resize((1024, 1024))] if image_path else None
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        started = time.perf_counter()
        staged.to_device()
        with torch.no_grad():
            embeds, mask, _ = pipe._get_qwen_prompt_embeds(prompt, image, torch.device("cuda"))
        staged.to_host()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        peak = (torch.cuda.max_memory_allocated() - base) / 2**30
        results[name] = (embeds[0, mask[0].bool()].float().cpu(), seconds, peak)
        print(f"  {name:6s} tokens {results[name][0].shape[0]:5d}  {seconds:6.2f} s  peak {peak:.2f} GiB")
    return results


def main() -> None:
    from transformers import AutoProcessor

    source = config.MODEL_DIR / "text_encoder"
    target = config.TE_INT8_DIR
    started = time.perf_counter()
    if text_encoder.ensure(source, target, out=print):
        print(f"conversion: {time.perf_counter() - started:.0f} s")
    processor = AutoProcessor.from_pretrained(str(config.MODEL_DIR / "processor"))

    print("bf16, streamed by block:")
    reference = encode_all(bf16_encoder(source), processor)
    print("int8, streamed by block:")
    candidate = encode_all(text_encoder.load(source, target), processor)

    print("\nprompt  cos mean   cos min   rel err")
    for name, _, _ in PROMPTS:
        ref, cand = reference[name][0], candidate[name][0]
        cos = torch.nn.functional.cosine_similarity(ref, cand, dim=-1)
        rel = ((cand - ref).norm() / ref.norm()).item()
        print(f"{name:6s}  {cos.mean():.6f}  {cos.min():.6f}  {rel:.4f}")


if __name__ == "__main__":
    main()
