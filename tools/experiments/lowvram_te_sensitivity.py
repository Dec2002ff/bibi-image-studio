r"""Опыт: откуда ошибка эмбеддингов у энкодера в int8 и чем её убрать.

Первый опыт (``lowvram_text_encoder.py``) дал int8 по строкам относительную
ошибку 7–19 % к bf16 и худший токен с косинусом 0.38 на промте правки. Для
int8 «только веса» это много, и прежде чем решать, надо знать две вещи:

1. **шумовой пол** — насколько от точного ответа отстоит сам bf16 (эталон —
   fp32: веса и вычисления в float32);
2. **источник** — какая часть сжатия даёт ошибку.

Сжатие здесь имитируется: веса блока квантуются и тут же разжимаются в
момент подъёма блока на карту. Арифметика та же, что у ``quant.Int8Linear``
(вес разжат в float32 и округлён в bf16), а варианты меняются без пересборки
файлов на диске.

Варианты:

* ``bf16``       — bf16 как есть (повтор эталона bf16 — проверка детерминизма);
* ``row``        — int8 по строкам, все семь линейных слоёв (как в первом опыте);
* ``g128``/``g64`` — int8 по группам из 128/64 входов строки;
* ``row-no-down`` — int8 по строкам, кроме ``down_proj``;
* ``row-skip4``  — int8 по строкам, кроме слоёв 0–3 (где рождаются «массивные
  активации» — Sun et al., 2024).

Ошибки считаются к fp32 и к bf16. Таблица вложений здесь не сжимается: её
вклад меряется отдельно (``emb``).

Запуск:
    .venv\Scripts\python tools\experiments\lowvram_te_sensitivity.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402

use_utf8_console()

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import streaming  # noqa: E402
from fooocus_qwen.engine.residency import _named_tensors, _place  # noqa: E402

sys.path.insert(0, str(ROOT / "tools" / "experiments"))
from lowvram_text_encoder import PROMPTS, bf16_encoder  # noqa: E402

LINEAR = re.compile(r"self_attn\.[qkvo]_proj\.weight$|mlp\.(gate|up|down)_proj\.weight$")
LAYER = re.compile(r"^p:(\d+)\.")


def fake_rowwise(weight: torch.Tensor) -> torch.Tensor:
    values = weight.float()
    scale = values.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    return (torch.round(values / scale).clamp_(-127, 127) * scale).to(torch.bfloat16)


def fake_grouped(weight: torch.Tensor, group: int) -> torch.Tensor:
    out, width = weight.shape
    values = weight.float().reshape(out, width // group, group)
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
    q = torch.round(values / scale).clamp_(-127, 127)
    return (q * scale).reshape(out, width).to(torch.bfloat16)


class Variant(streaming.StreamedModule):
    """Подача по блоку с преобразованием весов блока при подъёме."""

    def __init__(self, module, blocks, device, transform, dtype=torch.bfloat16):
        super().__init__(module, blocks, device)
        self._transform = transform
        self._dtype = dtype
        self._layer_of = {id(block): index for index, block in enumerate(module.model.language_model.layers)}

    def to_device(self):
        super().to_device()
        if self._dtype != torch.bfloat16:
            for name, tensor in _named_tensors(self.module):
                if name in self._shared and tensor.is_floating_point():
                    _place(self.module, name, tensor, tensor.to(self._dtype))

    def _lift(self, block, _args):
        layer = self._layer_of.get(id(block))
        for name, tensor in _named_tensors(block):
            value = tensor.to(self._device)
            if layer is not None and LINEAR.search(name):
                value = self._transform(layer, name, value)
            if value.is_floating_point():
                value = value.to(self._dtype)
            _place(block, name, tensor, value)


VARIANTS = {
    "bf16": lambda layer, name, w: w,
    "row": lambda layer, name, w: fake_rowwise(w),
    "g128": lambda layer, name, w: fake_grouped(w, 128),
    "g64": lambda layer, name, w: fake_grouped(w, 64),
    "row-no-down": lambda layer, name, w: w if "down_proj" in name else fake_rowwise(w),
    "row-skip4": lambda layer, name, w: w if layer < 4 else fake_rowwise(w),
}


def run(model, processor, transform, dtype=torch.bfloat16):
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=model, processor=processor, transformer=None)
    staged = Variant(model, streaming.text_encoder_blocks(model), "cuda", transform, dtype)
    results = {}
    for name, prompt, image_path in PROMPTS:
        image = [Image.open(ROOT / image_path).convert("RGB").resize((1024, 1024))] if image_path else None
        staged.to_device()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            embeds, mask, _ = pipe._get_qwen_prompt_embeds(prompt, image, torch.device("cuda"))
        staged.to_host()
        results[name] = embeds[0, mask[0].bool()].float().cpu()
    return results


def compare(candidate, reference):
    cells = []
    for name, _, _ in PROMPTS:
        ref, cand = reference[name], candidate[name]
        cos = torch.nn.functional.cosine_similarity(ref, cand, dim=-1)
        rel = ((cand - ref).norm() / ref.norm()).item()
        cells.append(f"{cos.mean():.5f}/{cos.min():.3f}/{rel:.3f}")
    return "  ".join(f"{cell:>22s}" for cell in cells)


def main() -> None:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(config.MODEL_DIR / "processor"))
    model = bf16_encoder(config.MODEL_DIR / "text_encoder")
    print("fp32 reference...")
    fp32 = run(model, processor, VARIANTS["bf16"], torch.float32)
    bf16 = run(model, processor, VARIANTS["bf16"])
    header = "  ".join(f"{name + ' cos/min/rel':>22s}" for name, _, _ in PROMPTS)
    print(f"{'variant':12s} {'vs':5s} {header}")
    print(f"{'bf16':12s} fp32  {compare(bf16, fp32)}")
    for variant, transform in VARIANTS.items():
        result = run(model, processor, transform)
        print(f"{variant:12s} fp32  {compare(result, fp32)}")
        print(f"{'':12s} bf16  {compare(result, bf16)}")


if __name__ == "__main__":
    main()
