"""Трансформер из GGUF: 4–8 бит на вес для карт на 6–12 ГБ.

GGUF — формат llama.cpp; для Qwen-Image-2.1 его выпускают Unsloth
(``unsloth/Qwen-Image-2.1-GGUF``) в вариантах от Q3_K_M (2.95 ГБ) до Q8_0
(7.12 ГБ). Q4_K_M — 3.91 ГБ против 13.3 у bf16 и 6.8 у INT8 — единственный,
с которым трансформер и VAE вместе с активациями кадра 1024² помещаются в
восемь гигабайт (``docs/research/2026-10-02-8-gb.md``).

Веса остаются сжатыми и в видеопамяти: ``GGUFLinear`` из diffusers
разжимает вес слоя в bf16 прямо перед умножением и тут же его отпускает.
Это стоит нескольких миллисекунд на слой — на кадре 1024² меньше процента
шага, который упирается в умножения, а не в разжатие.

Файлы GGUF собраны конвертером ComfyUI, и имена в них — его:

* все ключи начинаются с ``model.diffusion_model.``;
* вход SwiGLU хранится одним тензором ``img_mlp.gate_up`` — у diffusers это
  два слоя, ``gate_layer`` и ``proj``. Строки квантованного тензора
  независимы (блоки GGUF идут вдоль входной оси), поэтому тензор режется по
  строкам без пересжатия: первая половина — ``gate_layer``, вторая — ``proj``
  (порядок сверен с INT8-весами Unsloth в именах diffusers на блоках 0, 5 и
  31: косинус 1.00 у совпадающих половин против 0.00 у переставленных).

Неквантованные тензоры (нормы, F32 и BF16) становятся обычными тензорами bf16:
модель считает в bf16, и держать рядом F32-копии значило бы платить вдвое за
то, что всё равно приводится к bf16 на входе слоя.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import torch

from .quant import assign

LOGGER = logging.getLogger(__name__)

PREFIX = "model.diffusion_model."
FUSED_GATE_UP = "img_mlp.gate_up.weight"
GATE = "img_mlp.gate_layer.weight"
PROJ = "img_mlp.proj.weight"


def diffusers_name(name: str) -> str:
    """Имя тензора ComfyUI без префикса — имя diffusers (кроме ``gate_up``)."""
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def split_gate_up(name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Режет слитый вход SwiGLU на ``gate_layer`` и ``proj`` по строкам.

    Работает и для сжатого тензора (строки байтов), и для обычного: в обоих
    случаях первая ось — выходные каналы.
    """
    if tensor.shape[0] % 2:
        raise ValueError(f"{name}: odd number of rows {tensor.shape[0]}, cannot split gate/up")
    half = tensor.shape[0] // 2
    stem = name[: -len(FUSED_GATE_UP)]
    return {stem + GATE: tensor[:half], stem + PROJ: tensor[half:]}


def read_state_dict(path: Path, device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
    """Читает GGUF в словарь тензоров с именами diffusers.

    Сжатые тензоры приходят ``GGUFParameter`` (байты плюс тип квантования),
    остальные — bf16. С ``device`` каждый тензор уезжает туда прямо из
    отображения файла: оперативная память процесса не растёт вовсе, что на
    машине с 16 ГБ различимо.
    """
    import gguf
    from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter

    plain = {
        gguf.GGMLQuantizationType.F32: None,
        gguf.GGMLQuantizationType.F16: None,
        gguf.GGMLQuantizationType.BF16: torch.bfloat16,
    }
    reader = gguf.GGUFReader(str(path))
    state: dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        kind = tensor.tensor_type
        if device is None:
            data = torch.from_numpy(tensor.data.copy())
        else:
            # Прямо из отображения файла на устройство, без копии в ОЗУ:
            # освобождённые копии куча Windows процессу не возвращает, и
            # четыре гигабайта оставались бы за ним до конца работы.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r".*not writable.*")
                data = torch.from_numpy(tensor.data)
        if kind in plain:
            if plain[kind] is not None:
                # BF16 numpy не знает: байты лежат как uint8, по два на число.
                data = data.view(torch.uint8).view(torch.bfloat16).reshape(tensor.data.shape[:-1] + (-1,))
            value = data.to(torch.bfloat16)
        elif kind in SUPPORTED_GGUF_QUANT_TYPES:
            value = GGUFParameter(data, quant_type=kind)
        else:
            raise ValueError(f"{tensor.name}: quantization type {kind!r} is not supported by diffusers")
        if device is not None:
            value = value.to(device)
        name = diffusers_name(tensor.name)
        if name.endswith(FUSED_GATE_UP):
            state.update(split_gate_up(name, value))
        else:
            state[name] = value
    return state


def load_transformer(model_dir: Path, gguf_file: Path, device: str | torch.device | None = None):
    """``QwenImage21Transformer2DModel`` с весами из GGUF.

    Модель строится по конфигурации основной модели без выделения памяти
    под веса (буферы — настоящие, как у INT8: частоты временного вложения в
    файле не хранятся). Линейные слои со сжатым весом заменяются на
    ``GGUFLinear``, после чего тензоры ставятся на места как есть
    (``quant.assign``: ``load_state_dict`` спотыкается о байтовую форму
    сжатых тензоров).
    """
    from accelerate import init_empty_weights
    from diffusers import QwenImage21Transformer2DModel
    from diffusers.quantizers.gguf.utils import _replace_with_gguf_linear

    config = QwenImage21Transformer2DModel.load_config(str(Path(model_dir) / "transformer"))
    with init_empty_weights(include_buffers=False):
        transformer = QwenImage21Transformer2DModel.from_config(config)

    state = read_state_dict(gguf_file, device)
    _replace_with_gguf_linear(transformer, torch.bfloat16, state)
    problems = assign(transformer, state)
    if problems:
        raise RuntimeError(
            f"GGUF weights do not match the transformer: {len(problems)} mismatches, first: {problems[0]}. "
            "The file is corrupted or belongs to another model; delete it and it will be downloaded again."
        )
    if device is not None:
        # Буферы (частоты поворотных вложений) созданы на процессоре.
        transformer.to(device)
    quantized = sum(1 for module in transformer.modules() if type(module).__name__ == "GGUFLinear")
    LOGGER.info("GGUF transformer %s: %d quantized layers", Path(gguf_file).name, quantized)
    return transformer.eval()
