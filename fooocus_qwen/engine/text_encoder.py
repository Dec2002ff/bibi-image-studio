"""Текстовый энкодер в INT8 для карт с малой видеопамятью.

Энкодер Qwen-Image-2.1 — Qwen3-VL 8B, 16.3 ГБ в bf16: больше, чем вся
видеопамять RTX 4060, и две трети оперативной памяти машины с 24 ГБ. Здесь он
сжимается до 8.6 ГБ (int8 по группам из 128, ``engine/quant.py``) и в таком виде
читается с диска, а на видеокарту поднимается по одному слою
(``engine/streaming.py``).

Что сжимается, а что нет:

* семь линейных слоёв каждого из 36 слоёв языковой модели — int8; это 6.9 из
  8.2 млрд параметров;
* таблица вложений — int8 по тем же группам: разжимаются только строки
  токенов промта;
* ``lm_head`` выбрасывается: пайплайн читает скрытые состояния, логиты
  словаря ему не нужны (``quant.Passthrough``);
* визуальная часть (0.4 млрд) остаётся в bf16: она мала, а от неё зависит,
  как модель видит исходник правки и референсы.

Сжатие делается один раз, из bf16-весов, уже лежащих в каталоге модели, и
пишется рядом — ``Qwen-Image-2.1-TE-INT8/`` — по файлу на каждый исходный
шард. Пик оперативной памяти — один шард, а не модель: на машине с 16 ГБ
иначе было бы не собрать. Вместе с весами пишется ``quantization.json``:
версия схемы и размеры исходных шардов. Если поменялось одно или другое,
копия считается устаревшей и пересобирается.

Чужая сжатая копия (``Comfy-Org/Qwen-Image-2.1``, ``qwen3vl_8b_w4a8``) не
взята сознательно: это 4-битный кодбук с поворотом активаций (``convrot``) и
int8-активациями, формат ComfyUI без описания, и воспроизводить его вслепую
значило бы гадать о точности, а не измерять её.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import torch

from . import quant

LOGGER = logging.getLogger(__name__)

FORMAT = "fooocus-int8-grouped"
# Версия 1 — масштаб на строку, вдвое менее точная (quant.py); копия старой
# версии пересобирается сама.
VERSION = 2
MANIFEST = "quantization.json"
INDEX = "model.safetensors.index.json"

LINEAR = re.compile(r"^model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$")
EMBEDDING = "model.language_model.embed_tokens.weight"
DROPPED = ("lm_head.weight",)


def _scale_name(weight_name: str) -> str:
    return weight_name[: -len("weight")] + "scale"


def _source_shards(source_dir: Path) -> dict[str, int]:
    """Исходные шарды и их размеры — по индексу bf16-энкодера."""
    index = json.loads((Path(source_dir) / INDEX).read_text(encoding="utf-8"))
    names = sorted(set(index["weight_map"].values()))
    return {name: (Path(source_dir) / name).stat().st_size for name in names}


def is_current(target_dir: Path, source_dir: Path | None = None) -> bool:
    """Сжатая копия на месте и собрана этой версией схемы из этих же шардов.

    Без исходного каталога (bf16-шарды удалены ради места) сверяются только
    версия и наличие файлов: сравнивать не с чем, а копия рабочая.
    """
    target_dir = Path(target_dir)
    try:
        manifest = json.loads((target_dir / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        return False
    shards = manifest.get("shards", {})
    if not shards or not all(_present(target_dir / name) for name in shards):
        return False
    if source_dir is not None and (Path(source_dir) / INDEX).is_file():
        try:
            return _source_shards(source_dir) == manifest.get("source", {})
        except OSError:
            return False
    return True


def _present(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def convert(source_dir: Path, target_dir: Path, device: str | None = None, out=LOGGER.info) -> None:
    """Сжимает bf16-энкодер в INT8 и пишет копию в ``target_dir``.

    Шард за шардом: читается отображением файла, сжимается (на видеокарте,
    если она есть, — в десятки раз быстрее), пишется и отпускается. Запись
    — через временное имя: оборванная на середине сборка не оставит
    полуфайла, который сошёл бы за готовый.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_dir, target_dir = Path(source_dir), Path(target_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    source = _source_shards(source_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / MANIFEST).unlink(missing_ok=True)

    started = time.perf_counter()
    weight_map: dict[str, str] = {}
    for position, shard in enumerate(source, start=1):
        out(f"Text encoder INT8: shard {position}/{len(source)} ({shard})")
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(source_dir / shard), framework="pt") as handle:
            for name in handle.keys():
                if name in DROPPED:
                    continue
                tensor = handle.get_tensor(name)
                if LINEAR.match(name) or name == EMBEDDING:
                    q, scale = quant.quantize(tensor.to(device))
                    tensors[name] = q.cpu()
                    tensors[_scale_name(name)] = scale.cpu()
                else:
                    tensors[name] = tensor.to(torch.bfloat16).contiguous()
        partial = target_dir / (shard + ".partial")
        save_file(tensors, str(partial), metadata={"format": "pt"})
        partial.replace(target_dir / shard)
        weight_map.update({name: shard for name in tensors})
        del tensors

    (target_dir / INDEX).write_text(json.dumps({"weight_map": weight_map}, indent=1), encoding="utf-8")
    manifest = {"format": FORMAT, "version": VERSION, "source": source, "shards": sorted(set(weight_map.values()))}
    # Манифест — последним: он и есть отметка «готово».
    (target_dir / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    out(f"Text encoder INT8 ready in {time.perf_counter() - started:.0f} s: {target_dir}")


def ensure(source_dir: Path, target_dir: Path, out=LOGGER.info) -> bool:
    """Собирает сжатую копию, если её нет или она устарела. ``True`` — собиралась."""
    if is_current(target_dir, source_dir):
        return False
    if not (Path(source_dir) / INDEX).is_file():
        raise FileNotFoundError(
            f"Cannot build the INT8 text encoder: bf16 weights are missing in {source_dir}. "
            "Run the installation (--fetch-model) to download them."
        )
    convert(source_dir, target_dir, out=out)
    return True


_DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "I8": torch.int8, "U8": torch.uint8, "I32": torch.int32, "I64": torch.int64,
}


def read_mapped(path: Path) -> dict[str, torch.Tensor]:
    """Тензоры safetensors поверх отображения файла **только для чтения**.

    ``safetensors.torch.load_file`` на Windows отображает файл с
    копированием при записи, и система сразу резервирует под весь файл
    частную память: 8.4 ГБ энкодера ложились в выделенную память процесса
    (предел — ОЗУ плюс файл подкачки) ещё до первого чтения. Отображение
    только для чтения ничего не резервирует: страницы — это страницы файла,
    их система читает по требованию и вытесняет без записи в подкачку.

    Тензоры не записываемы — и не должны: веса хоста только читаются
    (копия уезжает на карту, назад ставится тот же тензор).
    """
    import mmap
    import struct
    import warnings

    with open(path, "rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    size = struct.unpack("<Q", mapped[:8])[0]
    header = json.loads(mapped[8 : 8 + size])
    base = 8 + size
    tensors: dict[str, torch.Tensor] = {}
    with warnings.catch_warnings():
        # «The given buffer is not writable» — ровно то, чего мы добиваемся.
        warnings.filterwarnings("ignore", message=r".*not writable.*")
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = _DTYPES[info["dtype"]]
            start, end = info["data_offsets"]
            count = (end - start) // torch.empty((), dtype=dtype).element_size()
            if count == 0:
                tensors[name] = torch.empty(info["shape"], dtype=dtype)
                continue
            flat = torch.frombuffer(mapped, dtype=dtype, count=count, offset=base + start)
            tensors[name] = flat.reshape(info["shape"])
    return tensors


def load(config_dir: Path, weights_dir: Path):
    """``Qwen3VLForConditionalGeneration`` с весами из сжатой копии.

    Модель строится по конфигурации без памяти под веса; линейные слои
    языковой модели и таблица вложений заменяются на int8-версии, голова —
    на ``Passthrough``. Тензоры читаются отображением файла и ставятся на
    места без копирования: в оперативной памяти они занимают ровно те
    страницы, которые сейчас нужны, и система может их вытеснить.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(str(config_dir))
    with init_empty_weights(include_buffers=False):
        model = Qwen3VLForConditionalGeneration._from_config(config, dtype=torch.bfloat16)

    # Замены — тоже без памяти (``meta``): настоящие пустые тензоры int8 — это
    # 7.7 ГБ, которые через миг заменятся весами из файла, а освобождённое
    # куча Windows процессу обратно не отдаёт.
    with torch.device("meta"):
        for name, module in list(model.named_modules()):
            if LINEAR.match(name + ".weight"):
                quant.replace_module(model, name, quant.Int8Linear(module.in_features, module.out_features))
        vocab, width = model.model.language_model.embed_tokens.weight.shape
        quant.replace_module(model, "model.language_model.embed_tokens", quant.Int8Embedding(vocab, width))
    model.lm_head = quant.Passthrough()

    weights_dir = Path(weights_dir)
    manifest = json.loads((weights_dir / MANIFEST).read_text(encoding="utf-8"))
    state: dict[str, torch.Tensor] = {}
    for shard in manifest["shards"]:
        state.update(read_mapped(weights_dir / shard))
    problems = quant.assign(model, state)
    if problems:
        raise RuntimeError(
            f"INT8 text encoder does not match the model: {len(problems)} mismatches, first: {problems[0]}. "
            f"Delete {weights_dir} and it will be rebuilt."
        )
    return model.eval()
