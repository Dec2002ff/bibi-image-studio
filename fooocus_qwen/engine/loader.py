"""Сборка пайплайна: загрузка весов, размещение, вспомогательные режимы."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import torch

from . import attention, vae_tiling
from .embeds_cache import EmbedsCache
from .pipeline import QwenImage21StudioPipeline, assert_contract
from .plan import STREAM, SWAP
from .residency import ResidencyManager

LOGGER = logging.getLogger(__name__)



# Плитка и шаг для тайлинга VAE. Размер плитки задаёт память, шаг — как
# часто встречается шов. Полосы убирает не размер, а сам декодер
# (``engine/vae_tiling.py``): измерено на одном латенте, кадр 1280x1888,
# цветные линии считаются по хроматической мерке —
#
#   вариант                линий на решётке плиток   с      ГиБ
#   цельное декодирование                        0   1.3   15.77
#   512/256, штатный tiled_decode                5   5.7    2.37
#   512/256, здешний декодер                     0   4.3    2.36
#
# То есть здешний тайлинг неотличим от цельного декодирования по швам и
# стоит при этом в семь раз меньше памяти
# (docs/research/2026-09-22-shvy-ot-kraev-plitok-vae.md).
VAE_TILE = 512
VAE_TILE_STRIDE = 256
# Профиль «low»: на 8 ГБ пик денойзинга даёт декодирование VAE, а не
# трансформер. Плитка 384 — +0.95 ГиБ к весам VAE против +1.67 у 512; на
# настоящем кадре 1024² расхождение с цельным декодированием 1.9 уровня из
# 255 в среднем (у 512 — 1.3), худший столбец 4.0 (у 512 — 2.7). Плитка 256
# дешевле ещё вдвое, но её худший столбец — 8.9 уровня: это уже полоса
# (tools/experiments/lowvram_vae_tile.py, docs/research/2026-10-02-8-gb.md).
VAE_TILE_LOW = 384


def _configure_vae_tiling(pipe, tile: int = VAE_TILE) -> None:
    """Включает тайлинг VAE и заменяет его декодер на бесшовный.

    Сам тайлинг нужен: цельное декодирование кадра 1280x1888 требует 15.8
    ГиБ поверх резидентного трансформера в 13.3 — вместе это больше карты,
    и она уходит в вытеснение (23 с на декодирование вместо двух).

    Шаг задаётся полем напрямую: ``enable_tiling`` его не принимает.

    Размер плитки полос не убирает — он их только разрежает. Убирает их
    замена самого тайлового декодера: штатный смешивает плитки по всему
    перекрытию и потому вносит в кадр их края, где декодер врёт в тридцать
    пять раз сильнее, чем в середине. Здешний края отбрасывает
    (``engine/vae_tiling.py``).
    """
    pipe.vae.enable_tiling(
        tile_sample_min_height=tile,
        tile_sample_min_width=tile,
    )
    # Шаг — половина плитки: перекрытие должно покрывать отбрасываемый край
    # плитки (vae_tiling.TRIM, 64 px).
    pipe.vae.tile_sample_stride_height = tile // 2
    pipe.vae.tile_sample_stride_width = tile // 2
    vae_tiling.install(pipe.vae)


def load_int8_transformer(model_dir: Path, int8_file: Path):
    """Трансформер с INT8-весами Unsloth, в режиме «только веса».

    Файл — сериализация torchao: 224 линейных слоя в ``Int8Tensor``, 73
    тензора (нормы, смещения, вложения) в bf16. Модель строится по
    конфигурации основной модели без выделения памяти под веса
    (``init_empty_weights``; буферы — настоящие: частоты временного вложения
    в файле не хранятся) и получает тензоры из файла как есть.

    Квантование активаций, с которым Unsloth отдаёт веса (W8A8), снимается.
    На RTX 3090 перемножение в INT8 не быстрее bf16, а квантование
    активаций на каждом слое добавляет 60 % ко времени шага: 33.9 с против
    21.6 с на кадре 1024² (bf16 — 20.5 с). Веса при этом остаются INT8 —
    видеопамять та же, 6.8 ГиБ вместо 13.3.
    """
    from accelerate import init_empty_weights
    from diffusers import QwenImage21Transformer2DModel
    from safetensors import safe_open
    from torchao.prototype.safetensors.safetensors_support import unflatten_tensor_state_dict

    config = QwenImage21Transformer2DModel.load_config(str(Path(model_dir) / "transformer"))
    with init_empty_weights(include_buffers=False):
        transformer = QwenImage21Transformer2DModel.from_config(config)

    with safe_open(str(int8_file), framework="pt") as handle:
        metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    state, leftover = unflatten_tensor_state_dict(tensors, metadata)
    result = transformer.load_state_dict(state, strict=False, assign=True)
    problems = [*result.missing_keys, *result.unexpected_keys, *leftover]
    if problems:
        raise RuntimeError(
            f"INT8 weights do not match the transformer: {len(problems)} mismatches, first: {problems[0]}. "
            "The file is corrupted or belongs to another model version; delete it and it will be downloaded again."
        )

    quantized = 0
    for module in transformer.modules():
        weight = getattr(module, "weight", None)
        if weight is not None and hasattr(weight, "act_quant_kwargs"):
            weight.act_quant_kwargs = None
            quantized += 1
    LOGGER.info("INT8 transformer: %d layers in INT8 (weights only)", quantized)
    return transformer.eval()


def base_transformer_loader(
    model_dir: Path, int8_file: Path | None = None, gguf_file: Path | None = None
) -> Callable[[torch.device | None], torch.nn.Module]:
    """Строитель основного трансформера выбранной точности: ``device -> модуль``.

    Нужен дважды: при загрузке и при возврате с пресета на отдельном
    трансформере (Turbo4) в профиле «low», где копии основного на хосте нет
    и он читается с диска заново (``ResidencyManager.replace_transformer``).
    ``device=None`` — на хосте.
    """

    def build(device: torch.device | None) -> torch.nn.Module:
        if gguf_file is not None:
            from . import gguf

            return gguf.load_transformer(model_dir, gguf_file, device=device)
        if int8_file is not None:
            module = load_int8_transformer(model_dir, int8_file)
        else:
            from diffusers import QwenImage21Transformer2DModel

            module = QwenImage21Transformer2DModel.from_pretrained(
                str(model_dir), subfolder="transformer", torch_dtype=torch.bfloat16
            ).eval()
        return module.to(device) if device is not None else module

    return build


def prepare_transformer(transformer, sage_attention: bool, policy: str, device: str | torch.device) -> None:
    """Всё, что вешается на сам модуль трансформера: механизм внимания и KV-кэш в ОЗУ.

    Отдельно от загрузки — потому что трансформер бывает и подменён
    (пресет Turbo4), и подменённый должен быть подготовлен так же.
    """
    attention.apply(transformer, sage_attention)
    if policy == STREAM:
        from . import streaming

        streaming.offload_kv_cache(transformer, torch.device(device))


def load(
    model_dir: Path,
    device: str = "cuda",
    pin_memory: bool = True,
    cache_capacity: int = 4,
    int8_file: Path | None = None,
    sage_attention: bool = False,
    gguf_file: Path | None = None,
    text_encoder_dir: Path | None = None,
    policy: str = SWAP,
) -> tuple[QwenImage21StudioPipeline, ResidencyManager, EmbedsCache]:
    """Загружает модель и раскладывает её по памяти.

    Веса читаются на хост: размещением дальше управляет ResidencyManager, и
    позволить diffusers самому что-то перенести значило бы получить два хозяина
    у одной видеопамяти.

    Трансформер — bf16 из каталога модели, ``int8_file`` (INT8 Unsloth) или
    ``gguf_file`` (GGUF, ``engine/gguf.py``); ``text_encoder_dir`` — энкодер,
    сжатый в INT8 (``engine/text_encoder.py``), вместо bf16. ``policy`` —
    раскладка (``residency.SWAP`` для 24 ГБ, ``STREAM`` для 6–16 ГБ): при
    ``STREAM`` GGUF-трансформер грузится прямо на карту — копия на хосте ему
    не нужна, — а KV-кэш префикса, если он крупный, уезжает в оперативную
    память (``streaming.offload_kv_cache``). ``sage_attention`` — внимание
    SageAttention, если пакет установлен (``engine/attention.py``).
    """
    assert_contract()

    started = time.perf_counter()
    if gguf_file is not None:
        kind = f" (GGUF {Path(gguf_file).name})"
    elif int8_file is not None:
        kind = " (INT8 transformer)"
    else:
        kind = ""
    LOGGER.info("Loading model from %s%s, residency %s", model_dir, kind, policy)
    extra = {}
    if gguf_file is not None or int8_file is not None:
        build = base_transformer_loader(model_dir, int8_file=int8_file, gguf_file=gguf_file)
        extra["transformer"] = build(torch.device(device) if policy == STREAM and gguf_file is not None else None)
    if text_encoder_dir is not None:
        from . import text_encoder

        extra["text_encoder"] = text_encoder.load(Path(model_dir) / "text_encoder", text_encoder_dir)
    pipe = QwenImage21StudioPipeline.from_pretrained(str(model_dir), dtype=torch.bfloat16, **extra)

    _configure_vae_tiling(pipe, VAE_TILE_LOW if policy == STREAM else VAE_TILE)

    residency = ResidencyManager(pipe, device=device, pin_memory=pin_memory, policy=policy)
    residency.start()
    prepare_transformer(pipe.transformer, sage_attention, policy, device)

    cache = EmbedsCache(capacity=cache_capacity)
    pipe.attach(residency, cache, torch.device(device))

    LOGGER.info("Model ready in %.1f s", time.perf_counter() - started)
    return pipe, residency, cache
