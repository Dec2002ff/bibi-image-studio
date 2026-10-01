"""Поблочная подача весов и KV-кэш в оперативной памяти — для карт на 6–12 ГБ.

На 24 ГБ трансформер и энкодер меняются на карте местами целиком
(``residency.StagedModule``). На 8 ГБ не помещается ни один из них целиком
вместе с другим, а энкодер не помещается и в одиночку. Поэтому здесь иначе:
трансформер (GGUF, 3.9 ГБ) не покидает карту, а энкодер живёт на хосте и
поднимается **по одному блоку** на время вызова этого блока — пред-хук
поднимает веса, пост-хук отпускает. Остальное (таблица вложений, нормы,
проекции визуальной части) на время кодирования поднимается целиком: оно
мало.

Цена — проход весов энкодера по шине на каждый промах кэша эмбеддингов:
8.4 ГБ int8 за ~1–2 с (``docs/research/2026-10-02-8-gb.md``). Это
дешевле, чем выгнать и вернуть трансформер, и не требует держать в
оперативной памяти закреплённые копии: веса энкодера — страницы файла
(``engine/text_encoder.py``), их система вытесняет сама.

KV-кэш префикса (текст и условные изображения, ``QwenImage21KVCache``)
считается на первом шаге и читается на остальных. На правке кадра 1024² он
весит 2.1 ГБ — четверть карты. Такой кэш уезжает в оперативную память и
возвращается на карту по блоку на каждом шаге; маленький (промт без
картинок, десятки мегабайт) остаётся на карте.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
from torch import nn

from .residency import _named_tensors, _nbytes, _place

LOGGER = logging.getLogger(__name__)


class StreamedModule:
    """Модуль на хосте, чьи блоки поднимаются на устройство на время своего вызова.

    Тот же интерфейс, что у ``residency.StagedModule``: ``to_device()`` —
    «готов к вызову» (общая часть на устройстве, хуки на блоках),
    ``to_host()`` — всё на хосте, хуков нет. Копии на хосте не создаются:
    тензоры модуля и есть канонические копии.
    """

    def __init__(
        self,
        module: nn.Module,
        blocks: Iterable[nn.Module],
        device: str | torch.device,
        host_modules: Iterable[nn.Module] = (),
    ) -> None:
        """``host_modules`` — части, которые не поднимаются вовсе и считают на
        хосте (таблица вложений энкодера: поиск строк дёшев, а 0.64 ГБ на
        карте — нет)."""
        self.module = module
        self._device = torch.device(device)
        self._blocks = list(blocks)
        block_ids = {id(block) for block in self._blocks}
        self._prefixes = {id(child): name + "." for name, child in module.named_modules() if id(child) in block_ids}
        prefixes = tuple(self._prefixes.values())
        if len(prefixes) != len(self._blocks):
            raise ValueError("every streamed block must be a submodule of the streamed module")

        self._host: dict[str, torch.Tensor] = {}
        self._nbytes = 0
        for name, tensor in _named_tensors(module):
            host = tensor.detach()
            if host.device.type != "cpu":
                host = host.to("cpu")
                _place(module, name, tensor, host)
            self._host[name] = host
            self._nbytes += _nbytes(host)
        # Общая часть — всё, что не внутри блоков. Ключи — как у _named_tensors:
        # «p:имя» и «b:имя».
        host_ids = {id(child) for child in host_modules}
        pinned_to_host = tuple(name + "." for name, child in module.named_modules() if id(child) in host_ids)
        self._shared = {
            key for key in self._host
            if not key[2:].startswith(prefixes) and not (pinned_to_host and key[2:].startswith(pinned_to_host))
        }
        self._handles: list = []
        self._on_device = False

    @property
    def resident(self) -> bool:
        return self._on_device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def to_device(self) -> None:
        if self._on_device:
            return
        self._on_device = True
        try:
            for name, tensor in _named_tensors(self.module):
                if name in self._shared:
                    _place(self.module, name, tensor, self._host[name].to(self._device))
            for block in self._blocks:
                self._handles.append(block.register_forward_pre_hook(self._lift))
                self._handles.append(block.register_forward_hook(self._drop))
        except BaseException:
            self.to_host()
            raise

    def to_host(self) -> None:
        """Всё на хост и хуки долой. Вызывается и после сбоя посреди блока."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for name, tensor in _named_tensors(self.module):
            if tensor.device.type != "cpu":
                _place(self.module, name, tensor, self._host[name])
        self._on_device = False
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def _lift(self, block: nn.Module, _args) -> None:
        for name, tensor in _named_tensors(block):
            _place(block, name, tensor, tensor.to(self._device))

    def _drop(self, block: nn.Module, _args, _output) -> None:
        # Возврат — не копия: ставится обратно тот же тензор хоста. Он не
        # менялся, а копия на устройстве освобождается вместе с последней
        # ссылкой на неё.
        prefix = self._prefixes[id(block)]
        for name, tensor in _named_tensors(block):
            _place(block, name, tensor, self._host[name[:2] + prefix + name[2:]])


# --- KV-кэш префикса в оперативной памяти ------------------------------------------


class _HostLayer:
    """Кэш одного блока; где лежат тензоры, решает общий для всех блоков ``HostKVCache``."""

    def __init__(self, owner: HostKVCache) -> None:
        self._owner = owner
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if self._owner.offload(k.nbytes + v.nbytes):
            k, v = k.to("cpu"), v.to("cpu")
        self.k, self.v = k, v

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            raise RuntimeError("KV cache has not been populated yet.")
        return self.k.to(self._owner.device), self.v.to(self._owner.device)


class HostKVCache:
    """Замена ``QwenImage21KVCache``: крупный кэш живёт на хосте.

    Решение «на хост или на карту» принимается один раз, по первому блоку:
    размер кэша одинаков во всех блоках, и половинчатое размещение ничего
    бы не дало.
    """

    def __init__(self, num_layers: int, device: torch.device, threshold_bytes: int) -> None:
        self.device = device
        self._threshold = threshold_bytes
        self._num_layers = num_layers
        self._offload: bool | None = None
        self.layer_caches = [_HostLayer(self) for _ in range(num_layers)]

    def offload(self, layer_bytes: int) -> bool:
        if self._offload is None:
            total = layer_bytes * self._num_layers
            self._offload = total > self._threshold
            if self._offload:
                LOGGER.debug("KV cache %.2f GiB goes to host memory", total / 2**30)
        return self._offload

    @property
    def offloaded(self) -> bool:
        return bool(self._offload)

    def get_layer(self, layer_idx: int) -> _HostLayer:
        return self.layer_caches[layer_idx]


# Порог, выше которого кэш уезжает на хост. Промт без картинок — десятки
# мегабайт, и гонять их по шине незачем; исходник правки 1024² — 2.1 ГиБ.
KV_THRESHOLD_BYTES = 256 * 2**20


def offload_kv_cache(transformer: nn.Module, device: torch.device, threshold_bytes: int = KV_THRESHOLD_BYTES):
    """Подменяет KV-кэш, который пайплайн передаёт трансформеру, на ``HostKVCache``.

    Кэш создаёт ``__call__`` пайплайна, и переопределять ради этого цикл
    денойзинга не хочется (``ARCHITECTURE.md``, раздел 2). Поэтому подмена —
    на входе трансформера: пред-хук с аргументами заменяет ``kv_cache`` на
    свою копию, привязанную к исходному объекту. Объект живёт один вызов
    пайплайна, и вместе с ним уходит и подмена. Трансформеру нужен от кэша
    только ``get_layer`` — его и даёт замена.

    Возвращает дескриптор хука (``remove()`` снимает подмену).
    """

    def swap(_module, args, kwargs):
        original = kwargs.get("kv_cache")
        if original is None or isinstance(original, HostKVCache):
            return None
        replacement = getattr(original, "_studio_host", None)
        if replacement is None:
            replacement = HostKVCache(len(original.layer_caches), device, threshold_bytes)
            original._studio_host = replacement
        kwargs["kv_cache"] = replacement
        return args, kwargs

    return transformer.register_forward_pre_hook(swap, with_kwargs=True)


def text_encoder_blocks(text_encoder: nn.Module) -> list[nn.Module]:
    """Блоки энкодера Qwen3-VL, подаваемые по одному: слои языковой модели и визуальной части.

    Пути — по устройству ``Qwen3VLForConditionalGeneration`` в transformers
    5: ``model.language_model.layers`` (36) и ``model.visual.blocks`` (27).
    Изменится устройство — упадёт здесь, при загрузке, с понятной причиной, а
    не тихим подъёмом энкодера целиком.
    """
    try:
        return [*text_encoder.model.language_model.layers, *text_encoder.model.visual.blocks]
    except AttributeError as error:
        raise RuntimeError(
            f"Unexpected text encoder layout ({type(text_encoder).__name__}): {error}. "
            "fooocus_qwen/engine/streaming.py expects Qwen3-VL from transformers 5."
        ) from error


def text_encoder_host_modules(text_encoder: nn.Module) -> list[nn.Module]:
    """Части энкодера, которые считают на хосте: таблица вложений (int8 умеет).

    Обычная ``nn.Embedding`` (bf16-энкодер) считать на хосте с номерами на
    карте не умеет — её не трогаем.
    """
    from .quant import Int8Embedding

    embedding = text_encoder.model.language_model.embed_tokens
    return [embedding] if isinstance(embedding, Int8Embedding) else []
