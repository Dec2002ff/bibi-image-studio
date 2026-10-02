"""INT8 «только веса» по группам и постановка готовых тензоров на места модели.

Своё, а не torchao, по одной причине: веса здесь — обычные тензоры
(``int8`` и ``float32``), а не подклассы. Их переносит между устройствами
простое присваивание ``.data`` — так, как это делают ``engine/residency.py``
и ``engine/streaming.py``, — без той возни с подклассами, которой потребовал
INT8-трансформер Unsloth (``residency.place_tensor``).

Схема — симметричная, масштаб на группу из ``GROUP`` входов строки:
``w ≈ q · s``, где ``s = max|w| / 127`` по группе. Масштаб на всю строку
(«по строкам») вдвое хуже: выбросы весов в строке огрубляют шаг для всех
остальных. Замер на энкодере (относительная ошибка эмбеддингов промта к
вычислению в fp32): bf16 — 5–6 %, int8 по группам 128 — 5.5–8 %, по строкам
— 11–15 % (``tools/experiments/lowvram_te_sensitivity.py``,
``docs/research/2026-10-02-8-gb.md``). Масштабы — float32: 3 % к объёму.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

GROUP = 128


def quantize(weight: torch.Tensor, group: int = GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q, scale)``: int8 веса ``[out, in]`` и масштабы float32 ``[out, in / group]``.

    Считается в float32 на том устройстве, где лежит ``weight``: у bf16
    всего восемь бит мантиссы, и деление на масштаб в нём само добавило бы
    ошибку того же порядка, что и квантование.
    """
    rows, width = weight.shape
    if width % group:
        raise ValueError(f"width {width} is not a multiple of the group {group}")
    values = weight.float().reshape(rows, width // group, group)
    scale = values.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny) / 127.0
    q = torch.round(values / scale[..., None]).clamp_(-127, 127).to(torch.int8)
    return q.reshape(rows, width), scale


def dequantize(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Обратно к весу: разжатие в float32 и одно округление в ``dtype``."""
    rows, width = q.shape
    groups = scale.shape[-1]
    values = q.float().reshape(rows, groups, width // groups) * scale[..., None]
    return values.reshape(rows, width).to(dtype)


class Int8Linear(nn.Module):
    """Линейный слой без смещения с весом int8 и масштабом на группу входов.

    Вес разжимается в float32 и округляется в bf16 перед умножением — тот же
    путь, каким получились бы исходные bf16-веса, — и отпускается сразу
    после. Временная память — один слой, не модель.
    """

    def __init__(self, in_features: int, out_features: int, group: int = GROUP) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.scale = nn.Parameter(
            torch.empty(out_features, in_features // group, dtype=torch.float32), requires_grad=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, dequantize(self.weight, self.scale, inputs.dtype))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8"


class Int8Embedding(nn.Module):
    """Таблица вложений int8 с масштабами по группам; разжимаются только нужные строки.

    Считает там, где лежит таблица, и отдаёт строки туда, откуда пришли
    номера: в профиле «low» таблица (0.64 ГБ) остаётся в оперативной памяти,
    а на карту уезжают только строки промта.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, group: int = GROUP, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=torch.int8), requires_grad=False
        )
        self.scale = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim // group, dtype=torch.float32), requires_grad=False
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).to(self.weight.device)
        rows = dequantize(F.embedding(flat, self.weight), F.embedding(flat, self.scale), self.dtype)
        return rows.reshape(*ids.shape, self.embedding_dim).to(ids.device)


class Passthrough(nn.Module):
    """Возвращает вход. Замена ``lm_head``: логиты словаря энкодеру не нужны.

    Пайплайн читает скрытые состояния, а не логиты. Настоящая голова — это
    1.24 ГБ весов и, на промте с референсами, ещё гигабайт активаций
    (151 936 логитов на каждый токен) ради результата, который выбрасывается.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def replace_module(model: nn.Module, name: str, module: nn.Module) -> None:
    owner_name, _, attribute = name.rpartition(".")
    owner = model.get_submodule(owner_name) if owner_name else model
    setattr(owner, attribute, module)


def assign(module: nn.Module, state: dict[str, torch.Tensor]) -> list[str]:
    """Ставит тензоры на места параметров; возвращает список расхождений.

    ``load_state_dict`` годится не всегда: он сверяет форму, а у сжатого
    тензора GGUF она байтовая, не логическая. Поэтому соответствие
    проверяется по именам: лишний ключ в файле и параметр, оставшийся без
    значения (на устройстве ``meta``), — оба расхождения.
    """
    parameters = dict(module.named_parameters())
    problems = [f"unexpected {name}" for name in state if name not in parameters]
    for name, value in state.items():
        if name not in parameters:
            continue
        owner_name, _, attribute = name.rpartition(".")
        owner = module.get_submodule(owner_name) if owner_name else module
        if not isinstance(value, nn.Parameter):
            value = nn.Parameter(value, requires_grad=False)
        owner._parameters[attribute] = value
    problems += [f"missing {name}" for name, tensor in module.named_parameters() if tensor.is_meta]
    return problems
