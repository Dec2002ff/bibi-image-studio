"""Опрос о производительности при установке: точность весов и SageAttention.

Два вопроса, оба с разумным ответом по умолчанию, и оба можно поменять
потом во вкладке «Настройки»:

* **точность трансформера** — bf16 (исходные веса, 13.3 ГиБ видеопамяти),
  INT8 (веса Unsloth, 6.8 ГиБ, скорость почти та же) или вариант GGUF
  (Unsloth, от 6.6 ГиБ у Q8_0 до 2.7 у Q3_K_M; Q4_K_M — для 6–8 ГБ). От
  ответа зависит, что качает следующий шаг установки: при INT8 и GGUF
  bf16-шарды трансформера (14 ГБ) не нужны вовсе. По умолчанию предлагается
  то, что подходит к найденной видеокарте (``settings.recommended_precision``);
  профиль памяти не спрашивается — «auto» решает его по той же карте, а
  поменять можно во вкладке «Настройки»;
* **SageAttention** — внимание на −15…25 % быстрее. Пакет необязательный и
  ставится отдельно: под Windows — готовая сборка
  (github.com/woct0rdho/SageAttention) плюс ``triton-windows`` той версии,
  что совместима с установленным torch.

Отказ отвечать — полноправный ответ: остаётся текущий выбор, а если его
не было — точность, рекомендованная для этой карты, без SageAttention.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from .. import settings as settings_module

LOGGER = logging.getLogger(__name__)

SAGE_RELEASE = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/"
# Сборки выпуска v2.2.0-windows.post6: для torch 2.10 и новее, CUDA 12.8 и 13.0.
SAGE_WHEELS = {
    "cu128": "sageattention-2.2.0+cu128torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl",
    "cu130": "sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl",
}
SAGE_SOURCE = "git+https://github.com/thu-ml/SageAttention.git"


def configure(
    path: Path | None = None,
    ask: Callable[[str], str] = input,
    out: Callable[..., None] = print,
    install_sage: Callable[[Callable[..., None]], bool] | None = None,
    sage_available: Callable[[], bool] | None = None,
    vram_gib: float | None | object = ...,
) -> settings_module.Settings:
    """Задаёт оба вопроса, сохраняет ответы и возвращает итоговые настройки.

    ``vram_gib`` — объём видеопамяти; по умолчанию определяется
    (``engine/hardware.py``), тесты подставляют свой.
    """
    ask = _forgiving(ask)
    install_sage = install_sage or install_sage_attention
    sage_available = sage_available or _sage_importable
    if vram_gib is ...:
        from . import hardware

        vram_gib = hardware.vram_gib()
    settings_path = path or settings_module.config.SETTINGS_FILE
    current = settings_module.load(path)
    recommended = settings_module.recommended_precision(vram_gib)
    known = current.precision if settings_path.is_file() else recommended

    card = f"{vram_gib:.1f} GiB" if vram_gib is not None else "no CUDA device"
    profile = settings_module.resolve_profile(current.memory_profile, vram_gib)
    out(f"  Video card: {card}; memory profile: {profile} (change it on the Settings tab)")
    out("  Transformer weight precision:")
    options = list(settings_module.PRECISIONS)
    for number, name in enumerate(options, start=1):
        mark = "  <- recommended for this card" if name == recommended else ""
        out(f"    {number} - {_PRECISION_TEXT[name]}{mark}")
    default = str(options.index(known) + 1)
    answer = ask(f"  Choice (Enter = {default}): ").strip() or default
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        precision = options[int(answer) - 1]
    else:
        precision = known
        out(f"  Did not understand '{answer}', keeping {precision}")

    out("  SageAttention speeds up generation by 15-25%; the package is installed separately.")
    default_sage = "yes" if current.sage_attention else "no"
    reply = ask(f"  Install and enable SageAttention? [yes/no] (Enter = {default_sage}): ").strip().lower()
    wants_sage = current.sage_attention if not reply else reply in ("да", "д", "yes", "y")

    if wants_sage and not sage_available():
        wants_sage = install_sage(out)

    chosen = settings_module.Settings(
        precision=precision, sage_attention=wants_sage, memory_profile=current.memory_profile
    )
    settings_module.save(chosen, path)
    out(f"  Saved: precision {precision}, SageAttention {'enabled' if wants_sage else 'disabled'}")
    return chosen


_PRECISION_TEXT = {
    "bf16": "bf16: original precision, 13.3 GiB VRAM, ~33 GB of weights (24 GB cards)",
    "int8": "INT8: 6.8 GiB VRAM, nearly the same speed, ~26 GB of weights (12-24 GB cards)",
    "Q8_0": "GGUF Q8_0: 6.6 GiB VRAM, closest to bf16 (10-16 GB cards)",
    "Q6_K": "GGUF Q6_K: 5.4 GiB VRAM (10-12 GB cards)",
    "Q5_K_M": "GGUF Q5_K_M: 4.7 GiB VRAM (8-10 GB cards, tight on 8)",
    "Q4_K_M": "GGUF Q4_K_M: 3.9 GiB VRAM, the choice for 6-8 GB cards",
    "Q4_K_S": "GGUF Q4_K_S: 3.4 GiB VRAM, fallback if Q4_K_M runs out of memory",
    "Q3_K_M": "GGUF Q3_K_M: 2.7 GiB VRAM, visible quality loss; last resort",
}


def install_sage_attention(out: Callable[..., None] = print, python: str | None = None) -> bool:
    """Ставит SageAttention и Triton в текущее окружение. ``True`` — удалось.

    Неудача не роняет установку: SageAttention — ускорение, а не условие
    работы. Причина печатается, приложение работает штатным вниманием.
    """
    python = python or sys.executable
    plan = sage_install_plan()
    if plan is None:
        return False
    if isinstance(plan, str):
        out(f"  {plan}")
        return False
    for step in plan:
        out(f"  pip install {' '.join(step)}")
        result = subprocess.run([python, "-m", "pip", "install", *step], check=False)
        if result.returncode != 0:
            out("  SageAttention failed to install; using default attention, everything else works.")
            return False
    ok = subprocess.run(
        [python, "-c", "from diffusers.models import attention_dispatch as d; raise SystemExit(0 if d._CAN_USE_SAGE_ATTN else 1)"],
        check=False,
    ).returncode == 0
    out("  SageAttention installed" if ok else "  SageAttention installed, but diffusers does not see it: keeping default attention")
    return ok


def sage_install_plan(torch_version: str | None = None, cuda: str | None = None, platform: str | None = None):
    """Что ставить: список аргументов pip по шагам или строка-объяснение, почему нечего.

    Разделено с установкой, чтобы подбор сборки проверялся тестами без сети.
    """
    platform = platform or sys.platform
    if torch_version is None or cuda is None:
        try:
            import torch
        except ImportError:
            return "torch is not installed: nothing to install SageAttention for."
        torch_version = torch_version or torch.__version__
        cuda = cuda if cuda is not None else (torch.version.cuda or "")

    if not platform.startswith("win"):
        return (
            "There are no prebuilt SageAttention 2 wheels for Linux; build from source: "
            f"pip install {SAGE_SOURCE} (requires the CUDA Toolkit). Using default attention for now."
        )
    major_minor = _major_minor(torch_version)
    if major_minor is None or major_minor < (2, 10):
        return f"No SageAttention wheel for torch {torch_version} (2.10 or newer is required)."
    cuda_tag = "cu" + cuda.replace(".", "")
    wheel = SAGE_WHEELS.get(cuda_tag)
    if wheel is None:
        return f"No SageAttention wheel for CUDA {cuda} (available for 12.8 and 13.0)."
    # triton-windows 3.N работает с torch 2.(N+4): 3.6 — 2.10, 3.7 — 2.11.
    triton_minor = major_minor[1] - 4
    triton = f"triton-windows>=3.{triton_minor},<3.{triton_minor + 1}"
    return [[triton], [SAGE_RELEASE + wheel]]


def _major_minor(version: str) -> tuple[int, int] | None:
    try:
        major, minor = version.split("+")[0].split(".")[:2]
        return int(major), int(minor)
    except ValueError:
        return None


def _sage_importable() -> bool:
    from . import attention

    return attention.sage_available()


def _forgiving(ask: Callable[[str], str]) -> Callable[[str], str]:
    """Конец ввода и Ctrl+C — «оставить как есть» (см. ``llm/setup.py``)."""

    def guarded(question: str) -> str:
        try:
            return ask(question)
        except (EOFError, KeyboardInterrupt):
            return ""

    return guarded
