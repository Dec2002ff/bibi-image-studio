"""Плитка для своей позы: героиня каталога в этой позе, нарисованная Qwen-Image.

Подача выбрана опытом ``tools/experiments/pose_tile.py`` (разбор — в
``docs/research/2026-09-26-pozy.md``): единственный референс — скелет, а
образ и стиль каталога описаны словами. Второй референс — плитка каталога
как образец — не работает: модель копирует образец вместе с его позой и
скелет игнорирует. Со скелетом одним поза держится: ошибка 0.7–2.6 %
размера фигуры по DWPose на результате.

На непривычных позах (лёжа, вверх ногами, на четвереньках) одного скелета
мало: модель по привычке рисует голову сверху. Поэтому к промту добавляется
поза словами из точек скелета (``poses.describe``), и рисуются два
кандидата, из которых остаётся тот, чья поза по DWPose ближе к скелету
(``best``). Опыт ``tools/experiments/pose_tile_prompt.py`` (2026-09-30):
перевёрнутая поза «Nu08» — ошибка 51 и 32 % без описания, 15 и 27 % с ним;
пояснение цветового кода скелета вредит (на ней же — ни одной фигуры).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PIL import Image

from ..engine import presets
from ..engine.generator import GenerationRequest
from . import describe, detect
from . import skeleton as sk

# Стиль каталога openposes.com словами: живопись крупным мазком, фон —
# разводы краски, одежда повседневная, героиня одна.
PROMPT = (
    "Full-body digital painting of Emma Watson, a young woman with shoulder-length wavy light "
    "brown hair, in a casual outfit, loose expressive painterly brushstrokes, soft cinematic light, "
    "the background is an abstract swirl of paint splashes in teal, blue and violet on a pale grey "
    "canvas. She is posed exactly like the pose skeleton in the reference image: the colored lines "
    "mark her head, shoulders, arms, hips and legs. Do not draw the skeleton lines."
)

# Turbo рисует плитку за ~13 с на RTX 3090 и чист именно на 1024², а это и
# есть размер плитки. Без его адаптера — самый быстрый из обычных пресетов:
# качать ради иконки 1.3 ГБ без спроса незачем.
FAST_PRESET = "Turbo"
FALLBACK_PRESET = "LowQuality"
# Кандидатов на обложку: из них остаётся ближайший к скелету по позе.
CANDIDATES = 2

# Перестановка левых и правых точек BODY_18 — для зеркальной меры.
_MIRROR = {2: 5, 3: 6, 4: 7, 8: 11, 9: 12, 10: 13, 14: 15, 16: 17}
_MIRROR.update({value: key for key, value in _MIRROR.items()})


def prompt_for(pose: sk.Pose | None) -> str:
    """Промт обложки: стиль каталога и, если поза известна, она же словами."""
    words = describe.describe(pose) if pose is not None else ""
    return f"{PROMPT} {words}".strip()


def request(
    skeleton_image: Image.Image, turbo_ready: bool, seed: int = -1, pose: sk.Pose | None = None,
) -> GenerationRequest:
    preset = presets.get(FAST_PRESET if turbo_ready else FALLBACK_PRESET)
    prompt = prompt_for(pose)
    return GenerationRequest(
        prompt=prompt,
        prompt_original=prompt,
        preset=preset,
        references=(skeleton_image.convert("RGB"),),
        aspect="1:1",
        seed=seed,
        image_number=CANDIDATES,
    )


def _normalized(points: np.ndarray, visible: np.ndarray) -> np.ndarray:
    seen = points[visible]
    low, high = seen.min(0), seen.max(0)
    return (points - (low + high) / 2) / max(float((high - low).max()), 1.0)


def pose_error(target: sk.Pose, result: sk.Pose) -> tuple[float, float]:
    """Насколько поза результата далека от заданной: (прямо, зеркально), % рамки.

    Обе приводятся к общей рамке (центр и размер), ошибка — средняя по точкам,
    видимым в обеих; «зеркально» — с переставленными левой и правой сторонами.
    Меньше четырёх общих точек — ``nan``.
    """
    a = np.array([p[:2] for p in target.points])
    b = np.array([p[:2] for p in result.points])
    va = np.array([p[2] > 0 for p in target.points])
    vb = np.array([p[2] > 0 for p in result.points])
    both = va & vb
    if both.sum() < 4:
        return float("nan"), float("nan")
    na, nb = _normalized(a, va), _normalized(b, vb)
    direct = float(np.linalg.norm(na[both] - nb[both], axis=1).mean() * 100)
    order = [_MIRROR.get(i, i) for i in range(sk.POINTS)]
    nm, vm = nb[order], vb[order]
    both_m = va & vm
    mirrored = float(np.linalg.norm(na[both_m] - nm[both_m], axis=1).mean() * 100)
    return direct, mirrored


def best(candidates: Sequence[Image.Image], pose: sk.Pose, detector) -> Image.Image:
    """Кандидат, чья поза по DWPose ближе к скелету; без человека — в конец."""
    def score(image: Image.Image) -> float:
        try:
            direct, _mirrored = pose_error(pose, detect.to_pose(detector.detect(image)))
        except detect.NoPersonFound:
            return math.inf
        return math.inf if math.isnan(direct) else direct

    return min(candidates, key=score)
