"""Поза словами — подсказка модели к скелету-референсу.

Скелет OpenPose модель читает неуверенно, когда поза непривычная: лёжа,
вверх ногами, на четвереньках. Цветового кода OpenPose она не знает, и
голову по умолчанию рисует сверху. У нас же точки скелета есть, так что
положение тела в кадре вычисляется и говорится прямо: «вверх ногами, голова
внизу справа, ноги подняты к верхнему левому углу».

Стороны называются по картинке («рука слева на картинке»), а не
анатомически: лицом ли к нам фигура, по скелету не всегда ясно, а сторона
картинки однозначна. Текст — английский: так написан промт плитки.
"""

from __future__ import annotations

import math

from . import skeleton as sk

# Направления в кадре по углу (0° — вправо, 90° — вверх), восемь секторов.
_DIRECTIONS = ("right", "upper right", "up", "upper left", "left", "lower left", "down", "lower right")


def _point(pose: sk.Pose, index: int) -> tuple[float, float] | None:
    x, y, c = pose.points[index]
    return (x, y) if c > 0 else None


def _mid(*points):
    seen = [p for p in points if p is not None]
    if not seen:
        return None
    return (sum(p[0] for p in seen) / len(seen), sum(p[1] for p in seen) / len(seen))


def _angle(a, b) -> float:
    """Угол вектора a→b в градусах, ось y вверх (как на картинке смотрит человек)."""
    return math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0]))


def _direction(angle: float) -> str:
    return _DIRECTIONS[int(((angle % 360) + 22.5) // 45) % 8]


def _place(point, width: int, height: int) -> str:
    """Где в кадре точка: «top left», «center», «bottom»…"""
    column = ("left", "", "right")[min(int(3 * point[0] / width), 2)]
    row = ("top", "", "bottom")[min(int(3 * point[1] / height), 2)]
    place = " ".join(part for part in (row, column) if part)
    return place or "center"


def _limb(pose: sk.Pose, root: int, middle: int, end: int) -> tuple[str, bool] | None:
    """Куда тянется конечность и прямая ли она: (направление, прямая)."""
    a, b, c = _point(pose, root), _point(pose, middle), _point(pose, end)
    if a is None or c is None:
        return None
    reach = math.dist(a, c)
    length = (math.dist(a, b) + math.dist(b, c)) if b is not None else reach
    straight = length > 0 and reach / length > 0.9
    return _direction(_angle(a, c)), straight


def _side_of_picture(pose: sk.Pose, first: int, second: int) -> tuple[int, int]:
    """Какая из двух парных точек левее на картинке: (левая, правая)."""
    a, b = _point(pose, first), _point(pose, second)
    if a is None or b is None or a[0] <= b[0]:
        return first, second
    return second, first


def body_orientation(pose: sk.Pose) -> str | None:
    """Положение тела: ``upright``, ``leaning``, ``lying`` или ``upside down``."""
    neck = _point(pose, sk.NECK) or _mid(_point(pose, sk.R_SHOULDER), _point(pose, sk.L_SHOULDER))
    hips = _mid(_point(pose, sk.R_HIP), _point(pose, sk.L_HIP))
    if neck is None or hips is None:
        return None
    angle = _angle(hips, neck)
    if 60 <= angle <= 120:
        return "upright"
    if -120 <= angle <= -60:
        return "upside down"
    if -30 <= angle <= 30 or angle >= 150 or angle <= -150:
        return "lying"
    return "leaning"


def _on_all_fours(pose: sk.Pose, neck, hips) -> bool:
    """Туловище горизонтально, а руки и колени под ним — опора, а не «лежит».

    Кисти ниже плеч и колени ниже бёдер больше чем на треть длины туловища.
    """
    torso = math.dist(neck, hips)
    wrists = [p for p in (_point(pose, sk.R_WRIST), _point(pose, sk.L_WRIST)) if p]
    knees = [p for p in (_point(pose, sk.R_KNEE), _point(pose, sk.L_KNEE)) if p]
    if not wrists or not knees:
        return False
    return (all(w[1] > neck[1] + torso / 3 for w in wrists)
            and all(k[1] > hips[1] + torso / 3 for k in knees))


def describe(pose: sk.Pose) -> str:
    """Поза словами для промта; пустая строка, если точек мало."""
    neck = _point(pose, sk.NECK) or _mid(_point(pose, sk.R_SHOULDER), _point(pose, sk.L_SHOULDER))
    hips = _mid(_point(pose, sk.R_HIP), _point(pose, sk.L_HIP))
    head = _mid(*(_point(pose, i) for i in (sk.NOSE, sk.R_EYE, sk.L_EYE, sk.R_EAR, sk.L_EAR))) or neck
    if neck is None or hips is None or head is None:
        return ""
    orientation = body_orientation(pose)
    head_place = _place(head, pose.width, pose.height)
    torso = _direction(_angle(hips, neck))
    sentences = []
    if orientation == "upright":
        sentences.append(f"Her body is upright, head at the {head_place} of the picture.")
    elif orientation == "upside down":
        sentences.append(
            f"Her body is upside down: her head is at the {head_place} of the picture, "
            "below her hips, and her legs are above."
        )
    elif orientation == "lying" and _on_all_fours(pose, neck, hips):
        sentences.append(
            f"She is on all fours, supported on her hands and knees: her torso is horizontal, "
            f"her head is at the {head_place} of the picture."
        )
    elif orientation == "lying":
        sentences.append(
            f"She is lying horizontally: her head is at the {head_place} of the picture "
            f"and her torso points {torso} from the hips."
        )
    else:
        lower = " Her head is lower than her hips." if head[1] > hips[1] else ""
        sentences.append(
            f"Her torso is tilted, pointing {torso} from the hips, head at the {head_place} "
            f"of the picture.{lower}"
        )

    left, right = _side_of_picture(pose, sk.R_SHOULDER, sk.L_SHOULDER)
    arms = {sk.R_SHOULDER: (sk.R_SHOULDER, sk.R_ELBOW, sk.R_WRIST), sk.L_SHOULDER: (sk.L_SHOULDER, sk.L_ELBOW, sk.L_WRIST)}
    for side, shoulder in (("left", left), ("right", right)):
        limb = _limb(pose, *arms[shoulder])
        if limb:
            direction, straight = limb
            sentences.append(
                f"The arm on the {side} of the picture is {'straight' if straight else 'bent'}, "
                f"reaching {direction}."
            )

    left, right = _side_of_picture(pose, sk.R_HIP, sk.L_HIP)
    legs = {sk.R_HIP: (sk.R_HIP, sk.R_KNEE, sk.R_ANKLE), sk.L_HIP: (sk.L_HIP, sk.L_KNEE, sk.L_ANKLE)}
    for side, hip in (("left", left), ("right", right)):
        limb = _limb(pose, *legs[hip])
        if limb:
            direction, straight = limb
            sentences.append(
                f"The leg on the {side} of the picture is {'straight' if straight else 'bent'}, "
                f"going {direction} from the hip."
            )
    return " ".join(sentences)
