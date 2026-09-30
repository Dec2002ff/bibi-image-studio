"""Поза словами: положение тела, голова в кадре, руки и ноги по сторонам картинки."""

from __future__ import annotations

from fooocus_qwen.poses import describe
from fooocus_qwen.poses import skeleton as sk


def _pose(points: dict[int, tuple[float, float]]) -> sk.Pose:
    full = tuple((*points[i], 1.0) if i in points else (0.0, 0.0, 0.0) for i in range(sk.POINTS))
    return sk.Pose(full, 768, 768)


STANDING = {
    sk.NOSE: (384, 110), sk.NECK: (384, 180),
    sk.R_SHOULDER: (330, 180), sk.L_SHOULDER: (438, 180),
    sk.R_ELBOW: (320, 280), sk.L_ELBOW: (448, 280),
    sk.R_WRIST: (315, 370), sk.L_WRIST: (453, 370),
    sk.R_HIP: (350, 400), sk.L_HIP: (418, 400),
    sk.R_KNEE: (348, 540), sk.L_KNEE: (420, 540),
    sk.R_ANKLE: (346, 680), sk.L_ANKLE: (422, 680),
}


def _flipped(points):
    """Та же поза вверх ногами (поворот на 180° в кадре 768)."""
    return {i: (768 - x, 768 - y) for i, (x, y) in points.items()}


def test_standing_is_upright_with_arms_and_legs_down():
    pose = _pose(STANDING)
    assert describe.body_orientation(pose) == "upright"
    text = describe.describe(pose)
    assert text.startswith("Her body is upright, head at the top of the picture.")
    assert text.count("is straight, reaching down") == 2 and text.count("straight, going down") == 2


def test_an_upside_down_figure_is_named_so():
    """Тот случай, что модель путала: голова внизу, ноги сверху."""
    pose = _pose(_flipped(STANDING))
    assert describe.body_orientation(pose) == "upside down"
    text = describe.describe(pose)
    assert "upside down" in text and "head is at the bottom of the picture" in text
    assert "going up from the hip" in text


def test_lying_horizontally():
    lying = {i: (y, 768 - x) for i, (x, y) in STANDING.items()}  # поворот на 90°: голова слева
    pose = _pose(lying)
    assert describe.body_orientation(pose) == "lying"
    assert "lying horizontally" in describe.describe(pose) and "left" in describe.describe(pose)


def test_on_all_fours_is_not_lying():
    fours = {
        sk.NOSE: (200, 300), sk.NECK: (250, 330), sk.R_SHOULDER: (250, 330), sk.L_SHOULDER: (260, 335),
        sk.R_WRIST: (250, 520), sk.L_WRIST: (262, 520), sk.R_ELBOW: (250, 430), sk.L_ELBOW: (262, 430),
        sk.R_HIP: (500, 340), sk.L_HIP: (510, 345), sk.R_KNEE: (500, 520), sk.L_KNEE: (512, 520),
        sk.R_ANKLE: (620, 530), sk.L_ANKLE: (630, 530),
    }
    text = describe.describe(_pose(fours))
    assert "on all fours" in text and "lying" not in text


def test_a_raised_arm_reaches_up():
    raised = dict(STANDING)
    raised[sk.L_ELBOW], raised[sk.L_WRIST] = (450, 100), (455, 20)
    text = describe.describe(_pose(raised))
    assert "The arm on the right of the picture is straight, reaching up." in text


def test_sides_follow_the_picture_not_the_anatomy():
    """Фигура спиной: анатомически правая рука — справа на картинке; говорится по картинке."""
    back = dict(STANDING)
    back[sk.R_SHOULDER], back[sk.L_SHOULDER] = STANDING[sk.L_SHOULDER], STANDING[sk.R_SHOULDER]
    back[sk.R_ELBOW], back[sk.R_WRIST] = (448, 100), (453, 20)  # правая рука поднята, справа на картинке
    text = describe.describe(_pose(back))
    assert "The arm on the right of the picture is straight, reaching up." in text


def test_too_few_points_say_nothing():
    assert describe.describe(_pose({sk.NOSE: (10, 10)})) == ""
