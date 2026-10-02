"""Подписи выбора качества и точности: понятными словами, из тех же чисел, что считаются."""

from __future__ import annotations

from fooocus_qwen import settings as settings_module
from fooocus_qwen.engine import presets
from fooocus_qwen.ui import quality, tab_settings


def test_every_preset_is_offered_once_from_fast_to_best():
    assert sorted(quality.ORDER) == sorted(presets.NAMES)
    full = [name for name in quality.ORDER if not presets.get(name).turbo and not presets.get(name).transformer]
    assert full == ["LowQuality", "MiddleQuality", "MaxQuality"], "полная модель — по возрастанию, после дистиллятов"
    assert [value for _label, value in quality.choices("en")] == list(quality.ORDER)


def test_labels_say_steps_and_size_taken_from_the_preset():
    for name in presets.NAMES:
        preset = presets.get(name)
        for lang in ("ru", "en"):
            text = quality.label(name, lang)
            assert str(preset.num_inference_steps) in text and f"{preset.output_resolution} px" in text
    assert quality.label("MiddleQuality", "ru") == "Среднее — 28 шагов, 1536 px"
    assert quality.label("Turbo4", "ru").startswith("Turbo4 — 4 шага, 1024 px;"), "4 — «шага», не «шагов»"


def test_every_precision_has_a_label_in_the_list_order():
    assert list(settings_module.PRECISION_INFO) == list(settings_module.PRECISIONS)
    label = settings_module.precision_label("Q4_K_M", "ru", recommended=True)
    assert "3.9 ГиБ" in label and "6–8 ГБ" in label and label.endswith("под вашу карту")


def test_the_settings_tab_marks_what_fits_the_card():
    marked = [value for text, value in tab_settings.precision_choices("en", 8.0) if "fits your card" in text]
    assert marked == [settings_module.recommended_precision(8.0)]
    auto = tab_settings.profile_choices("ru", 8.0)[0][0]
    assert auto == "Авто — по видеокарте (сейчас: для карт 6–16 ГБ)"
