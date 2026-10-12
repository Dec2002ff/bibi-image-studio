"""Bibi inspiration atlas: a lightweight visual prompt builder.

The first version intentionally ships a compact curated seed catalogue. The
catalogue is data-driven so it can grow into the full atlas without changing
the event graph or generation pipeline.
"""

from __future__ import annotations

import gradio as gr

from .i18n import Localizer, pick

CATALOG: dict[str, list[tuple[tuple[str, str], str]]] = {
    "roles": [
        (("成年女性", "성인 여성"), "adult woman"),
        (("成年男性", "성인 남성"), "adult man"),
        (("儿童", "어린이"), "child"),
        (("老人", "노인"), "elderly person"),
        (("吸血鬼", "뱀파이어"), "vampire"),
        (("兽人", "수인"), "beastfolk"),
        (("精灵", "엘프"), "elf"),
        (("人鱼", "인어"), "merfolk"),
        (("恶魔", "악마"), "demon"),
        (("机械生命", "기계 생명체"), "android"),
    ],
    "angles": [
        (("正面 0°", "정면 0°"), "front view"),
        (("前侧 45°", "45° 반측면"), "three-quarter view, 45 degrees"),
        (("左侧 90°", "좌측면 90°"), "left profile view, 90 degrees"),
        (("背面 180°", "후면 180°"), "back view"),
        (("右侧 270°", "우측면 270°"), "right profile view, 270 degrees"),
        (("俯视", "하이 앵글"), "high angle"),
        (("仰视", "로우 앵글"), "low angle"),
        (("鸟瞰", "버드아이 뷰"), "bird's-eye view"),
    ],
    "scenes": [
        (("古堡走廊", "고성 복도"), "gothic castle corridor"),
        (("卧室", "침실"), "bedroom"),
        (("咖啡馆", "카페"), "cafe interior"),
        (("城市街道", "도시 거리"), "city street"),
        (("森林", "숲"), "forest"),
        (("海边", "해변"), "seaside"),
        (("摄影棚", "촬영 스튜디오"), "photo studio"),
        (("未来都市", "미래 도시"), "futuristic city"),
    ],
    "styles": [
        (("日漫", "일본 애니풍"), "Japanese anime style"),
        (("国漫", "중국 애니풍"), "Chinese animation style"),
        (("韩漫", "한국 웹툰풍"), "Korean webtoon style"),
        (("2D", "2D"), "2D illustration"),
        (("2.5D", "2.5D"), "2.5D illustration"),
        (("3D", "3D"), "3D render"),
        (("写实摄影", "실사 사진"), "photorealistic photography"),
        (("电影感", "시네마틱"), "cinematic"),
    ],
    "plants": [
        (("玫瑰", "장미"), "roses"),
        (("牡丹", "모란"), "peonies"),
        (("梅花", "매화"), "plum blossoms"),
        (("紫藤", "등나무꽃"), "wisteria"),
        (("百合", "백합"), "lilies"),
        (("竹", "대나무"), "bamboo"),
        (("银杏", "은행나무"), "ginkgo"),
        (("幻想发光植物", "환상 발광 식물"), "fantasy bioluminescent plants"),
    ],
}


def _choices(category: str, lang: str) -> list[tuple[str, str]]:
    index = 0 if lang == "ru" else 1
    return [(labels[index], prompt) for labels, prompt in CATALOG[category]]


def _combine(*groups: list[str] | None) -> str:
    items: list[str] = []
    for group in groups:
        for item in group or []:
            if item and item not in items:
                items.append(item)
    return ", ".join(items)


def build(localizer: Localizer, language=None) -> dict:
    lang = localizer.default
    if language is None:
        language = gr.State(lang)

    gr.HTML(
        """
        <div class="bibi-atlas-intro">
            <div class="bibi-atlas-kicker">PROMPT ATLAS</div>
            <div class="bibi-atlas-title">灵感百科 <span>·</span> 영감 도감</div>
            <div class="bibi-atlas-desc">选择元素 → 自动组合提示词 → 继续编辑 → 发送到生成页</div>
        </div>
        """
    )

    with gr.Row(elem_classes=["bibi-atlas-grid"]):
        with gr.Column(elem_classes=["bibi-atlas-card"]):
            roles = localizer.bind(
                gr.CheckboxGroup(
                    choices=_choices("roles", lang),
                    label=pick("inspiration_roles", lang),
                ),
                label=("角色与生物", "캐릭터 · 생물"),
                choices=(_choices("roles", "ru"), _choices("roles", "en")),
            )
            angles = localizer.bind(
                gr.CheckboxGroup(
                    choices=_choices("angles", lang),
                    label=pick("inspiration_angles", lang),
                ),
                label=("视角与朝向", "시점 · 방향"),
                choices=(_choices("angles", "ru"), _choices("angles", "en")),
            )

        with gr.Column(elem_classes=["bibi-atlas-card"]):
            scenes = localizer.bind(
                gr.CheckboxGroup(
                    choices=_choices("scenes", lang),
                    label=pick("inspiration_scenes", lang),
                ),
                label=("场景", "장면"),
                choices=(_choices("scenes", "ru"), _choices("scenes", "en")),
            )
            plants = localizer.bind(
                gr.CheckboxGroup(
                    choices=_choices("plants", lang),
                    label=pick("inspiration_plants", lang),
                ),
                label=("植物", "식물"),
                choices=(_choices("plants", "ru"), _choices("plants", "en")),
            )

        with gr.Column(elem_classes=["bibi-atlas-card"]):
            styles = localizer.bind(
                gr.CheckboxGroup(
                    choices=_choices("styles", lang),
                    label=pick("inspiration_styles", lang),
                ),
                label=("风格与表现", "스타일 · 표현"),
                choices=(_choices("styles", "ru"), _choices("styles", "en")),
            )

            assembled = localizer.bind(
                gr.Textbox(
                    label=pick("inspiration_current", lang),
                    placeholder=pick("inspiration_current_placeholder", lang),
                    lines=7,
                    interactive=True,
                    elem_classes=["bibi-atlas-prompt"],
                ),
                label=("当前提示词组合", "현재 프롬프트 조합"),
                placeholder=("选择左侧灵感，也可以继续手动编辑。", "왼쪽에서 영감을 고른 뒤 직접 수정할 수 있어요."),
            )
            apply_button = localizer.bind(
                gr.Button(
                    pick("inspiration_apply", lang),
                    variant="primary",
                    elem_classes=["bibi-atlas-apply"],
                ),
                value=("✦ 加入生成", "✦ 생성에 적용"),
            )

    inputs = [roles, angles, scenes, plants, styles]
    for component in inputs:
        component.change(_combine, inputs, assembled, queue=False)

    return {
        "roles": roles,
        "angles": angles,
        "scenes": scenes,
        "plants": plants,
        "styles": styles,
        "prompt": assembled,
        "apply": apply_button,
    }
