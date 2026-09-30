"""Инструменты ячейки референса: окно выбора позы и окно эскиза.

На каждой ячейке сетки — две кнопки-значка. Первая открывает библиотеку поз
(плитки openposes.com и свои позы, в конце — плитка «Добавить позу»): выбор
кладёт в ячейку скелет позы. Вторая открывает холст для наброска: «Принять»
кладёт эскиз в ячейку, «Отмена» закрывает окно.

Окна — колонки поверх страницы (``layout.MODAL``), а не отдельные вкладки:
результат, промт и сетка остаются под ними, и после выбора человек сразу
видит ячейку, куда легла картинка. Своего модального окна в Gradio 6.5.1
нет, поэтому это обычная колонка, которую показывают и прячут, а место на
экране ей даёт CSS.

Имя позы задаётся при добавлении (необязательно), а у любой плитки — и
каталога, и своей — есть карандаш: он открывает правку имени и
перерисовку обложки. Внутрь ``gr.Gallery`` кнопку не вложить, поэтому
карандаш ставит скрипт (``POSE_EDIT_JS``), а его клик идёт той же дорогой,
что выбор плитки, — с пометкой «правка» в скрытом поле (``PICK_JS``).

«Добавить позу» — три шага подряд: окно показывает поле загрузки, по фото
DWPose (``poses.detect``) строит скелет, поза сохраняется в ``user/outputs/poses`` и
сразу ложится в ячейку, а затем Qwen-Image рисует для неё плитку в стиле
каталога (``poses.tile``). Плитка — последним шагом и в общей очереди GPU:
скелет нужен человеку сразу, а плитке не к спеху.
"""

from __future__ import annotations

import json
import logging

import gradio as gr
from PIL import Image

from .. import config
from ..engine.generator import MASK_ANNOTATION
from ..poses import detect, library, skeleton, tile
from . import layout, painter
from .i18n import Localizer, T, painter_labels, pick, say
from .painter import payload
from .references import MAX_REFERENCES, Grid, place
from .state import GPU_CONCURRENCY_ID

LOGGER = logging.getLogger(__name__)

ADD_POSE_TILE = config.RESOURCES_DIR / "poses" / "add_pose.png"

SKETCH_PAINTER_ID = "qs-sketch-painter"
# Холст эскиза: квадрат, как плитки поз, — рисовать на нём можно что угодно,
# а форма результата задаётся не референсом, а соотношением сторон вкладки.
SKETCH_SIDE = 1024
# Палитра наброска: карандаш и основные цвета. Белый — это «замазать»;
# ластик кисти стирает мазки до холста, то есть тоже до белого.
SKETCH_COLOURS: tuple[str, ...] = (
    "#000000", "#6b7280", "#ffffff", "#ef4444", "#f97316", "#facc15",
    "#22c55e", "#3b82f6", "#8b5cf6", "#92400e",
)

# Подсказки к значкам: у ``gr.Button`` нет своего ``title``, и его ставит
# скрипт — при загрузке страницы и при смене языка. Значки вкладки, которая
# ещё не открывалась, Gradio рисует позже загрузки, поэтому скрипт ещё и
# следит за появлением новых значков (наблюдатель ставится один раз и
# проверяет только новые узлы, так что стоит копейки).
#
# Язык скрипт читает с кнопки переключателя (на ней — текущий язык), а не из
# аргумента: язык приложения живёт в ``gr.State``, а состояние Gradio в
# JS-обработчик не передаётся — аргумент приходил пустым, и в английском
# интерфейсе подсказки оставались русскими (найдено, когда английский стал
# языком по умолчанию).
TOOL_TITLES = {
    layout.REF_POSE: ("Поза: выбрать из библиотеки или по фото", "Pose: pick from the library or from a photo"),
    layout.REF_SKETCH: ("Эскиз: нарисовать от руки", "Sketch: draw by hand"),
}
TITLES_JS = """
() => {
    const titles = __TITLES__;
    const button = document.querySelector('.qs-lang');
    window.__qsToolLang = button ? button.innerText.trim().toLowerCase() : 'en';
    const apply = root => {
        for (const [cls, pair] of Object.entries(titles)) {
            const nodes = root.classList && root.classList.contains(cls)
                ? [root] : (root.querySelectorAll ? root.querySelectorAll('.' + cls) : []);
            nodes.forEach(node => {
                node.title = pair[window.__qsToolLang === 'en' ? 1 : 0];
                node.setAttribute('aria-label', node.title);
            });
        }
    };
    apply(document);
    if (!window.__qsToolObserver) {
        window.__qsToolObserver = new MutationObserver(records => {
            for (const record of records) record.addedNodes.forEach(apply);
        });
        window.__qsToolObserver.observe(document.body, {childList: true, subtree: true});
    }
    return [];
}
""".replace("__TITLES__", json.dumps(TOOL_TITLES, ensure_ascii=False))


# Категории каталога openposes.com (начало имени файла) → перевод.
POSE_CATEGORIES = {
    "dance": "pose_cat_dance",
    "flexing": "pose_cat_flexing",
    "jumping": "pose_cat_jumping",
    "laying": "pose_cat_laying",
    "sitting": "pose_cat_sitting",
    "standing": "pose_cat_standing",
    "tpose": "pose_cat_tpose",
}


def display_title(entry: library.PoseEntry, lang: str, number: int = 0) -> str:
    """Имя позы в окне: заданное человеком или по умолчанию.

    По умолчанию у позы каталога — категория из имени файла и номер
    («standing_03» → «Стоя 3»), у своей — «Моя поза N» по её порядку.
    """
    if entry.title:
        return entry.title
    if entry.custom:
        return say("pose_default_title", lang, number=number or 1)
    category, _, index = entry.name.rpartition("_")
    key = POSE_CATEGORIES.get(category)
    if key is None or not index.isdigit():
        return entry.name
    return f"{pick(key, lang)} {int(index)}"


def titles(entries: list[library.PoseEntry], lang: str) -> list[str]:
    numbers = iter(range(1, len(entries) + 1))
    return [display_title(entry, lang, next(numbers) if entry.custom else 0) for entry in entries]


def pose_tiles(entries: list[library.PoseEntry], lang: str) -> list[tuple[str, str | None]]:
    """Значение галереи поз: плитки с именами и последней — «Добавить позу»."""
    tiles: list[tuple[str, str | None]] = [
        (str(entry.preview()), title) for entry, title in zip(entries, titles(entries, lang))
    ]
    tiles.append((str(ADD_POSE_TILE), pick("pose_add", lang)))
    return tiles


# Карандаш на плитках поз. Ставится наблюдателем за страницей: галерея
# перерисовывает плитки при каждом новом значении. Клик по карандашу не
# доходит до плитки, а поднимает пометку «правка» и сам нажимает плитку:
# выбор идёт штатной дорогой Gradio (select с номером плитки), а скрипт
# события (PICK_JS) подставляет пометку в скрытое поле. Последняя плитка —
# «Добавить позу», у неё карандаша нет.
POSE_EDIT_JS = """
() => {
    if (window.__qsPoseEdit) return [];
    const state = window.__qsPoseEdit = { next: false, frame: 0 };
    const TIP = __TIP__;
    const langIndex = () => {
        const button = document.querySelector('.qs-lang');
        return button && button.innerText.trim().toLowerCase() === 'ru' ? 0 : 1;
    };
    const decorate = () => {
        document.querySelectorAll('.__GRID__').forEach(grid => {
            const items = [...grid.querySelectorAll('.thumbnail-item')];
            items.forEach((item, index) => {
                let pen = item.querySelector(':scope > .__EDIT__');
                if (index === items.length - 1) { if (pen) pen.remove(); return; }
                if (!pen) {
                    item.style.position = 'relative';
                    pen = document.createElement('span');
                    pen.className = '__EDIT__';
                    pen.setAttribute('role', 'button');
                    pen.addEventListener('click', event => {
                        event.preventDefault();
                        event.stopPropagation();
                        state.next = true;
                        item.click();
                    }, true);
                    item.appendChild(pen);
                }
                const tip = TIP[langIndex()];
                if (pen.title !== tip) { pen.title = tip; pen.setAttribute('aria-label', tip); }
            });
        });
    };
    new MutationObserver(() => {
        cancelAnimationFrame(state.frame);
        state.frame = requestAnimationFrame(decorate);
    }).observe(document.body, { childList: true, subtree: true });
    decorate();
    return [];
}
""".replace("__TIP__", json.dumps(list(T["pose_edit_tip"]), ensure_ascii=False)).replace(
    "__GRID__", layout.POSE_GRID
).replace("__EDIT__", layout.POSE_EDIT)

# Скрипт выбора плитки: первым входом идёт скрытое поле режима — «правка»,
# если плитку нажал карандаш, иначе «выбор».
PICK_JS = """
(...args) => {
    const edit = Boolean(window.__qsPoseEdit && window.__qsPoseEdit.next);
    if (window.__qsPoseEdit) window.__qsPoseEdit.next = false;
    args[0] = edit ? 'edit' : 'pick';
    return args;
}
"""


def selected_index(event: gr.EventData) -> int | None:
    """Номер выбранной плитки. ``gr.SelectData`` падает на событии без
    значения (Gradio 6.5.1), поэтому индекс берётся из сырых данных."""
    data = getattr(event, "_data", None) or {}
    index = data.get("index")
    if isinstance(index, (list, tuple)):
        index = index[0] if index else None
    return index if isinstance(index, int) else None


def build(
    studio,
    localizer: Localizer,
    lang: str,
    language,
    grid: Grid,
    status,
    mode=None,
    sketch_id: str = SKETCH_PAINTER_ID,
) -> dict:
    """Собирает оба окна для сетки ``grid`` и связывает с её значками.

    ``status`` — строка состояния вкладки, ``mode`` — режим области вкладки
    правки (от него зависят теги, см. ``references``) или ``None`` на
    генерации. ``sketch_id`` — имя кисти эскиза на странице: окон эскиза
    два, по одному на вкладку, и скрипт кисти находит их по имени.
    ``lang`` — язык сборки, ``language`` — компонент с текущим языком.
    """
    references = grid.state
    reference_targets = grid.targets(status)
    if mode is None:
        mode = gr.State(None)
    target = gr.State(0)
    entries_state = gr.State([])

    # --- окно поз ---
    with gr.Column(visible=False, elem_classes=[layout.MODAL]) as pose_window:
        with gr.Column(elem_classes=[layout.MODAL_BOX]):
            pose_title = gr.Markdown(elem_classes=[layout.MODAL_TITLE])
            pose_grid = gr.Gallery(
                columns=8,
                allow_preview=False,
                object_fit="cover",
                show_label=False,
                interactive=False,
                buttons=[],
                elem_classes=[layout.POSE_GRID],
            )
            # Режим выбора плитки: «pick» — в ячейку, «edit» — правка (PICK_JS).
            pick_mode = gr.Textbox("pick", visible=False)
            with gr.Column(visible=False) as add_panel:
                pose_name = localizer.bind(
                    gr.Textbox(label=pick("pose_name", lang), placeholder=pick("pose_name_hint", lang), max_lines=1),
                    label=("Имя позы", "Pose name"),
                    placeholder=("необязательно — можно задать и потом", "optional — you can set it later too"),
                )
                pose_photo = localizer.bind(
                    gr.Image(
                        type="pil",
                        sources=["upload", "clipboard"],
                        label=pick("pose_photo", lang),
                        buttons=[],
                        elem_classes=[layout.POSE_PHOTO],
                    ),
                    label=("Фото с нужной позой", "A photo with the pose"),
                )
            # Правка позы: имя и перерисовка обложки.
            editing = gr.State("")
            with gr.Column(visible=False) as edit_panel:
                edit_heading = gr.Markdown(elem_classes=[layout.MODAL_TITLE])
                with gr.Row():
                    edit_cover = gr.Image(
                        interactive=False, show_label=False, buttons=[], elem_classes=[layout.POSE_COVER],
                    )
                    with gr.Column():
                        edit_name = localizer.bind(
                            gr.Textbox(label=pick("pose_name", lang), max_lines=1),
                            label=("Имя позы", "Pose name"),
                        )
                        with gr.Row():
                            save_name = localizer.bind(
                                gr.Button(pick("pose_save_name", lang), variant="primary"),
                                value=("Сохранить имя", "Save name"),
                            )
                            redraw = localizer.bind(
                                gr.Button(pick("pose_redraw", lang)),
                                value=("Перерисовать обложку", "Redraw cover"),
                            )
                        edit_back = localizer.bind(
                            gr.Button(pick("pose_back", lang), size="sm"), value=("К позам", "Back to poses")
                        )
            pose_message = gr.Markdown(elem_classes=[layout.MODAL_MESSAGE])
            with gr.Row():
                pose_close = localizer.bind(
                    gr.Button(pick("modal_close", lang), size="sm"), value=("Закрыть", "Close")
                )

    # --- окно эскиза ---
    with gr.Column(visible=False, elem_classes=[layout.MODAL]) as sketch_window:
        with gr.Column(elem_classes=[layout.MODAL_BOX, layout.SKETCH_BOX]):
            sketch_title = gr.Markdown(elem_classes=[layout.MODAL_TITLE])
            sketch = localizer.bind(
                painter.MaskPainter(
                    lang=lang,
                    region=MASK_ANNOTATION,
                    labels=painter_labels(),
                    palette=list(SKETCH_COLOURS),
                    # Эскизу — любой цвет и полупрозрачная кисть.
                    free_colour=True,
                    elem_id=sketch_id,
                    elem_classes=[layout.SKETCH],
                ),
                lang=("ru", "en"),
            )
            sketch_message = gr.Markdown(elem_classes=[layout.MODAL_MESSAGE])
            with gr.Row():
                sketch_cancel = localizer.bind(
                    gr.Button(pick("modal_cancel", lang)), value=("Отмена", "Cancel")
                )
                sketch_accept = localizer.bind(
                    gr.Button(pick("modal_accept", lang), variant="primary"), value=("Принять", "Accept")
                )

    # --- обработчики: позы ---

    def pose_opener(index: int):
        def open_pose_window(lang):
            return (
                index,
                gr.Column(visible=True),
                gr.Column(visible=False),
                say("pose_title", lang, cell=index + 1),
                "",
                gr.Column(visible=False),
            )

        return open_pose_window

    def load_tiles(lang):
        """Плитки в окно: каталог из поставки, затем свои позы."""
        entries = library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir())
        return [entry.name for entry in entries], pose_tiles(entries, lang), ""

    def pick_pose(action, index, names, current, mode_value, lang, event: gr.EventData):
        """Выбор плитки: поза — в ячейку и окно закрыть; «Добавить позу» — поле
        загрузки; карандаш (``action == "edit"``) — правка имени и обложки.

        Выходы: запись в сетку, окно, поле загрузки, сообщение, затем панель
        правки (панель, заголовок, обложка, имя, какая поза) и сама галерея —
        её выбор сбрасывается, чтобы плитку можно было нажать снова.
        """
        chosen = selected_index(event)
        keep = (gr.update(),) * (1 + 2 * MAX_REFERENCES + 1)
        no_edit = (gr.Column(visible=False), gr.update(), gr.update(), gr.update(), gr.update())
        reset = gr.Gallery(selected_index=None)
        if chosen is None:
            return (*keep, gr.update(), gr.update(), "", *no_edit, reset)
        if chosen >= len(names):
            return (*keep, gr.update(), gr.Column(visible=True), say("pose_add_hint", lang), *no_edit, reset)
        entry = _entry(names[chosen])
        if entry is None:
            return (*keep, gr.update(), gr.update(), say("pose_missing", lang), *no_edit, reset)
        if action == "edit":
            title = _title_of(entry, lang)
            return (
                *keep, gr.update(), gr.Column(visible=False), "",
                gr.Column(visible=True), say("pose_edit_title", lang, title=title),
                str(entry.preview()), entry.title or title, entry.name, reset,
            )
        with Image.open(entry.skeleton) as opened:
            image = opened.convert("RGB")
        placed = place(current, index, image, lang, say("pose_placed", lang, cell=index + 1), mode_value)
        return (*placed, gr.Column(visible=False), gr.Column(visible=False), "", *no_edit, reset)

    def _title_of(entry: library.PoseEntry, lang: str) -> str:
        entries = library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir())
        for item, title in zip(entries, titles(entries, lang)):
            if item.name == entry.name:
                return title
        return entry.name

    def save_pose_name(name, title_text, lang):
        """Имя позы: заданное — сохранить, пустое — вернуть имя по умолчанию."""
        entry = _entry(name) if name else None
        if entry is None:
            return gr.update(), gr.update(), gr.update(), say("pose_missing", lang)
        written = library.set_title(config.user_pose_dir(), entry.name, title_text or "")
        entries = library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir())
        title = _title_of(entry, lang)
        message = say("pose_name_saved", lang, title=written) if written else say("pose_name_reset", lang)
        return (
            [item.name for item in entries], pose_tiles(entries, lang),
            say("pose_edit_title", lang, title=title), message,
        )

    def back_to_poses():
        return gr.Column(visible=False), ""

    def add_pose(photo, title_text, index, current, mode_value, lang):
        """Фото → скелет → своя поза (с именем, если задано) → в ячейку.
        Плитку рисует следующий шаг."""
        keep = (gr.update(),) * (1 + 2 * MAX_REFERENCES + 1)
        if photo is None:
            return (*keep, gr.update(), gr.update(), gr.update(), "", None)
        try:
            found = detect.to_pose(studio.pose_detector().detect(photo))
        except detect.NoPersonFound:
            return (*keep, gr.update(), gr.update(), gr.update(), say("pose_not_found", lang), None)
        except Exception as error:  # noqa: BLE001 — веса, сеть, onnxruntime: строка в окне
            LOGGER.exception("Pose not recognized")
            return (*keep, gr.update(), gr.update(), gr.update(), say("pose_failed", lang, error=error), None)

        user = config.user_pose_dir()
        # Имя не задано — «Моя поза N» на языке интерфейса, чтобы у позы было
        # имя, по которому её узнать; поменять его можно карандашом.
        own = sum(1 for item in library.list_poses(config.POSE_LIBRARY_DIR, user) if item.custom)
        title = (title_text or "").strip() or say("pose_default_title", lang, number=own + 1)
        entry = library.add_custom(user, found, title)
        with Image.open(entry.skeleton) as opened:
            image = opened.convert("RGB")
        placed = place(current, index, image, lang, say("pose_placed", lang, cell=index + 1), mode_value)
        entries = library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir())
        return (
            *placed,
            [item.name for item in entries],
            pose_tiles(entries, lang),
            gr.Column(visible=False),
            say("pose_added", lang),
            entry.name,
        )

    def draw_tile(name, lang, progress=gr.Progress()):
        """Плитка новой позы — Qwen-Image по скелету, в стиле каталога."""
        return _draw(name, lang, progress, "pose_tile_done")

    def redraw_cover(name, lang, progress=gr.Progress()):
        """«Перерисовать обложку» в правке позы — тот же рисунок, другое сообщение."""
        return _draw(name, lang, progress, "pose_cover_done")

    def _draw(name, lang, progress, done_key: str):
        """Рисует обложку и отдаёт: имена поз, плитки, сообщение, новую обложку."""
        entry = _entry(name) if name else None
        if entry is None:
            return gr.update(), gr.update(), gr.update(), gr.update()
        progress(0, desc=say("pose_tile_drawing", lang))
        with Image.open(entry.skeleton) as opened:
            bones = opened.convert("RGB")
        try:
            pose = skeleton.load(entry.keypoints)
        except (OSError, ValueError):
            pose = None
        # Поза словами в промте и несколько кандидатов (см. poses/tile.py).
        request = tile.request(bones, turbo_ready=studio.turbo_weights_present(), pose=pose)

        def report(index: int, step: int, total: int) -> None:
            # Кандидаты рисуются подряд: общая полоса на все, а не по кругу на каждого.
            progress((index * total + step, total * request.image_number), desc=say("pose_tile_drawing", lang))

        produced, failure = studio.run_generation(request, lang, progress=report)
        if failure is not None or not produced:
            return gr.update(), gr.update(), say("pose_tile_failed", lang, error=failure or "—"), gr.update()
        chosen = produced[0].image
        if pose is not None and len(produced) > 1:
            try:
                chosen = tile.best([item.image for item in produced], pose, studio.pose_detector())
            except Exception:  # noqa: BLE001 — без распознавания остаётся первый кандидат
                LOGGER.exception("Could not score the cover candidates; keeping the first one")
        library.set_tile(entry, chosen)
        entries = library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir())
        fresh = _entry(entry.name)
        return (
            [item.name for item in entries], pose_tiles(entries, lang), say(done_key, lang),
            str(fresh.preview()) if fresh else gr.update(),
        )

    def _entry(name: str) -> library.PoseEntry | None:
        for entry in library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir()):
            if entry.name == name:
                return entry
        return None

    # --- обработчики: эскиз ---

    def sketch_opener(index: int):
        def open_sketch_window(lang):
            blank = Image.new("RGB", (SKETCH_SIDE, SKETCH_SIDE), "white")
            return (
                index,
                gr.Column(visible=True),
                say("sketch_title", lang, cell=index + 1),
                payload.encode(blank),
                "",
            )

        return open_sketch_window

    def accept_sketch(value, index, current, mode_value, lang):
        keep = (gr.update(),) * (1 + 2 * MAX_REFERENCES + 1)
        try:
            canvas = payload.decode(value)
        except payload.PayloadError as error:
            return (*keep, gr.update(), say("sketch_failed", lang, error=error))
        if canvas.background is None:
            return (*keep, gr.update(), say("sketch_empty", lang))
        image = canvas.background.convert("RGBA")
        if canvas.layer is not None:
            image = Image.alpha_composite(image, canvas.layer.convert("RGBA"))
        placed = place(
            current, index, image.convert("RGB"), lang, say("sketch_placed", lang, cell=index + 1), mode_value
        )
        return (*placed, gr.Column(visible=False), "")

    def close():
        return gr.Column(visible=False)

    # --- связи ---
    pose_outputs = [target, pose_window, add_panel, pose_title, pose_message, edit_panel]
    for index, button in enumerate(grid.pose_buttons):
        button.click(
            pose_opener(index), language, pose_outputs, queue=False, show_progress="hidden",
        ).then(
            load_tiles, language, [entries_state, pose_grid, pose_message], show_progress="hidden",
        )
    pose_grid.select(
        pick_pose,
        [pick_mode, target, entries_state, references, mode, language],
        [*reference_targets, pose_window, add_panel, pose_message,
         edit_panel, edit_heading, edit_cover, edit_name, editing, pose_grid],
        js=PICK_JS,
        show_progress="hidden",
    )
    new_pose = gr.State(None)
    pose_photo.upload(
        add_pose,
        [pose_photo, pose_name, target, references, mode, language],
        [*reference_targets, entries_state, pose_grid, add_panel, pose_message, new_pose],
        show_progress="minimal",
    ).then(
        draw_tile,
        [new_pose, language],
        [entries_state, pose_grid, pose_message, edit_cover],
        concurrency_id=GPU_CONCURRENCY_ID,
    )
    save_name.click(
        save_pose_name, [editing, edit_name, language],
        [entries_state, pose_grid, edit_heading, pose_message], show_progress="hidden",
    )
    redraw.click(
        redraw_cover, [editing, language], [entries_state, pose_grid, pose_message, edit_cover],
        concurrency_id=GPU_CONCURRENCY_ID,
    )
    edit_back.click(back_to_poses, None, [edit_panel, pose_message], queue=False)
    pose_close.click(close, None, pose_window, queue=False)

    sketch_outputs = [target, sketch_window, sketch_title, sketch, sketch_message]
    for index, button in enumerate(grid.sketch_buttons):
        button.click(sketch_opener(index), language, sketch_outputs, show_progress="hidden")
    sketch_accept.click(
        accept_sketch,
        [sketch, target, references, mode, language],
        [*reference_targets, sketch_window, sketch_message],
        js=painter.flush_js(sketch_id),
        show_progress="hidden",
    )
    sketch_cancel.click(close, None, sketch_window, queue=False)

    return {
        "pose_window": pose_window,
        "pose_grid": pose_grid,
        "pose_photo": pose_photo,
        "pose_name": pose_name,
        "edit_panel": edit_panel,
        "edit_name": edit_name,
        "sketch_window": sketch_window,
        "sketch": sketch,
    }

