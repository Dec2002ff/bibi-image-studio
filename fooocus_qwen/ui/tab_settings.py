"""Вкладка настроек: адрес языковой модели, системные промты, память видеокарты.

Токен языковой модели в браузер не отправляется никогда. Раньше адрес
правился одним текстовым полем с содержимым файла целиком — вместе с
токеном открытым текстом, а интерфейс слушает ``0.0.0.0``: токен читал
любой в локальной сети, открывший вкладку. Теперь адрес — обычное поле,
токен — поле пароля, которое сервер не заполняет; пустое при сохранении
значит «оставить прежний», для удаления есть отдельная кнопка. Файл пишет
тот же ``llm.setup.render``, что и установка, — формат один на оба пути.

Системные промты правятся прямо здесь и перечитываются заново при каждом
обращении к языковой модели (см. ``prompting.boost``), а адрес — при каждом
``Studio.llm_client()``: оболочку не нужно перезапускать, чтобы подхватить
правку.

Отсутствие настроенной языковой модели — обычное состояние свежей установки,
а не ошибка: ``check_connection`` обязана сообщить об этом читаемым текстом,
а не уронить вкладку исключением.
"""

from __future__ import annotations

import gradio as gr

from .. import config
from .. import settings as settings_module
from ..engine import attention, hardware
from ..llm import LlmError, load_endpoint, parse_endpoint_file
from ..llm import setup as llm_setup
from . import layout
from .i18n import MESSAGES, Localizer, T, pick, say, sentences
from .state import GPU_CONCURRENCY_ID


def precision_choices(lang: str, vram_gib: float | None) -> list[tuple[str, str]]:
    """Варианты точности: что это, сколько видеопамяти, для каких карт.

    Подписи — из общего каталога (``settings.PRECISION_INFO``), того же, что
    у установки; подходящий к этой карте вариант помечен.
    """
    recommended = settings_module.recommended_precision(vram_gib)
    return [
        (settings_module.precision_label(name, lang, recommended=name == recommended), name)
        for name in settings_module.PRECISIONS
    ]


# VRAM profiles shown in the product languages.
_PROFILE_CARDS = {
    settings_module.MEMORY_HIGH: ("20GB 以上显存", "20GB 이상 VRAM"),
    settings_module.MEMORY_LOW: ("6–16GB 显存", "6–16GB VRAM"),
}
_PROFILE_HOW = {
    settings_module.MEMORY_HIGH: ("模型依次完整放入显卡", "모델을 번갈아 GPU에 전체 로드"),
    settings_module.MEMORY_LOW: (
        "主模型驻留显卡，编码器分层加载，缓存放在内存",
        "주 모델은 GPU에 유지하고 인코더는 레이어별 로드, 캐시는 RAM 사용",
    ),
}


def profile_choices(lang: str, vram_gib: float | None) -> list[tuple[str, str]]:
    """VRAM profiles with a resolved Auto label for the current GPU."""
    index = 0 if lang == "zh" else 1
    resolved_name = settings_module.resolve_profile(settings_module.MEMORY_AUTO, vram_gib)
    cards = _PROFILE_CARDS[resolved_name][index]
    auto = (
        f"自动 — 根据显卡选择（当前：{cards}）"
        if lang == "zh"
        else f"자동 — GPU에 맞게 선택 (현재: {cards})"
    )
    return [(auto, settings_module.MEMORY_AUTO)] + [
        (f"{_PROFILE_CARDS[name][index]} — {_PROFILE_HOW[name][index]}", name)
        for name in (settings_module.MEMORY_HIGH, settings_module.MEMORY_LOW)
    ]


_PROMPT_FILES: tuple[str, ...] = (
    "system_prompt_t2i.txt",
    "system_prompt_edit.txt",
    "system_prompt_describe.txt",
)

# Дисплей-имена файлов промтов — переводимый текст, а не служебные
# идентификаторы, поэтому у выпадающего списка ниже, как и у mask_mode во
# вкладке редактирования, переводятся и label, и choices.
_PROMPT_KEYS: dict[str, str] = {
    "system_prompt_t2i.txt": "sysprompt_t2i",
    "system_prompt_edit.txt": "sysprompt_edit",
    "system_prompt_describe.txt": "sysprompt_describe",
}


def _current_endpoint():
    """Настроенный адрес или ``None``, если настройки нет или она не читается."""
    try:
        return load_endpoint(config.ENDPOINT_FILE)
    except (OSError, ValueError):
        return None


def describe_endpoint(lang: str) -> str:
    """Что настроено сейчас — без токена: о нём говорится только, есть ли он."""
    endpoint = _current_endpoint()
    if endpoint is None:
        return say("endpoint_not_configured", lang)
    token = say("token_set" if endpoint.token else "token_unset", lang)
    return say("endpoint_current", lang, url=endpoint.base_url, token=token)


def describe_performance(studio, lang: str) -> str:
    """Что выбрано сейчас: точность, механизм внимания, готовность turbo."""
    chosen = settings_module.load()
    if chosen.sage_attention and attention.sage_available():
        attention_text = "SageAttention"  # имя собственное, не переводится
    elif chosen.sage_attention:
        attention_text = say("perf_attention_missing", lang)
    else:
        attention_text = say("perf_attention_native", lang)
    turbo = say("perf_turbo_ready" if studio.turbo_weights_present() else "perf_turbo_later", lang)
    return say(
        "perf_current", lang, precision=chosen.precision.upper(), attention=attention_text, turbo=turbo,
        profile=studio.memory_profile(),
    )


def _read_prompt(name: str, lang: str) -> str:
    try:
        return (config.SYSTEM_PROMPT_DIR / name).read_text(encoding="utf-8")
    except OSError:
        return say("prompt_file_missing", lang, name=name)


def build(studio, localizer: Localizer, language=None) -> dict:
    """Собирает вкладку.

    ``language`` — компонент с текущим языком; см. докстринг
    ``tab_generate.build``.
    """
    lang = studio.config.lang
    if language is None:
        language = gr.State(lang)

    with gr.Row(elem_classes=[layout.WORK_ROW]):
        with gr.Column(min_width=layout.SIDE_MIN_WIDTH, elem_classes=[layout.FORM_COL]):
            current = _current_endpoint()
            address = localizer.bind(
                gr.Textbox(
                    label=pick("llm_address", lang),
                    placeholder=pick("llm_address_placeholder", lang),
                    value=current.base_url if current else "",
                ),
                label=("语言模型服务器地址", "언어 모델 서버 주소"),
                placeholder=("192.168.1.10:8000 或 https://…", "192.168.1.10:8000 또는 https://…"),
            )
            # Поле пароля и всегда пустое: сервер токен не показывает — ни
            # здесь, ни где-либо ещё в интерфейсе.
            token = localizer.bind(
                gr.Textbox(
                    label=pick("llm_token", lang),
                    placeholder=pick("llm_token_placeholder", lang),
                    type="password",
                    value="",
                ),
                label=("Token", "토큰"),
                placeholder=("留空则保留当前 Token", "비워 두면 현재 토큰 유지"),
            )
            with gr.Row():
                save_endpoint = localizer.bind(
                    gr.Button(pick("save", lang)), value=("保存", "저장")
                )
                check = localizer.bind(
                    gr.Button(pick("llm_check", lang), variant="primary"),
                    value=("检查连接", "연결 확인"),
                )
            forget = localizer.bind(
                gr.Button(pick("llm_forget_token", lang), size="sm"),
                value=("移除 Token", "토큰 제거"),
            )
            endpoint_status = localizer.bind(
                gr.Textbox(
                    label=pick("status", lang), interactive=False, lines=2,
                    value=describe_endpoint(lang),
                ),
                label=("状态", "상태"),
            )

        # Системный промт — единственное место вкладки, где ширина идёт в
        # дело: это абзацы текста, который правят руками.
        with gr.Column(min_width=layout.SIDE_MIN_WIDTH, elem_classes=[layout.TEXT_COL]):
            zh_choices = [(pick(_PROMPT_KEYS[name], "zh"), name) for name in _PROMPT_FILES]
            ko_choices = [(pick(_PROMPT_KEYS[name], "ko"), name) for name in _PROMPT_FILES]
            chosen_file = localizer.bind(
                gr.Dropdown(
                    choices=zh_choices if lang == "zh" else ko_choices,
                    value=_PROMPT_FILES[0],
                    label=pick("sysprompt_select", lang),
                ),
                label=("系统提示词", "시스템 프롬프트"),
                choices=(zh_choices, ko_choices),
            )
            prompt_text = gr.Textbox(
                lines=18,
                value=_read_prompt(_PROMPT_FILES[0], lang),
                show_label=False,
                elem_classes=[layout.LONG_TEXT],
            )
            save_prompt_button = localizer.bind(
                gr.Button(pick("save_prompt", lang)), value=("保存提示词", "프롬프트 저장")
            )
            prompt_status = localizer.bind(
                gr.Textbox(label=pick("status", lang), interactive=False, lines=2),
                label=("状态", "상태"),
            )

        # Память стояла отдельной строкой под обеими колонками и занимала
        # её целиком ради двух строк текста. Третьей колонкой она и читается
        # лучше (рядом с тем, что тоже про состояние), и не отнимает высоту
        # у того, что под ней.
        with gr.Column(min_width=layout.SIDE_MIN_WIDTH, elem_classes=[layout.FORM_COL]):
            # Производительность: точность трансформера и механизм внимания.
            # Смена точности — это докачка весов и перезагрузка модели, поэтому
            # отдельная кнопка, а не мгновенная реакция на выбор; внимание
            # переключается на лету.
            chosen = settings_module.load()
            vram = hardware.vram_gib()
            # Столбцом, а не выпадающим списком: восемь вариантов различаются
            # ценой, и сравнивать их удобнее, когда видны все сразу.
            precision = localizer.bind(
                gr.Radio(
                    choices=precision_choices(lang, vram),
                    value=chosen.precision,
                    label=pick("perf_precision", lang),
                    info=pick("perf_precision_info", lang),
                    elem_classes=[layout.CHOICES],
                ),
                label=T["perf_precision"],
                info=T["perf_precision_info"],
                choices=(precision_choices("zh", vram), precision_choices("ko", vram)),
            )
            apply_precision_button = localizer.bind(
                gr.Button(pick("perf_apply", lang)), value=("应用精度", "정밀도 적용")
            )
            # Профиль памяти — раскладка весов (settings.MEMORY_PROFILES). Как
            # и точность, меняется перезагрузкой модели, поэтому — кнопкой.
            memory_profile = localizer.bind(
                gr.Radio(
                    choices=profile_choices(lang, vram),
                    value=chosen.memory_profile,
                    label=pick("perf_profile", lang),
                    elem_classes=[layout.CHOICES],
                ),
                label=T["perf_profile"],
                choices=(profile_choices("zh", vram), profile_choices("ko", vram)),
            )
            apply_profile_button = localizer.bind(
                gr.Button(pick("perf_profile_apply", lang)), value=("应用显存模式", "VRAM 프로필 적용")
            )
            sage = localizer.bind(
                gr.Checkbox(
                    label=pick("perf_sage", lang),
                    value=chosen.sage_attention,
                    # Без пакета галочка ничего бы не включила — и не обещает.
                    interactive=attention.sage_available(),
                ),
                label=("SageAttention — 加速注意力", "SageAttention — 빠른 어텐션"),
            )
            performance_status = localizer.bind(
                gr.Textbox(
                    label=pick("status", lang), interactive=False, lines=2,
                    value=describe_performance(studio, lang),
                ),
                label=("状态", "상태"),
            )

            memory = localizer.bind(
                gr.Textbox(
                    label=pick("memory", lang),
                    interactive=False,
                    lines=2,
                    value=say("model_not_loaded", lang),
                ),
                label=("显存", "GPU 메모리"),
                # Значение поля тоже переводимо: это заглушка «ещё не
                # загружена», а не результат измерения. Переключение языка
                # сбрасывает поле к ней — для снимка, который обновляют
                # кнопкой, это верное поведение, а вот русская строка вокруг
                # английских подписей — нет.
                value=MESSAGES["model_not_loaded"],
            )
            memory_refresh = localizer.bind(
                gr.Button(pick("refresh", lang)), value=("刷新", "새로고침")
            )

    def store_endpoint(address_text, token_text, lang):
        address_text = (address_text or "").strip()
        if not address_text:
            return "", say("endpoint_need_address", lang)
        current = _current_endpoint()
        # Пустое поле токена — «оставить прежний»: сервер токен не показывает,
        # и заставлять вводить его заново при каждой правке адреса нельзя.
        new_token = (token_text or "").strip() or (current.token if current else None)
        text = llm_setup.render(address_text, new_token, current.backend if current else "")
        try:
            parse_endpoint_file(text)
        except ValueError as error:
            return "", say("endpoint_bad_address", lang, error=error)
        config.ENDPOINT_FILE.write_text(text, encoding="utf-8")
        return "", sentences(say("endpoint_saved", lang), describe_endpoint(lang))

    def forget_token(lang):
        current = _current_endpoint()
        if current is None or not current.token:
            return describe_endpoint(lang)
        config.ENDPOINT_FILE.write_text(
            llm_setup.render(current.base_url, None, current.backend), encoding="utf-8"
        )
        return sentences(say("token_forgotten", lang), describe_endpoint(lang))

    def check_connection(lang):
        # load_endpoint бросает FileNotFoundError (файла нет — обычное дело
        # для свежей установки) или ValueError (в файле нет ни одного хоста),
        # ping() — LlmError (сервер недоступен или ответил ошибкой). Все три
        # случая — читаемое сообщение, а не падение вкладки.
        try:
            client = studio.llm_client()
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            return say("llm_no_connection", lang, error=error)
        return say(
            "llm_connected", lang, model=client.model or say("llm_no_model_named", lang)
        )

    def load_prompt_file(name, lang):
        if name not in _PROMPT_FILES:
            return say("prompt_file_missing", lang, name=name)
        return _read_prompt(name, lang)

    def store_prompt_file(name, text, lang):
        # Имя пришло из браузера и ложится в путь записи. Gradio и сам сверяет
        # значение списка с вариантами, но граница, на которой стоит запись
        # файла, не должна держаться на чужой проверке.
        if name not in _PROMPT_FILES:
            return say("prompt_file_missing", lang, name=name)
        (config.SYSTEM_PROMPT_DIR / name).write_text(text, encoding="utf-8")
        return say("prompt_file_saved", lang, name=name)

    def memory_report(lang):
        return studio.memory_report(lang)

    def apply_precision(choice, lang, progress=gr.Progress(track_tqdm=True)):
        if choice not in settings_module.PRECISIONS:
            return describe_performance(studio, lang)
        if choice == settings_module.load().precision:
            return sentences(say("precision_unchanged", lang), describe_performance(studio, lang))
        progress(0, desc=say("precision_downloading", lang))
        try:
            studio.ensure_precision_weights(choice)
        except Exception as error:  # noqa: BLE001 — сеть и диск: строка состояния, не падение
            return say("precision_download_failed", lang, error=error)
        studio.switch_precision(choice)
        if studio.config.preload:
            studio.preload_in_background()
        return sentences(say("precision_switched", lang, precision=choice.upper()), describe_performance(studio, lang))

    def apply_memory_profile(choice, lang):
        if choice not in settings_module.MEMORY_PROFILES:
            return describe_performance(studio, lang)
        if choice == settings_module.load().memory_profile:
            return sentences(say("profile_unchanged", lang), describe_performance(studio, lang))
        studio.switch_memory_profile(choice)
        if studio.config.preload:
            studio.preload_in_background()
        return sentences(say("profile_switched", lang, profile=choice), describe_performance(studio, lang))

    def toggle_sage(enabled, lang):
        studio.set_sage_attention(bool(enabled))
        return describe_performance(studio, lang)

    # Смена точности выгружает модель — в одной очереди с генерацией, чтобы не
    # вклиниться в чужую работу видеокарты (замок генератора её всё равно
    # дождётся, но очередь не выпустит генерацию навстречу перезагрузке).
    apply_precision_button.click(
        apply_precision, [precision, language], performance_status, concurrency_id=GPU_CONCURRENCY_ID
    )
    apply_profile_button.click(
        apply_memory_profile, [memory_profile, language], performance_status, concurrency_id=GPU_CONCURRENCY_ID
    )
    # input, а не change: change срабатывает и от программного обновления при
    # открытии вкладки, и каждое открытие «переключало» бы внимание.
    sage.input(toggle_sage, [sage, language], performance_status, concurrency_id=GPU_CONCURRENCY_ID)
    save_endpoint.click(store_endpoint, [address, token, language], [token, endpoint_status])
    forget.click(forget_token, language, endpoint_status)
    check.click(check_connection, language, endpoint_status)
    chosen_file.change(load_prompt_file, [chosen_file, language], prompt_text)
    save_prompt_button.click(
        store_prompt_file, [chosen_file, prompt_text, language], prompt_status
    )
    memory_refresh.click(memory_report, language, memory)

    def refresh(lang):
        # Вкладка перечитывает настройку при каждом открытии: иначе она
        # показывала бы состояние на момент запуска процесса — и отставала,
        # если адрес поменяли из другого окна или руками в файле.
        current = _current_endpoint()
        chosen = settings_module.load()
        return (
            (current.base_url if current else ""),
            describe_endpoint(lang),
            studio.memory_report(lang),
            chosen.precision,
            gr.update(value=chosen.sage_attention, interactive=attention.sage_available()),
            describe_performance(studio, lang),
            chosen.memory_profile,
        )

    return {
        "memory": memory,
        "endpoint_status": endpoint_status,
        "prompt_status": prompt_status,
        "address": address,
        "precision": precision,
        "sage": sage,
        "performance_status": performance_status,
        "memory_profile": memory_profile,
        "refresh": refresh,
    }
