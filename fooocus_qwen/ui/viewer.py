"""Просмотр изображения в полном размере поверх страницы.

Открывается кликом по большой картинке результата (генерация, правка) и по
картинке в заполненной ячейке референса; на вкладке галереи — двойным
кликом по миниатюре: одиночный там уже открывает карточку параметров.

Без него клик вёл не туда. Большая картинка галереи Gradio по клику листает
к следующему кадру, а картинка в ячейке референса лежит в поле загрузки и
открывала выбор файла. Поэтому клик перехватывается на уровне документа в
фазе погружения — раньше, чем до картинки доберутся обработчики Gradio, — и
дальше не идёт.

Всё на клиенте: картинки уже на странице, и ссылка у них — на полный файл
(у галереи — исходный PNG результата), так что серверу делать нечего.

Просмотр: «вписать в окно» и «100 %» — по клику на картинку (в 100 % точка
под курсором остаётся на месте), перетаскивание сдвигает крупную картинку,
← и → листают остальные картинки того же поля, Esc, клик по фону или
крестик закрывают. Подписи — на языке интерфейса: язык берётся с кнопки
переключателя, потому что ``gr.State`` в скрипт не передаётся.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .i18n import T

HERE = Path(__file__).parent
TEXT_KEYS = ("viewer_hint", "modal_close")

_SCRIPT = r"""
() => {
    if (window.__qsViewer) return [];
    const CSS = __CSS__;
    const TEXT = __TEXT__;
    const langIndex = () => {
        const button = document.querySelector('.qs-lang');
        return button && button.innerText.trim().toLowerCase() === 'ru' ? 0 : 1;
    };
    const say = key => TEXT[key][langIndex()];

    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);

    const root = document.createElement('div');
    root.className = 'qv-root';
    root.hidden = true;
    root.dataset.mode = 'fit';
    root.setAttribute('role', 'dialog');
    root.setAttribute('aria-modal', 'true');
    root.innerHTML = '<img class="qv-image" alt=""><div class="qv-bar"></div>'
        + '<button class="qv-close" type="button">×</button>';
    document.body.appendChild(root);
    const image = root.querySelector('.qv-image');
    const bar = root.querySelector('.qv-bar');
    const closeButton = root.querySelector('.qv-close');

    const state = { list: [], index: 0, mode: 'fit', drag: null, moved: false, overflow: '' };

    function caption() {
        const size = image.naturalWidth ? `${image.naturalWidth}×${image.naturalHeight}` : '';
        const place = state.list.length > 1 ? `${state.index + 1}/${state.list.length}` : '';
        const zoom = state.mode === 'actual' ? '100%' : '';
        bar.textContent = [size, zoom, place, say('viewer_hint')].filter(Boolean).join(' · ');
        closeButton.title = say('modal_close');
        closeButton.setAttribute('aria-label', say('modal_close'));
    }

    function setMode(mode, pointer) {
        const before = image.getBoundingClientRect();
        state.mode = mode;
        root.dataset.mode = mode;
        if (mode === 'actual' && pointer && before.width) {
            // Точка картинки под курсором остаётся под курсором.
            const fx = (pointer.x - before.left) / before.width;
            const fy = (pointer.y - before.top) / before.height;
            const after = image.getBoundingClientRect();
            root.scrollLeft += after.left + fx * after.width - pointer.x;
            root.scrollTop += after.top + fy * after.height - pointer.y;
        }
        caption();
    }

    function show(index) {
        state.index = (index + state.list.length) % state.list.length;
        image.src = state.list[state.index];
        setMode('fit');
    }

    function open(list, index) {
        state.list = list.length ? list : [];
        if (!state.list.length) return;
        state.overflow = document.documentElement.style.overflow;
        document.documentElement.style.overflow = 'hidden';
        root.hidden = false;
        show(Math.max(0, index));
        closeButton.focus({ preventScroll: true });
    }

    function close() {
        if (root.hidden) return;
        root.hidden = true;
        image.removeAttribute('src');
        document.documentElement.style.overflow = state.overflow;
    }

    image.addEventListener('load', caption);
    image.addEventListener('click', event => {
        if (state.moved) { state.moved = false; return; }
        setMode(state.mode === 'fit' ? 'actual' : 'fit', { x: event.clientX, y: event.clientY });
    });
    image.addEventListener('pointerdown', event => {
        if (state.mode !== 'actual' || event.button !== 0) return;
        state.drag = { x: event.clientX, y: event.clientY, left: root.scrollLeft, top: root.scrollTop };
        state.moved = false;
        image.setPointerCapture(event.pointerId);
        event.preventDefault();
    });
    image.addEventListener('pointermove', event => {
        if (!state.drag) return;
        const dx = event.clientX - state.drag.x, dy = event.clientY - state.drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 4) { state.moved = true; root.dataset.dragging = 'true'; }
        root.scrollLeft = state.drag.left - dx;
        root.scrollTop = state.drag.top - dy;
    });
    const endDrag = () => { state.drag = null; delete root.dataset.dragging; };
    image.addEventListener('pointerup', endDrag);
    image.addEventListener('pointercancel', endDrag);
    root.addEventListener('click', event => { if (event.target === root) close(); });
    closeButton.addEventListener('click', close);
    document.addEventListener('keydown', event => {
        if (root.hidden) return;
        if (event.key === 'Escape') close();
        else if (event.key === 'ArrowRight' && state.list.length > 1) show(state.index + 1);
        else if (event.key === 'ArrowLeft' && state.list.length > 1) show(state.index - 1);
        else return;
        event.preventDefault();
        event.stopPropagation();
    }, true);

    // Без повторов: у Gradio на каждый кадр по две миниатюры (лента и сетка).
    const sources = (scope, selector) => [...new Set([...scope.querySelectorAll(selector)]
        .map(node => node.currentSrc || node.src).filter(Boolean))];

    function intercept(event, list, src) {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        const index = list.indexOf(src);
        open(index >= 0 ? list : [src], Math.max(0, index));
    }

    document.addEventListener('click', event => {
        const img = event.target && event.target.closest ? event.target.closest('img') : null;
        if (!img || root.contains(img) || event.button !== 0) return;
        const src = img.currentSrc || img.src;
        if (!src) return;
        // Ячейка референса: картинка лежит в поле загрузки — клик по ней
        // открыл бы выбор файла, а нужен просмотр.
        if (img.closest('.qs-refslot')) { intercept(event, [src], src); return; }
        // Большая картинка результата: Gradio по клику листал бы дальше.
        const board = img.closest('.qs-board');
        if (board && img.closest('.preview') && img.closest('.media-button')) {
            const list = sources(board, '.thumbnail-item img');
            intercept(event, list.length ? list : [src], src);
        }
    }, true);

    // Галерея: одиночный клик открывает карточку параметров, просмотр — двойной.
    document.addEventListener('dblclick', event => {
        const img = event.target && event.target.closest ? event.target.closest('.qs-browse img') : null;
        if (!img) return;
        const browse = img.closest('.qs-browse');
        intercept(event, sources(browse, '.thumbnail-item img'), img.currentSrc || img.src);
    }, true);

    window.__qsViewer = {
        open, close,
        state: () => ({ open: !root.hidden, mode: state.mode, index: state.index, count: state.list.length,
                        src: image.currentSrc || image.src || '', natural: [image.naturalWidth, image.naturalHeight] }),
    };
    return [];
}
"""


@lru_cache(maxsize=1)
def script() -> str:
    """Скрипт просмотра со встроенными стилями и подписями — один раз на процесс."""
    css = (HERE / "viewer.css").read_text(encoding="utf-8")
    text = {key: list(T[key]) for key in TEXT_KEYS}
    return (
        _SCRIPT.replace("__CSS__", json.dumps(css))
        .replace("__TEXT__", json.dumps(text, ensure_ascii=False))
    )
