"""Весь вывод в консоль — по-английски: журнал, установка, командная строка, инструменты.

Требование заказчика (2026-09-27): проект открыт англоязычному сообществу, и
русская консоль его отпугивает. Интерфейс двуязычен и идёт через i18n — это
не консоль и здесь не проверяется. Комментарии и докстринги остаются
русскими (соглашение проекта), кроме тех, что сами уходят в консоль, —
описаний argparse.

Проверка — по синтаксическому дереву: строковые литералы с кириллицей в
аргументах вызовов, которые печатают (print, методы журнала, вопросы
установки ``out``/``ask``), в тексте исключений и в справке argparse. В
``tools/ui_check.py`` смотрится только текст отчёта (``report.check`` —
второй аргумент): первым идёт условие, и в нём законно встречаются русские
строки — поиск подписей русского интерфейса на странице.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
PRINTERS = {"print", "input", "out", "ask", "ask_secret"}


def _cyrillic(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    return [
        sub.value for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and CYRILLIC.search(sub.value)
    ]


def _console_texts(call: ast.Call) -> list[ast.AST]:
    """Аргументы вызова, которые окажутся в консоли."""
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    keywords = {kw.arg: kw.value for kw in call.keywords}
    if name in PRINTERS or (isinstance(func, ast.Attribute) and name in LOG_METHODS):
        return [*call.args, *keywords.values()]
    if name == "check":  # report.check(условие, сообщение)
        return call.args[1:]
    if name in ("add_argument", "ArgumentParser", "add_argument_group"):
        return [value for key, value in keywords.items() if key in ("help", "description", "epilog")]
    return []


def offenders(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        texts: list[str] = []
        if isinstance(node, ast.Call):
            for argument in _console_texts(node):
                texts += _cyrillic(argument)
            # description=__doc__… — русский докстринг ушёл бы в --help.
            for keyword in node.keywords:
                if keyword.arg == "description" and any(
                    isinstance(sub, ast.Name) and sub.id == "__doc__" for sub in ast.walk(keyword.value)
                ):
                    texts.append("description=__doc__ (module docstring in --help)")
        elif isinstance(node, ast.Raise):
            texts += _cyrillic(node.exc)
        found += [f"{path.relative_to(ROOT)}:{node.lineno}: {text[:60]!r}" for text in texts]
    return found


FILES = sorted(
    path for folder in ("fooocus_qwen", "tools") for path in (ROOT / folder).rglob("*.py")
)


@pytest.mark.parametrize("path", FILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_console_output_is_english(path):
    assert not offenders(path), "\n".join(offenders(path))


def test_the_check_sees_what_it_should():
    """Контроль: сама проверка ловит русский текст в каждом виде вывода."""
    sample = ROOT / "tmp" / "console_probe.py"
    sample.parent.mkdir(exist_ok=True)
    sample.write_text(
        "import logging\nLOGGER = logging.getLogger(__name__)\n"
        "print('готово')\nLOGGER.info('Качаю %s', 1)\nout('  Выбор')\n"
        "parser.add_argument('--x', help='справка')\nreport.check(ok, 'проверка')\n"
        "report.check('Сгенерировать' in text, 'fine')\n"
        "raise ValueError('ошибка')\n",
        encoding="utf-8",
    )
    try:
        texts = [line.split(": ", 1)[1] for line in offenders(sample)]
    finally:
        sample.unlink()
    # Условие с поиском подписи интерфейса ('Сгенерировать' in text) — не вывод.
    assert sorted(texts) == sorted(["'готово'", "'Качаю %s'", "'  Выбор'", "'справка'", "'проверка'", "'ошибка'"]), texts


def test_installer_scripts_are_english():
    """Установка и запуск — целиком, с комментариями."""
    for name in ("install.ps1", "install.sh", "run.ps1", "run.sh"):
        text = (ROOT / name).read_text(encoding="utf-8-sig")
        assert not CYRILLIC.search(text), f"{name}: {CYRILLIC.search(text)}"
