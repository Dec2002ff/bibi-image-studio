"""Точка входа. По умолчанию поднимает веб-интерфейс, с --selftest проверяет окружение."""

from __future__ import annotations

import sys

from . import config, logging_setup


def selftest() -> int:
    """Проверяет всё, без чего оболочка не запустится, и печатает отчёт.

    Возвращает код возврата процесса: ноль, если готово к работе.
    """
    problems: list[str] = []

    try:
        import torch
    except ImportError as error:
        print(f"[no ] torch cannot be imported: {error}")
        return 1

    print(f"[ok ] torch {torch.__version__}")
    if not torch.cuda.is_available():
        problems.append("CUDA is not available: the model would run on CPU, which is unacceptably slow")
        print("[no ] CUDA is not available")
    else:
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[ok ] CUDA: {name}, {total:.1f} GiB")

    try:
        from diffusers import QwenImage21Pipeline  # noqa: F401
    except ImportError as error:
        problems.append("QwenImage21Pipeline is not available: diffusers from git is required")
        print(f"[no ] QwenImage21Pipeline cannot be imported: {error}")
    else:
        import diffusers

        print(f"[ok ] diffusers {diffusers.__version__}, QwenImage21Pipeline found")

    index = config.MODEL_DIR / "model_index.json"
    if index.is_file():
        print(f"[ok ] model weights: {config.MODEL_DIR}")
    else:
        problems.append(f"{index} not found")
        print(f"[no ] model weights not found: {index}")

    # Производительность: выбор из user/settings.json и то, что ему нужно.
    from . import settings
    from .engine import attention, fetch

    chosen = settings.load()
    if chosen.precision == settings.PRECISION_INT8:
        if fetch.missing_extra(config.INT8_DIR, (fetch.INT8_FILE,)):
            problems.append("INT8 precision is selected but its weights are missing: run --fetch-model")
            print(f"[no ] INT8 precision: missing {config.INT8_DIR / fetch.INT8_FILE}")
        else:
            print("[ok ] INT8 precision, weights found")
    else:
        print("[ok ] bf16 precision")
    if chosen.sage_attention:
        state = (
            "[ok ] SageAttention enabled" if attention.sage_available()
            else "[--] SageAttention selected but not installed: using default attention"
        )
        print(state)
    from .poses import detect

    if fetch.missing_extra(config.DWPOSE_DIR, detect.FILES):
        problems.append("pose detection weights (DWPose) are missing: run --fetch-model")
        print(f"[no ] pose detection: DWPose weights missing in {config.DWPOSE_DIR}")
    else:
        print("[ok ] pose detection: DWPose weights found")
    turbo_ready = not fetch.missing_extra(config.TURBO_DIR, fetch.TURBO_FILES)
    print("[ok ] Turbo: weights found" if turbo_ready else "[--] Turbo: weights will be downloaded when the preset is first selected")

    if problems:
        print("\nNot ready:")
        for item in problems:
            print(f"  - {item}")
        return 1

    print("\nEnvironment is ready.")
    return 0


def generate_once(args) -> int:
    """Одна генерация без интерфейса: для проверки и для скриптов.

    Через ``Studio`` — ту же дорогу, что у интерфейса: точность весов и
    SageAttention из настроек, адаптер Turbo (докачивается при первом
    выборе). Прежде модель грузилась здесь напрямую, и ``--preset Turbo``
    падал трассировкой: адаптер к такому генератору не подключался.
    """
    from pathlib import Path

    from .engine import presets
    from .engine.generator import GenerationRequest
    from .imaging import metadata
    from .storage import gallery
    from .ui.state import Studio

    studio = Studio(config.AppConfig(pin_memory=args.pin_memory, preset=args.preset, lang="en"))
    preset = presets.get(args.preset)
    failure = studio.weights_for(preset, "en")
    if failure:
        print(failure)
        return 1

    def show(index: int, step: int, total: int) -> None:
        print(f"\rimage {index + 1}: step {step}/{total}", end="", flush=True)

    results, failure = studio.run_generation(
        GenerationRequest(prompt=args.prompt, prompt_original=args.prompt, preset=preset), "en", progress=show
    )
    print()
    if failure:
        print(failure)
        return 1
    if not results:
        print("Nothing was generated")
        return 1

    destination = Path(args.out) if args.out else gallery.next_path(config.OUTPUT_DIR)
    metadata.save_png(results[0].image, destination, results[0].parameters)
    print(f"Saved: {destination}")
    print(f"Seed: {results[0].seed}, time: {results[0].parameters['seconds']} s")
    print(f"Memory: {studio.memory_report('en')}")
    return 0


def fetch_model() -> int:
    """Доводит веса до полного состава под выбранную точность. Зовётся установкой.

    При INT8 bf16-шарды трансформера (14 ГБ) не качаются: их место занимает
    INT8-трансформер Unsloth (7.3 ГБ). Вместе с моделью — веса распознавания
    позы (DWPose, 350 МБ, «Добавить позу»): без них первое распознавание
    ждало бы загрузки посреди работы.
    """
    from . import settings
    from .engine import fetch
    from .poses import detect

    int8 = settings.load().precision == settings.PRECISION_INT8
    try:
        downloaded = fetch.ensure_model(config.MODEL_DIR, include_transformer=not int8)
        if int8:
            downloaded = fetch.ensure_int8(config.INT8_DIR) or downloaded
        poses = fetch.ensure_files(config.DWPOSE_DIR, detect.REPO, detect.FILES)
    except fetch.ModelDownloadError as error:
        print(f"[no ] {error}")
        return 1
    except OSError as error:
        # Сеть, диск, права: причина человеку важнее типа исключения.
        print(f"[no ] failed to download weights: {error}")
        return 1

    if downloaded:
        print(f"[ok ] weights downloaded: {config.MODEL_DIR}")
    else:
        print(f"[ok ] weights found: {config.MODEL_DIR}")
    print(f"[ok ] pose detection weights {'downloaded' if poses else 'found'}: {config.DWPOSE_DIR}")
    return 0


def setup_performance() -> int:
    """Спрашивает точность весов и SageAttention. Отказ отвечать — не ошибка."""
    from .engine import setup

    setup.configure()
    return 0


def setup_llm() -> int:
    """Спрашивает адрес и токен языковой модели.

    Отказ отвечать и отсутствие консоли — не ошибки: AI-буст промтов
    необязателен, без него работает всё остальное. Уронить установку на
    последнем шаге, когда зависимости уже поставлены, было бы худшим из
    возможных исходов.
    """
    from .llm import setup

    if not sys.stdin.isatty():
        print("  Skipping language model setup: no console attached.")
        print(f"  The address can be set later in {config.ENDPOINT_FILE.name} or on the Settings tab.")
        return 0

    setup.configure(config.ENDPOINT_FILE)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = config.build_parser().parse_args(argv)
    logging_setup.setup_logging(args.verbose)
    config.ensure_directories()

    if args.fetch_model:
        return fetch_model()

    if args.setup_llm:
        return setup_llm()

    if args.setup_performance:
        return setup_performance()

    if args.selftest:
        return selftest()

    if args.prompt:
        return generate_once(args)

    # Модуль ui.app появляется в задаче 12; до неё запуск без --selftest и без
    # --prompt упадёт на этом импорте, и это правильное поведение: интерфейса ещё нет.
    from .ui.app import launch

    cfg = config.parse_args(argv)
    launch(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
