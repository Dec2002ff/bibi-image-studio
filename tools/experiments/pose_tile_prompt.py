r"""Опыт: промт обложки позы — чем помочь модели на трудных позах.

Повод (2026-09-30): обложки своих поз «Nu01»…«Nu08» местами не держат позу —
перевёрнутую фигуру (голова внизу) модель рисует головой вверх, четвереньки
путает. Промт плитки говорил только «поза как у скелета на референсе».
Сравниваются варианты:

  ``current``  — промт как был (``tile.PROMPT`` без описания);
  ``describe`` — плюс поза словами из точек скелета (``poses.describe``):
                 положение тела, где голова, куда тянутся руки и ноги;
  ``legend``   — ``describe`` плюс пояснение цветового кода скелета.

И отдельно — «лучший из двух»: из двух сидов берётся тот, чья поза ближе к
скелету (выбор кандидата по DWPose при генерации обложки).

Мера — ошибка позы по DWPose на результате (как в ``pose_tile.py``): средняя
по точкам после приведения к общей рамке, % её размера; «провал» — больше
10 %. Картинки — в ``tmp/pose_tile_prompt/``.

Запуск (нужны веса модели и DWPose):
    .venv\Scripts\python tools\experiments\pose_tile_prompt.py --seeds 1 2
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "experiments"))

from PIL import Image  # noqa: E402
from pose_tile import pose_error  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import presets  # noqa: E402
from fooocus_qwen.engine.generator import GenerationRequest  # noqa: E402
from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402
from fooocus_qwen.poses import describe, detect, library, skeleton, tile  # noqa: E402

WORK = ROOT / "tmp" / "pose_tile_prompt"
CATALOG_HARD = ["laying_03", "jumping_03"]
OWN = ["Nu01", "Nu02", "Nu03", "Nu04", "Nu05", "Nu06", "Nu07", "Nu08", "Levitation", "Spell cast", "Scene"]

LEGEND = (
    " How to read the skeleton: the magenta and purple dots are her face (eyes, ears, nose), the dark "
    "blue line is her neck, the red and orange lines are her shoulders, the yellow and lime lines "
    "are her arms, the green and cyan-blue lines are her legs."
)


def poses() -> list[tuple[str, skeleton.Pose, Path]]:
    user = config.user_pose_dir()
    entries = library.list_poses(config.POSE_LIBRARY_DIR, user)
    by_title = {entry.title: entry for entry in entries if entry.custom and entry.title}
    chosen = [(name, by_title[name]) for name in OWN if name in by_title]
    chosen += [(entry.name, entry) for entry in entries if entry.name in CATALOG_HARD]
    return [(label, skeleton.load(entry.keypoints), entry.skeleton) for label, entry in chosen]


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(description="Pose cover prompt: help the model on hard poses")
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
    args = parser.parse_args()

    from fooocus_qwen.ui.state import Studio

    WORK.mkdir(parents=True, exist_ok=True)
    studio = Studio(config.AppConfig(lang="en"))
    preset = presets.get("Turbo" if studio.turbo_weights_present() else "LowQuality")
    detector = detect.PoseDetector(config.DWPOSE_DIR)
    variants = {
        "current": lambda pose: tile.PROMPT,
        "describe": lambda pose: tile.PROMPT + " " + describe.describe(pose),
        "legend": lambda pose: tile.PROMPT + " " + describe.describe(pose) + LEGEND,
    }
    errors: dict[str, list[list[float]]] = {name: [] for name in variants}
    for label, pose, bones_path in poses():
        bones = Image.open(bones_path).convert("RGB")
        for variant, prompt_of in variants.items():
            prompt = prompt_of(pose)
            row = []
            for seed in args.seeds:
                started = time.time()
                produced, failure = studio.run_generation(GenerationRequest(
                    prompt=prompt, prompt_original=prompt, preset=preset, references=(bones,),
                    aspect="1:1", seed=seed,
                ), "en")
                if failure:
                    print(failure)
                    return 1
                image = produced[0].image
                image.save(WORK / f"{label}_{variant}_{seed}.png")
                try:
                    error, _mirrored = pose_error(pose, detect.to_pose(detector.detect(image)))
                except detect.NoPersonFound:
                    error = 100.0
                row.append(error)
                print(f"{label:11} {variant:9} seed {seed}: pose error {error:5.1f} %, "
                      f"{time.time() - started:.0f} s", flush=True)
            errors[variant].append(row)

    print("\nvariant    | median | mean | fails >10% | best-of-seeds median | best-of fails")
    for variant, rows in errors.items():
        flat = [e for row in rows for e in row]
        best = [min(row) for row in rows]
        print(f"{variant:10} | {statistics.median(flat):6.1f} | {statistics.mean(flat):4.1f} | "
              f"{sum(e > 10 for e in flat):3}/{len(flat):<6} | {statistics.median(best):20.1f} | "
              f"{sum(e > 10 for e in best)}/{len(best)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
