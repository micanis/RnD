import sys
from pathlib import Path
from typing import Iterable

import cv2
import questionary

try:
    from utils.build_path import PATHS
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from utils.build_path import PATHS

from hpe.sapiens.sapiens_inference.detector import Detector


DEFAULT_INPUT_DIR = PATHS.output / "from_video"


def collect_images(image_dir: Path, extensions: Iterable[str], max_images: int) -> list[Path]:
    exts = {e if e.startswith(".") else f".{e}" for e in (e.lower() for e in extensions)}
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    return images[:max_images]


def average_boxes(detector: Detector, images: list[Path]) -> list[tuple[float, float]]:
    centers_by_index: list[list[tuple[float, float]]] = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skip unreadable image: {img_path}")
            continue

        bboxes = detector.detect(img)
        for i, bbox in enumerate(bboxes):
            if len(centers_by_index) <= i:
                centers_by_index.append([])
            x1, y1, x2, y2 = bbox[:4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers_by_index[i].append((cx, cy))

    averages: list[tuple[float, float]] = []
    for centers in centers_by_index:
        if centers:
            avg_x = sum(c[0] for c in centers) / len(centers)
            avg_y = sum(c[1] for c in centers) / len(centers)
            averages.append((avg_x, avg_y))
    return averages


def choose_coordinate(averages: list[tuple[float, float]]) -> tuple[float, float]:
    if not averages:
        raise RuntimeError("No bounding boxes detected in the provided images.")

    choices = [
        questionary.Choice(
            title=f"Person {i+1}: x={avg[0]:.1f}, y={avg[1]:.1f}",
            value=avg,
        )
        for i, avg in enumerate(averages)
    ]
    selection = questionary.select("使用する座標を選択してください:", choices=choices).ask()
    if selection is None:
        raise RuntimeError("Selection canceled.")
    return selection


def select_target_coordinate(
    image_dir: Path = DEFAULT_INPUT_DIR,
    extensions: Iterable[str] = (".jpg", ".jpeg", ".png"),
    max_images: int = 30,
    detector: Detector | None = None,
) -> tuple[float, float]:
    """
    画像フォルダ内の先頭max_images枚で人物検出し、インデックスごとの平均中心座標を算出。
    questionaryでユーザーに選ばせ、1つの(x, y)座標を返す。
    """
    images = collect_images(image_dir, extensions, max_images)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    det = detector or Detector()
    averages = average_boxes(det, images)
    return choose_coordinate(averages)
