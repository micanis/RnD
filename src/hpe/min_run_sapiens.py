import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from utils.paths import PATHS
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from utils.paths import PATHS

from sapiens.sapiens_inference import (
    SapiensPoseEstimation,
    SapiensPoseEstimationType,
)

# ---- 設定（必要に応じて書き換えてください） ----
# 任意の1枚の画像パスを指定する
IMAGE_PATH = PATHS.data / "sample.jpg"
# 出力は tmp/ 配下にまとめる
OUTPUT_DIR = PATHS.tmp / "sapiens_min"


def choose_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def save_keypoints_json(
    img_path: Path,
    keypoints_list,
    output_dir: Path,
    image_size: tuple[int, int],
) -> Path:
    height, width = image_size
    output_data = {
        "image_name": img_path.name,
        "image_size": {"width": width, "height": height},
        "persons": [],
    }

    for person_idx, keypoints in enumerate(keypoints_list):
        person_data = {"person_id": person_idx, "keypoints": {}}
        for joint_name, (x, y, confidence) in keypoints.items():
            person_data["keypoints"][joint_name] = {
                "x": float(x),
                "y": float(y),
                "confidence": float(confidence),
            }
        output_data["persons"].append(person_data)

    json_path = output_dir / f"{img_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    return json_path


def run_single_image() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device, dtype = choose_device()
    estimator = SapiensPoseEstimation(
        SapiensPoseEstimationType.POSE_ESTIMATION_1B,
        device=device,
        dtype=dtype,
    )

    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

    bboxes = estimator.detector.detect(img)
    if bboxes is None or len(bboxes) == 0:
        raise RuntimeError("No persons detected.")
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.tolist()

    result_img, keypoints_list = estimator.estimate_pose(img, bboxes, allowed_keypoints=None)
    json_path = save_keypoints_json(IMAGE_PATH, keypoints_list, OUTPUT_DIR, img.shape[:2])

    image_output_path = OUTPUT_DIR / f"{IMAGE_PATH.stem}_pose.jpg"
    cv2.imwrite(image_output_path, result_img)

    print(f"✓ Image saved: {image_output_path}")
    print(f"✓ JSON saved:  {json_path}")


if __name__ == "__main__":
    run_single_image()
