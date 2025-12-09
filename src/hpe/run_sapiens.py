import json
import sys
import time
from pathlib import Path
from typing import Iterable
import numpy as np

import cv2
import torch
import questionary

try:
    from src.utils.paths import PATHS, RESOLVE
    from src.utils.select_target import select_target_coordinate
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from utils.paths import PATHS, RESOLVE
    from utils.select_target import select_target_coordinate

from models.sapiens.sapiens_inference.pose import (
    SapiensPoseEstimation,
    SapiensPoseEstimationType,
)


FRAMES_ROOT = PATHS.interim / "frames"
HPE_MODEL_NAME = "sapiens"
DEFAULT_INPUT_DIR: Path | None = None  # questionaryで選択する前提
DEFAULT_OUTPUT_ROOT: Path | None = None  # 指定がなければ hpe_json/hpe_vis 下へ自動決定
DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png")

# 実行時に変更したい場合はここを編集する（CLIは提供しない）
RUN_CONFIG = {
    "input_dir": DEFAULT_INPUT_DIR,
    "output_root": DEFAULT_OUTPUT_ROOT,
    "extensions": DEFAULT_EXTENSIONS,
    "limit": None,  # 例: 10 にすると先頭10枚のみ
    "target_point": None,  # 例: (640.0, 360.0)
    "target_normalized": False,  # Trueなら0-1正規化座標として解釈
    "auto_select_target": True,  # Trueでutils.select_target経由の対話選択を有効化
    "max_images_for_target": 30,
}


def _select_dir(prompt: str, dirs: list[Path]) -> Path:
    if not dirs:
        raise RuntimeError(f"{prompt} 候補がありません。")
    selected = questionary.select(
        prompt,
        choices=[questionary.Choice(d.name, value=d) for d in sorted(dirs)],
    ).ask()
    if selected is None:
        raise RuntimeError("選択がキャンセルされました。")
    return selected


def choose_frames_dir(frames_root: Path = FRAMES_ROOT) -> Path:
    """
    data/interim/frames/<camera>/<subject>/<condition>/<surface> を順に選択し、
    入力ディレクトリを返す。
    """
    if not frames_root.exists():
        raise RuntimeError(f"入力元ディレクトリが見つかりません: {frames_root}")

    camera_dir = _select_dir("カメラディレクトリを選択してください:", [
        p for p in frames_root.iterdir() if p.is_dir()
    ])
    subject_dir = _select_dir("人物/シナリオディレクトリを選択してください:", [
        p for p in camera_dir.iterdir() if p.is_dir()
    ])
    condition_dir = _select_dir("条件ディレクトリを選択してください:", [
        p for p in subject_dir.iterdir() if p.is_dir()
    ])
    surface_dir = _select_dir("面 (single / left / right) を選択してください:", [
        p for p in condition_dir.iterdir() if p.is_dir()
    ])

    return surface_dir


def parse_frames_info(frames_dir: Path, frames_root: Path = FRAMES_ROOT) -> tuple[str, str, str, str]:
    rel = frames_dir.relative_to(frames_root)
    parts = rel.parts
    if len(parts) < 4:
        raise ValueError(
            f"想定パス data/interim/frames/<camera>/<subject>/<condition>/<surface> に合いません: {frames_dir}"
        )
    camera, subject, condition, surface = parts[:4]
    return camera, subject, condition, surface


def compute_output_dirs(camera: str, subject: str, condition: str, surface: str) -> tuple[Path, Path]:
    """
    出力先を hpe_vis / hpe_json で揃える。surface ごとにサブディレクトリを切る。
    """
    image_dir = RESOLVE.hpe_vis_dir(HPE_MODEL_NAME, camera, subject, condition) / surface
    json_dir = RESOLVE.hpe_json_dir(HPE_MODEL_NAME, camera, subject, condition) / surface
    return image_dir, json_dir

RIGHT_HAND_KEYPOINTS = {
    "right_thumb4",
    "right_thumb3",
    "right_thumb2",
    "right_thumb_third_joint",
    "right_forefinger4",
    "right_forefinger3",
    "right_forefinger2",
    "right_forefinger_third_joint",
    "right_middle_finger4",
    "right_middle_finger3",
    "right_middle_finger2",
    "right_middle_finger_third_joint",
    "right_ring_finger4",
    "right_ring_finger3",
    "right_ring_finger2",
    "right_ring_finger_third_joint",
    "right_pinky_finger4",
    "right_pinky_finger3",
    "right_pinky_finger2",
    "right_pinky_finger_third_joint",
    "right_wrist",
}

LEFT_HAND_KEYPOINTS = {
    "left_thumb4",
    "left_thumb3",
    "left_thumb2",
    "left_thumb_third_joint",
    "left_forefinger4",
    "left_forefinger3",
    "left_forefinger2",
    "left_forefinger_third_joint",
    "left_middle_finger4",
    "left_middle_finger3",
    "left_middle_finger2",
    "left_middle_finger_third_joint",
    "left_ring_finger4",
    "left_ring_finger3",
    "left_ring_finger2",
    "left_ring_finger_third_joint",
    "left_pinky_finger4",
    "left_pinky_finger3",
    "left_pinky_finger2",
    "left_pinky_finger_third_joint",
    "left_wrist",
}

UPPER_BODY_KEYPOINTS = {
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
}

TARGET_KEYPOINTS = RIGHT_HAND_KEYPOINTS | LEFT_HAND_KEYPOINTS | UPPER_BODY_KEYPOINTS


def prepare_paths(input_dir: Path, image_dir: Path, json_dir: Path) -> tuple[Path, Path, Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, image_dir, json_dir


def choose_device() -> tuple[torch.device, torch.dtype, str, float | None]:
    if torch.cuda.is_available():
        return (
            torch.device("cuda"),
            torch.float16,
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
        )
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16, "Apple Silicon GPU (MPS)", None
    return torch.device("cpu"), torch.float32, "CPU", None


def describe_device(device: torch.device, dtype: torch.dtype, name: str, memory: float | None) -> None:
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Device Name: {name}")
    if memory:
        print(f"GPU Memory: {memory:.2f} GB")
    print(f"Data Type: {dtype}")
    print("=" * 60)


def collect_images(input_dir: Path, extensions: Iterable[str]) -> list[Path]:
    normalized_exts = {
        ext if ext.startswith(".") else f".{ext}"
        for ext in (ext.lower() for ext in extensions)
    }

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in normalized_exts
    )
    return images


def save_keypoints_json(
    img_path: Path, keypoints_list, output_json_dir: Path, image_size: tuple[int, int]
) -> Path:
    height, width = image_size
    output_data = {
        "image_name": img_path.name,
        "image_size": {
            "width": width,
            "height": height,
        },
        "persons": [],
    }

    json_filename = img_path.stem + ".json"
    json_path = output_json_dir / json_filename
    for person_idx, keypoints in enumerate(keypoints_list):
        person_data = {"person_id": person_idx, "keypoints": {}}

        for joint_name, (x, y, confidence) in keypoints.items():
            person_data["keypoints"][joint_name] = {
                "x": float(x),
                "y": float(y),
                "confidence": float(confidence),
            }
        output_data["persons"].append(person_data)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return json_path


def resolve_target_point(
    target: tuple[float, float] | None, normalized: bool, image_shape: tuple[int, int, int]
) -> tuple[float, float] | None:
    if target is None:
        return None
    x, y = target
    if normalized:
        height, width = image_shape[:2]
        x *= width
        y *= height
    return float(x), float(y)


def select_bboxes_by_point(
    bboxes: list[list[float]] | np.ndarray,
    target_point: tuple[float, float] | None
) -> list[list[float]]:
    import numpy as np

    if bboxes is None:
        return []

    # ndarray の場合でも長さで判定
    if isinstance(bboxes, np.ndarray):
        if bboxes.size == 0:
            return []
        bboxes = bboxes.tolist()  # list化して統一
    elif len(bboxes) == 0:
        return []

    if target_point is None:
        return bboxes

    tx, ty = target_point

    def center(bbox: list[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox[:4]
        return (x1 + x2) / 2, (y1 + y2) / 2

    nearest = min(
        bboxes,
        key=lambda b: (center(b)[0] - tx) ** 2 + (center(b)[1] - ty) ** 2,
    )
    return [nearest]



def process_images(
    estimator: SapiensPoseEstimation,
    images: list[Path],
    output_image_dir: Path,
    output_json_dir: Path,
    target_point: tuple[float, float] | None,
    target_normalized: bool,
    allowed_keypoints: set[str] | None,
) -> None:
    total_start_time = time.time()

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] Processing: {img_path.name}")
        start_time = time.time()

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        bboxes = estimator.detector.detect(img)
        if bboxes is None or len(bboxes) == 0:
            print("No persons detected in this image.")
            continue

        resolved_target = resolve_target_point(target_point, target_normalized, img.shape)
        selected_bboxes = select_bboxes_by_point(bboxes, resolved_target)
        if not selected_bboxes:
            print("No bounding boxes matched target selection.")
            continue

        start_time = time.time()
        result_img, keypoints_list = estimator.estimate_pose(
            img, selected_bboxes, allowed_keypoints
        )
        processing_time = time.time() - start_time

        json_path = save_keypoints_json(
            img_path, keypoints_list, output_json_dir, img.shape[:2]
        )
        img_filename = img_path.stem + "_pose.jpg"
        img_output_path = output_image_dir / img_filename
        cv2.imwrite(img_output_path, result_img)

        print(f"  ✓ JSON saved: {json_path}")
        print(f"  ✓ Image saved: {img_output_path}")
        print(f"  ✓ Detected {len(keypoints_list)} person(s)")
        print(f"  ⏱️  Processing time: {processing_time:.2f}s")

    total_time = time.time() - total_start_time
    avg_time = total_time / len(images) if images else 0
    ips = len(images) / total_time if total_time else 0

    print(f"\n{'=' * 60}")
    print("All processing completed!")
    print("Results saved in:")
    print(f"  - Images: {output_image_dir}")
    print(f"  - JSON:   {output_json_dir}")
    print("\nPerformance:")
    print(f"  - Total images: {len(images)}")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Average time per image: {avg_time:.2f}s")
    print(f"  - Images per second: {ips:.2f}")
    print("=" * 60)


def run_sapiens(
    input_dir: Path | None = DEFAULT_INPUT_DIR,
    output_root: Path | None = DEFAULT_OUTPUT_ROOT,
    extensions: Iterable[str] = (".jpg", ".jpeg", ".png"),
    limit: int | None = None,
    target_point: tuple[float, float] | None = None,
    target_normalized: bool = False,
    auto_select_target: bool = False,
    max_images_for_target: int = 30,
    estimator: SapiensPoseEstimation | None = None,
) -> None:
    if input_dir is None:
        input_dir = choose_frames_dir(FRAMES_ROOT)

    camera, subject, condition, surface = parse_frames_info(input_dir, FRAMES_ROOT)

    if output_root is None:
        output_image_dir, output_json_dir = compute_output_dirs(
            camera, subject, condition, surface
        )
    else:
        output_image_dir = output_root / "images"
        output_json_dir = output_root / "json"

    device, dtype, device_name, gpu_memory = choose_device()
    describe_device(device, dtype, device_name, gpu_memory)

    print("\nLoading pose estimation model...")
    estimator = estimator or SapiensPoseEstimation(
        SapiensPoseEstimationType.POSE_ESTIMATION_1B, device=device, dtype=dtype
    )
    print("Model loaded!")

    if target_point is None and auto_select_target:
        target_point = select_target_coordinate(
            image_dir=input_dir,
            extensions=extensions,
            max_images=max_images_for_target,
            detector=estimator.detector,
        )

    input_dir, output_image_dir, output_json_dir = prepare_paths(
        input_dir, output_image_dir, output_json_dir
    )
    images = collect_images(input_dir, extensions)

    if limit:
        images = images[:limit]

    if not images:
        raise RuntimeError(f"No images found in {input_dir} for extensions {extensions}")

    print(f"Found {len(images)} images in {input_dir}")
    process_images(
        estimator,
        images,
        output_image_dir,
        output_json_dir,
        target_point,
        target_normalized,
        TARGET_KEYPOINTS,
    )


if __name__ == "__main__":
    try:
        run_sapiens(**RUN_CONFIG)
    except Exception as exc:
        print(f"Error during run_sapiens: {exc}")
        raise
