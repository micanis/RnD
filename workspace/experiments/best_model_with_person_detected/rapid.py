from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
DEFAULT_IMAGE_DIR = EXPERIMENT_DIR / "images"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "rapid"
DEFAULT_RAPID_DIR = PROJECT_ROOT / "workspace" / "src" / "RAPiD"
DEFAULT_WEIGHTS_PATH = DEFAULT_RAPID_DIR / "weights" / "pL1_MWHB1024_Mar11_4000.ckpt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(image_dir: Path) -> list[Path]:
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name.startswith(("cepdof_", "real_"))
    )
    if not image_paths:
        raise FileNotFoundError(f"no experiment images found in: {image_dir}")
    return image_paths


def load_detector_class(rapid_dir: Path):
    if not rapid_dir.exists():
        raise FileNotFoundError(f"RAPiD directory does not exist: {rapid_dir}")

    sys.path.insert(0, str(rapid_dir))
    try:
        api = importlib.import_module("api")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"failed to import RAPiD api.py from {rapid_dir}"
        ) from exc

    if not hasattr(api, "Detector"):
        raise AttributeError(f"Detector class does not exist in {rapid_dir / 'api.py'}")
    return api.Detector


def make_detector(
    rapid_dir: Path,
    weights_path: Path,
    use_cuda: bool,
    input_size: int,
    conf_thres: float,
):
    if not weights_path.exists():
        raise FileNotFoundError(f"RAPiD weights file does not exist: {weights_path}")

    Detector = load_detector_class(rapid_dir)
    return Detector(
        model_name="rapid",
        weights_path=str(weights_path),
        use_cuda=use_cuda,
        input_size=input_size,
        conf_thres=conf_thres,
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return to_jsonable(value.detach().cpu().numpy())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def count_detections(detections: Any) -> int:
    if detections is None:
        return 0
    if isinstance(detections, np.ndarray):
        return int(len(detections))
    if hasattr(detections, "detach") and hasattr(detections, "cpu"):
        return count_detections(detections.detach().cpu().numpy())
    if isinstance(detections, dict):
        for key in ("detections", "boxes", "bboxes", "bbox"):
            if key in detections:
                return count_detections(detections[key])
        return len(detections)
    if isinstance(detections, (list, tuple)):
        return len(detections)
    return 1


def detect_one(detector: Any, image_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_rapid.jpg"
    json_path = output_dir / f"{image_path.stem}_rapid.json"

    start = time.perf_counter()
    detections = detector.detect_one(
        img_path=str(image_path),
        return_img=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    rendered = detector.detect_one(
        img_path=str(image_path),
        return_img=True,
    )

    Image.fromarray(rendered).save(output_path)
    result = {
        "model": "RAPiD",
        "image": str(image_path),
        "rendered_image": str(output_path),
        "detections_json": str(json_path),
        "inference_ms": elapsed_ms,
        "num_detections": count_detections(detections),
        "detections": to_jsonable(detections),
    }
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAPiD person detection on prepared fisheye jpg images."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rapid-dir", type=Path, default=DEFAULT_RAPID_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--cpu", action="store_true", help="disable CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = make_detector(
        rapid_dir=args.rapid_dir,
        weights_path=args.weights,
        use_cuda=not args.cpu,
        input_size=args.input_size,
        conf_thres=args.conf,
    )

    model_size_mb = args.weights.stat().st_size / (1024 * 1024)
    results = []
    for image_path in collect_images(args.image_dir):
        result = detect_one(detector, image_path, args.output_dir)
        result["weights_path"] = str(args.weights)
        result["model_size_mb"] = model_size_mb
        results.append(result)
        print(f"{image_path.name}: {result['inference_ms']:.1f} ms")

    summary_path = args.output_dir / "summary_rapid.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
