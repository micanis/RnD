from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import ultralytics
from ultralytics import YOLO


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = EXPERIMENT_DIR / "images"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "yolo"
DEFAULT_MODELS = ["yolo26m.pt", "yolo11m.pt", "models/yolov8m.pt"]
PERSON_CLASS_ID = 0


def collect_images(image_dir: Path) -> list[Path]:
    image_paths = sorted(
        path
        for path in image_dir.glob("*.jpg")
        if path.name.startswith(("cepdof_", "real_"))
    )
    if not image_paths:
        raise FileNotFoundError(f"no experiment jpg images found in: {image_dir}")
    return image_paths


def draw_detections(image_bgr, detections: list[dict[str, Any]]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
        confidence = detection["confidence"]
        label = f"person {confidence:.2f}"
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_bgr,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def predict_person(model: YOLO, image_bgr, confidence_threshold: float):
    return model.predict(
        image_bgr,
        classes=[PERSON_CLASS_ID],
        conf=confidence_threshold,
        verbose=False,
    )


def timed_predict(
    model: YOLO,
    image_bgr,
    confidence_threshold: float,
    warmup_runs: int,
    measure_runs: int,
) -> tuple[Any, list[float]]:
    for _ in range(warmup_runs):
        predict_person(model, image_bgr, confidence_threshold)

    results = None
    elapsed_times_ms: list[float] = []
    for _ in range(measure_runs):
        start = time.perf_counter()
        results = predict_person(model, image_bgr, confidence_threshold)
        elapsed_times_ms.append((time.perf_counter() - start) * 1000)

    return results, elapsed_times_ms


def summarize_times(elapsed_times_ms: list[float]) -> dict[str, Any]:
    return {
        "inference_ms": statistics.mean(elapsed_times_ms),
        "inference_ms_median": statistics.median(elapsed_times_ms),
        "inference_ms_min": min(elapsed_times_ms),
        "inference_ms_runs": elapsed_times_ms,
    }


def detect_one(
    model: YOLO,
    image_path: Path,
    confidence_threshold: float,
    warmup_runs: int,
    measure_runs: int,
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {image_path}")

    results, elapsed_times_ms = timed_predict(
        model,
        image_bgr,
        confidence_threshold,
        warmup_runs,
        measure_runs,
    )

    detections: list[dict[str, Any]] = []
    boxes = results[0].boxes
    if boxes is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        for box, confidence, class_id in zip(xyxy, confidences, classes):
            detections.append(
                {
                    "label": "person",
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "bbox_xyxy": [float(value) for value in box],
                }
            )

    draw_detections(image_bgr, detections)
    return {
        "image": str(image_path),
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "detections": detections,
        "rendered_bgr": image_bgr,
        **summarize_times(elapsed_times_ms),
    }


def safe_model_name(model_ref: str) -> str:
    return Path(model_ref).stem.replace("/", "_")


def get_model_size_mb(model: YOLO, model_ref: str) -> float | None:
    candidate_paths = [Path(model_ref)]
    model_attr = getattr(model, "model", None)
    for attr_name in ("pt_path", "yaml_file"):
        attr_value = getattr(model_attr, attr_name, None)
        if attr_value:
            candidate_paths.append(Path(attr_value))

    ckpt_path = getattr(model, "ckpt_path", None)
    if ckpt_path:
        candidate_paths.append(Path(ckpt_path))

    for path in candidate_paths:
        if path.exists() and path.is_file():
            return path.stat().st_size / (1024 * 1024)
    return None


def write_result(
    result: dict[str, Any], output_dir: Path, model_name: str
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(result["image"])
    rendered_path = output_dir / f"{image_path.stem}_{model_name}.jpg"
    json_path = output_dir / f"{image_path.stem}_{model_name}.json"

    rendered_bgr = result.pop("rendered_bgr")
    ok = cv2.imwrite(str(rendered_path), rendered_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError(f"failed to write rendered image: {rendered_path}")

    result["rendered_image"] = str(rendered_path)
    result["detections_json"] = str(json_path)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO person detection on the prepared fisheye jpg images."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Ultralytics model names or local paths. "
            "Default: yolo26m.pt yolo11m.pt models/yolov8m.pt"
        ),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measure-runs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    images = collect_images(args.image_dir)

    for model_ref in args.models:
        model_ref = str(model_ref)
        model = YOLO(model_ref)
        model_name = safe_model_name(model_ref)
        model_size_mb = get_model_size_mb(model, model_ref)

        model_output_dir = args.output_dir / model_name
        for image_path in images:
            result = detect_one(
                model,
                image_path,
                args.conf,
                args.warmup_runs,
                args.measure_runs,
            )
            result["model"] = "YOLO"
            result["model_variant"] = model_name
            result["model_path"] = model_ref
            result["model_size_mb"] = model_size_mb
            result["ultralytics_version"] = ultralytics.__version__
            result["warmup_runs"] = args.warmup_runs
            result["measure_runs"] = args.measure_runs
            results.append(write_result(result, model_output_dir, model_name))
            print(
                f"{model_name} {image_path.name}: "
                f"{len(result['detections'])} detections, "
                f"{result['inference_ms']:.1f} ms avg"
            )

    summary_path = args.output_dir / "summary_yolo.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
