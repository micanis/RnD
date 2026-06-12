from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = EXPERIMENT_DIR / "images"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "yolo"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "yolov8m.pt"
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


def detect_one(model: YOLO, image_path: Path, confidence_threshold: float) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {image_path}")

    start = time.perf_counter()
    results = model.predict(
        image_bgr,
        classes=[PERSON_CLASS_ID],
        conf=confidence_threshold,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

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
        "inference_ms": elapsed_ms,
        "detections": detections,
        "rendered_bgr": image_bgr,
    }


def write_result(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(result["image"])
    rendered_path = output_dir / f"{image_path.stem}_yolo.jpg"
    json_path = output_dir / f"{image_path.stem}_yolo.json"

    rendered_bgr = result.pop("rendered_bgr")
    ok = cv2.imwrite(str(rendered_path), rendered_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError(f"failed to write rendered image: {rendered_path}")

    result["rendered_image"] = str(rendered_path)
    result["detections_json"] = str(json_path)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO person detection on the prepared fisheye jpg images."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"YOLO model does not exist: {args.model}")

    model = YOLO(str(args.model))
    model_size_mb = args.model.stat().st_size / (1024 * 1024)
    results = []

    for image_path in collect_images(args.image_dir):
        result = detect_one(model, image_path, args.conf)
        result["model"] = "YOLO"
        result["model_path"] = str(args.model)
        result["model_size_mb"] = model_size_mb
        results.append(write_result(result, args.output_dir))
        print(
            f"{image_path.name}: {len(result['detections'])} detections, "
            f"{result['inference_ms']:.1f} ms"
        )

    summary_path = args.output_dir / "summary_yolo.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
