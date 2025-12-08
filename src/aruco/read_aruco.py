from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Detect ArUco markers and print their positions.")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image in which to detect markers.",
    )
    parser.add_argument(
        "--marker-tv",
        type=Path,
        default=root / "marker_00_TV.jpg",
        help="Reference marker file (not used for detection, kept for traceability).",
    )
    parser.add_argument(
        "--marker-ac",
        type=Path,
        default=root / "marker_01_AC.jpg",
        help="Reference marker file (not used for detection, kept for traceability).",
    )
    parser.add_argument(
        "--marker-light",
        type=Path,
        default=root / "marker_02_Light.jpg",
        help="Reference marker file (not used for detection, kept for traceability).",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="DICT_4X4_50",
        help="OpenCV ArUco dictionary name (e.g., DICT_4X4_50, DICT_5X5_100, DICT_6X6_250).",
    )
    return parser.parse_args()


def get_aruco_dictionary(name: str) -> cv2.aruco_Dictionary:
    try:
        aruco_module = cv2.aruco
    except AttributeError as exc:
        raise RuntimeError("cv2.aruco is not available; ensure opencv-contrib-python is installed.") from exc

    dict_id = getattr(aruco_module, name, None)
    if dict_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return aruco_module.getPredefinedDictionary(dict_id)


def detect_markers(image_path: Path, dictionary_name: str) -> list[dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = get_aruco_dictionary(dictionary_name)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    results: list[dict] = []
    if ids is None:
        return results

    for marker_id, marker_corners in zip(ids.flatten(), corners):
        pts = marker_corners.reshape((4, 2))
        center_x = float(pts[:, 0].mean())
        center_y = float(pts[:, 1].mean())
        results.append(
            {
                "id": int(marker_id),
                "corners": pts.tolist(),  # [[x,y], ...] order: tl, tr, br, bl
                "center": [center_x, center_y],
            }
        )
    return results


def main() -> None:
    args = parse_args()
    detections = detect_markers(args.image, args.dictionary)

    if not detections:
        print(f"No markers found in {args.image}")
        return

    print(f"Image: {args.image}")
    print(f"Reference markers: tv={args.marker_tv}, ac={args.marker_ac}, light={args.marker_light}")
    for det in detections:
        cx, cy = det["center"]
        print(f"ID {det['id']:>3} center=({cx:.1f}, {cy:.1f}) corners={det['corners']}")


if __name__ == "__main__":
    main()
