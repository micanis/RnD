import json
import sys
from pathlib import Path
from typing import Tuple, Dict

import cv2
import questionary

try:
    from src.utils.paths import PATHS
    from src.utils.cli_select import select_path, select_hierarchy
    from src.ds_model.camera_models import FisheyeCamera
    from src.ds_model.ply_io import BinaryPLYWriter
    from src.ds_model.visualizer_core import PointingVisualizer
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from utils.paths import PATHS
    from utils.cli_select import select_path, select_hierarchy
    from ds_model.camera_models import FisheyeCamera
    from ds_model.ply_io import BinaryPLYWriter
    from ds_model.visualizer_core import PointingVisualizer


def find_exp_dir() -> Path:
    exp_dirs = sorted(p for p in PATHS.experiments.iterdir() if p.is_dir() and p.name.startswith("exp001"))
    if not exp_dirs:
        raise RuntimeError(f"exp001* ディレクトリが見つかりません: {PATHS.experiments}")
    return exp_dirs[0]


def select_aruco_json() -> Path:
    root = PATHS.processed / "aruco_cord_json"
    if not root.exists():
        raise FileNotFoundError(f"ArUco結果ディレクトリがありません: {root}")
    json_files = sorted(root.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"{root} に json がありません。")
    selected = select_path(
        "使用する ArUco 結果 JSON を選択してください:",
        base_dir=root,
        choices=json_files,
        allow_manual=False,
        kind="file",
    )
    if selected is None:
        raise RuntimeError("選択がキャンセルされました。")
    return selected


def parse_image_and_hpe_paths(aruco_json: Path) -> tuple[Path, Path, str, str, str]:
    with aruco_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    img_path = Path(data["image"])
    frames_root = PATHS.interim / "frames" / "fisheye"
    rel = img_path.relative_to(frames_root)
    person, condition, surface, filename = rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
    hpe_json = PATHS.processed / "hpe_json" / "sapiens" / "fisheye" / person / condition / surface / (Path(filename).stem + ".json")
    return img_path, hpe_json, person, condition, surface


def load_hpe_keypoints(hpe_json: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    with hpe_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    persons = data.get("persons", [])
    if not persons:
        raise RuntimeError(f"HPE JSON に persons がありません: {hpe_json}")
    kpts = persons[0].get("keypoints", {})

    def _get(name: str) -> tuple[float, float]:
        coords = kpts.get(name)
        if not coords:
            raise RuntimeError(f"{name} が見つかりません: {hpe_json}")
        return float(coords["x"]), float(coords["y"])

    return _get("right_elbow"), _get("right_wrist")


def load_targets_from_aruco(aruco_json: Path) -> Dict[str, Tuple[float, float]]:
    with aruco_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    targets = {}
    for det in data.get("detections", []):
        name = det.get("name") or f"id_{det.get('id')}"
        center = det.get("center", {})
        targets[name] = (float(center["x"]), float(center["y"]))
    return targets


def save_summary(output_dir: Path, image_path: Path, hpe_path: Path, elbow, wrist, targets: Dict[str, Tuple[float, float]], aruco_json: Path):
    summary = {
        "image": str(image_path),
        "hpe_json": str(hpe_path),
        "aruco_json": str(aruco_json),
        "right_elbow": {"x": elbow[0], "y": elbow[1]},
        "right_wrist": {"x": wrist[0], "y": wrist[1]},
        "detections": [{"name": k, "x": v[0], "y": v[1]} for k, v in targets.items()],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    aruco_json = select_aruco_json()
    image_path, hpe_json, person, condition, surface = parse_image_and_hpe_paths(aruco_json)
    elbow_uv, wrist_uv = load_hpe_keypoints(hpe_json)
    targets = load_targets_from_aruco(aruco_json)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {image_path}")
    H, W = img.shape[:2]

    camera_model = FisheyeCamera(width=W, height=H, fov_deg=200)
    exp_dir = find_exp_dir()
    output_dir = exp_dir / "pointing" / f"{person}_{condition}_{surface}_{Path(image_path).stem}"
    ply_writer = BinaryPLYWriter()
    app = PointingVisualizer(camera_model, ply_writer, str(output_dir))

    app.process_background(img)
    app.process_scene(elbow_uv, wrist_uv, targets)
    app.render_final_image("result.jpg")

    save_summary(output_dir, image_path, hpe_json, elbow_uv, wrist_uv, targets, aruco_json)
    print(f"出力: {output_dir}")


if __name__ == "__main__":
    main()
