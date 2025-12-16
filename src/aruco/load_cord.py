import json
from pathlib import Path

import cv2
import numpy as np
import questionary

try:
    from src.utils.paths import PATHS
    from src.utils.cli_select import select_hierarchy
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SRC_DIR = PROJECT_ROOT / "src"
    from utils.paths import PATHS
    from utils.cli_select import select_hierarchy


ARUCO_REF_DIR = PATHS.raw / "image" / "aruco"
FRAME_ROOT = PATHS.interim / "frames"
OUTPUT_PATH = PATHS.processed / "aruco_cord_json"


def load_reference_markers(detector) -> dict[int, str]:
    registered_objects: dict[int, str] = {}
    for ref_path in sorted(ARUCO_REF_DIR.glob("marker_*.jpg")):
        filename = ref_path.name
        try:
            obj_name = filename.split("_")[2].replace(".jpg", "")
        except IndexError:
            obj_name = filename

        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            continue
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray_ref)
        if ids is not None:
            marker_id = int(ids[0][0])
            registered_objects[marker_id] = obj_name
    return registered_objects


def select_target_image() -> Path | None:
    """
    data/interim/frames/<camera>/<person>/<condition>/*.jpg から1枚選択
    例: data/interim/frames/fisheye/person00/light_on/0001.jpg
    """
    if not FRAME_ROOT.exists():
        return None

    try:
        condition_dir = select_hierarchy(
            FRAME_ROOT,
            [
                ("カメラディレクトリを選択してください:", lambda p: (d for d in p.iterdir() if d.is_dir())),
                ("人物ディレクトリを選択してください:", lambda p: (d for d in p.iterdir() if d.is_dir())),
                ("条件ディレクトリを選択してください:", lambda p: (d for d in p.iterdir() if d.is_dir())),
                ("どちらの面を選択しますか:", lambda p: (d for d in p.iterdir() if d.is_dir()))
            ],
        )
    except RuntimeError:
        return None

    images = sorted(condition_dir.glob("*.jpg"))
    if not images:
        return None

    selection = questionary.select(
        "解析する画像を選択してください",
        choices=[
            questionary.Choice(str(p.relative_to(FRAME_ROOT)), value=p) for p in images
        ],
    ).ask()
    return selection


def main():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    registered_objects = load_reference_markers(detector)
    if not registered_objects:
        raise RuntimeError(f"参照マーカーが見つかりませんでした: {ARUCO_REF_DIR}")

    target_image_path = select_target_image()
    if target_image_path is None:
        raise RuntimeError("ファイルが見つかりませんでした")
    
    rel = target_image_path.relative_to(FRAME_ROOT)
    _, person, condition, _, _ = rel.parts[:5]
    output_file_name = f"{person}_{condition}_result.json"

    scene_img = cv2.imread(str(target_image_path))
    if scene_img is None:
        raise FileNotFoundError(f"ターゲット画像が読み込めません: {target_image_path}")

    gray_scene = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray_scene)

    detections = []
    if ids is not None:
        for i, detected_id in enumerate(ids.flatten()):
            detected_id = int(detected_id)
            pts = corners[i][0]
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            obj_name = registered_objects.get(detected_id, None)
            detections.append(
                {
                    "id": detected_id,
                    "name": obj_name,
                    "center": {"x": cx, "y": cy},
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.joinpath(output_file_name).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "image": str(target_image_path),
                "detections": detections,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
