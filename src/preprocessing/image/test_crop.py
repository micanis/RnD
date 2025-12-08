from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

from utils.paths import PATHS

# 入出力パス
# 入力画像はhpeする前のパス、座標は検出後のjsonファイルのパス
IMAGE_PATH = PATHS.output / "from_video" / "person00_light_on" / "left" / "00.jpg"
JSON_PATH  = PATHS.output / "hpe" / "sapiens" / "person00_light_on" / "left" / "json"   / "00.json"
OUT_DEBUG  = PATHS.root / "tmp" / "test_crop_debug.png"     # キーポイント可視化
OUT_CROP   = PATHS.root / "tmp" / "right_hand_crop.png"     # 右手クロップ結果

# 設定
CONF_THRESHOLD = 0.05         # キーポイント採用の下限
BBOX_MARGIN_RATIO = 1.00      # bboxに上下左右へ足す余白（正方形辺長に対する割合）
OUTPUT_SIZE = (128, 128)      # 必要ならリサイズ

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


def load_keypoints(json_path: Path) -> Dict[str, Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    persons = data.get("persons", [])
    if not persons:
        return {}
    return persons[0].get("keypoints", {})


def draw_keypoints(img: np.ndarray, keypoints: Dict[str, Dict]) -> np.ndarray:
    vis = img.copy()
    for name, coords in keypoints.items():
        x = int(coords["x"])
        y = int(coords["y"])
        conf = float(coords.get("confidence", 0))
        color = (0, 255, 0) if conf >= 0.3 else (0, 128, 255)
        cv2.circle(vis, (x, y), 3, color, -1)
        cv2.putText(vis, name, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return vis


def collect_right_hand_points(keypoints: Dict[str, dict], conf_thr: float) -> np.ndarray:
    pts = []
    for name in RIGHT_HAND_KEYPOINTS:  # 手の21点のみ採用
        if name not in keypoints:
            continue
        coords = keypoints[name]
        if float(coords.get("confidence", 0.0)) < conf_thr:
            continue
        pts.append([float(coords["x"]), float(coords["y"])])
    return np.asarray(pts, dtype=np.float32)  # (N,2)



def compute_square_bbox_from_points(pts: np.ndarray, img_shape: Tuple[int, int, int],
                                    margin_ratio: float = 0.3) -> Optional[Tuple[int, int, int, int]]:
    """
    2D点群から正方形bbox (x1,y1,x2,y2) を作る。余白は margin_ratio で指定。
    画像外にはみ出さないようクリップする。
    """
    if pts.size == 0:
        return None

    h, w = img_shape[:2]
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)

    # 中心と一辺（最大辺に合わせる）
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    side = max(max_x - min_x, max_y - min_y)

    # 余白を追加して正方形
    side *= (1.0 + 2.0 * margin_ratio)
    half = side * 0.5

    x1 = int(np.floor(cx - half))
    y1 = int(np.floor(cy - half))
    x2 = int(np.ceil (cx + half))
    y2 = int(np.ceil (cy + half))

    # 画像内にクリップ
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_with_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int],
                   output_size: Tuple[int, int] | None = None) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2].copy()
    if output_size is not None:
        crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
    return crop


def main():
    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        raise FileNotFoundError(f"画像が読み込めませんでした: {IMAGE_PATH}")

    keypoints = load_keypoints(JSON_PATH)
    if not keypoints:
        raise ValueError(f"キーポイントが空です: {JSON_PATH}")

    # デバッグ表示
    vis = draw_keypoints(img, keypoints)
    OUT_DEBUG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DEBUG), vis)

    # 右手点群抽出 → bbox作成 → クロップ
    right_pts = collect_right_hand_points(keypoints, CONF_THRESHOLD)  # (N,2)
    bbox = compute_square_bbox_from_points(right_pts, img.shape, BBOX_MARGIN_RATIO)
    if bbox is None:
        raise RuntimeError("右手のbboxが作成できませんでした（有効点不足）")

    crop = crop_with_bbox(img, bbox, OUTPUT_SIZE)
    OUT_CROP.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_CROP), crop)

    print(f"✅ デバッグ: {OUT_DEBUG}")
    print(f"✅ 右手クロップ: {OUT_CROP}  bbox={bbox}, size={crop.shape[1]}x{crop.shape[0]}")


if __name__ == "__main__":
    main()
