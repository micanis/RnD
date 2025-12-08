from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import questionary

from utils.paths import PATHS

CANVAS_SIZE = 512
MARGIN_RATIO = 0.1
FPS = 15
NODE_COLOR = (255, 255, 255)
EDGE_COLOR = (180, 180, 180)
NODE_RADIUS = 4
EDGE_THICKNESS = 2
CONF_THRESHOLD = 0.05

BODY_EDGES: list[tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]

HAND_EDGES: list[tuple[str, str]] = [
    # left hand
    ("left_wrist", "left_thumb_third_joint"),
    ("left_thumb_third_joint", "left_thumb2"),
    ("left_thumb2", "left_thumb3"),
    ("left_thumb3", "left_thumb4"),
    ("left_wrist", "left_forefinger_third_joint"),
    ("left_forefinger_third_joint", "left_forefinger2"),
    ("left_forefinger2", "left_forefinger3"),
    ("left_forefinger3", "left_forefinger4"),
    ("left_wrist", "left_middle_finger_third_joint"),
    ("left_middle_finger_third_joint", "left_middle_finger2"),
    ("left_middle_finger2", "left_middle_finger3"),
    ("left_middle_finger3", "left_middle_finger4"),
    ("left_wrist", "left_ring_finger_third_joint"),
    ("left_ring_finger_third_joint", "left_ring_finger2"),
    ("left_ring_finger2", "left_ring_finger3"),
    ("left_ring_finger3", "left_ring_finger4"),
    ("left_wrist", "left_pinky_finger_third_joint"),
    ("left_pinky_finger_third_joint", "left_pinky_finger2"),
    ("left_pinky_finger2", "left_pinky_finger3"),
    ("left_pinky_finger3", "left_pinky_finger4"),
    # right hand
    ("right_wrist", "right_thumb_third_joint"),
    ("right_thumb_third_joint", "right_thumb2"),
    ("right_thumb2", "right_thumb3"),
    ("right_thumb3", "right_thumb4"),
    ("right_wrist", "right_forefinger_third_joint"),
    ("right_forefinger_third_joint", "right_forefinger2"),
    ("right_forefinger2", "right_forefinger3"),
    ("right_forefinger3", "right_forefinger4"),
    ("right_wrist", "right_middle_finger_third_joint"),
    ("right_middle_finger_third_joint", "right_middle_finger2"),
    ("right_middle_finger2", "right_middle_finger3"),
    ("right_middle_finger3", "right_middle_finger4"),
    ("right_wrist", "right_ring_finger_third_joint"),
    ("right_ring_finger_third_joint", "right_ring_finger2"),
    ("right_ring_finger2", "right_ring_finger3"),
    ("right_ring_finger3", "right_ring_finger4"),
    ("right_wrist", "right_pinky_finger_third_joint"),
    ("right_pinky_finger_third_joint", "right_pinky_finger2"),
    ("right_pinky_finger2", "right_pinky_finger3"),
    ("right_pinky_finger3", "right_pinky_finger4"),
]

EDGES = BODY_EDGES + HAND_EDGES
BASIC_UPPER_BODY_KEYS = {
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
}


def collect_nodes(edges: Iterable[tuple[str, str]]) -> set[str]:
    nodes: set[str] = set()
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)
    return nodes


BODY_KEYS = collect_nodes(BODY_EDGES)
HAND_KEYS = collect_nodes(HAND_EDGES)
LEFT_HAND_KEYS = {k for k in HAND_KEYS if k.startswith("left_")} | {"left_wrist"}
RIGHT_HAND_KEYS = {k for k in HAND_KEYS if k.startswith("right_")} | {"right_wrist"}
WHOLE_BODY_KEYS = BODY_KEYS | HAND_KEYS

MODES = {
    "全身": {"filename": "whole_body.mp4", "keys": WHOLE_BODY_KEYS},
    "上半身のみ": {"filename": "upper_body.mp4", "keys": BASIC_UPPER_BODY_KEYS},
    "右手のみ": {"filename": "right_hand.mp4", "keys": RIGHT_HAND_KEYS},
    "左手のみ": {"filename": "left_hand.mp4", "keys": LEFT_HAND_KEYS},
}


def find_json_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("json") if p.is_dir() and any(p.glob("*.json"))
    )


def choose_json_dir() -> Path | None:
    root = PATHS.output / "hpe"
    json_dirs = find_json_dirs(root)
    if not json_dirs:
        print(f"⚠️ jsonフォルダが見つかりませんでした: {root}")
        return None

    choices = [
        questionary.Choice(str(p.relative_to(PATHS.output)), value=p)
        for p in json_dirs
    ]
    selected = questionary.select(
        "使用する /output/hpe/ 以下の json ディレクトリを選択してください:",
        choices=choices,
    ).ask()
    return selected


def choose_mode() -> tuple[str, set[str]] | None:
    selection = questionary.select(
        "描画する部位を選択してください:",
        choices=[questionary.Choice(label, value=label) for label in MODES.keys()],
    ).ask()
    if selection is None:
        return None
    mode = MODES[selection]
    return mode["filename"], set(mode["keys"])


def load_keypoints(json_path: Path) -> Dict[str, Tuple[float, float]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    persons = data.get("persons", [])
    if not persons:
        return {}

    person = persons[0]
    keypoints = {}
    for name, coords in person.get("keypoints", {}).items():
        if coords.get("confidence", 0) < CONF_THRESHOLD:
            continue
        keypoints[name] = (float(coords["x"]), float(coords["y"]))
    return keypoints


def normalize_points(keypoints: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[int, int]]:
    xs = [p[0] for p in keypoints.values()]
    ys = [p[1] for p in keypoints.values()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    bbox_size = max(bbox_w, bbox_h)

    if bbox_size == 0:
        scale = 1.0
    else:
        scale = (1 - 2 * MARGIN_RATIO) * CANVAS_SIZE / bbox_size

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    normalized = {}
    for name, (x, y) in keypoints.items():
        nx = (x - cx) * scale + CANVAS_SIZE / 2
        ny = (y - cy) * scale + CANVAS_SIZE / 2
        normalized[name] = (int(round(nx)), int(round(ny)))
    return normalized


def draw_frame(points: Dict[str, Tuple[int, int]], edges: list[tuple[str, str]]) -> np.ndarray:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)

    for a, b in edges:
        if a in points and b in points:
            cv2.line(canvas, points[a], points[b], EDGE_COLOR, EDGE_THICKNESS)
    for p in points.values():
        cv2.circle(canvas, p, NODE_RADIUS, NODE_COLOR, -1)
    return canvas


def ensure_output_path(json_dir: Path, filename: str) -> Path:
    video_dir = json_dir.parent / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir / filename


def main():
    json_dir = choose_json_dir()
    if json_dir is None:
        print("キャンセルしました。")
        return

    mode = choose_mode()
    if mode is None:
        print("キャンセルしました。")
        return
    filename, allowed_keys = mode
    mode_edges = [e for e in EDGES if e[0] in allowed_keys and e[1] in allowed_keys]

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"⚠️ JSONが見つかりませんでした: {json_dir}")
        return

    output_path = ensure_output_path(json_dir, filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (CANVAS_SIZE, CANVAS_SIZE))

    frame_count = 0
    for json_file in json_files:
        keypoints = load_keypoints(json_file)
        if not keypoints:
            continue
        filtered = {k: v for k, v in keypoints.items() if k in allowed_keys}
        if not filtered:
            continue
        points = normalize_points(filtered)
        frame = draw_frame(points, mode_edges)
        writer.write(frame)
        frame_count += 1

    writer.release()
    if frame_count == 0:
        print("⚠️ 描画できるフレームがありませんでした。")
        output_path.unlink(missing_ok=True)
        return

    print(f"✅ {frame_count} フレームを書き出しました: {output_path}")


if __name__ == "__main__":
    main()
