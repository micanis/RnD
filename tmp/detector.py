import numpy as np
import zarr
from ultralytics import YOLO


def load_image_from_zarr(zarr_path: str) -> np.ndarray:
    """zarrファイルから画像を読み込む"""
    return zarr.open(zarr_path, mode="r")[:]


def detect_person(img: np.ndarray, model_path: str = "yolov8m.pt") -> list[int] | None:
    """
    画像から人物を検出し、バウンディングボックスを返す。

    Args:
        img: 入力画像 (numpy array)
        model_path: YOLOモデルのパス

    Returns:
        [x1, y1, x2, y2] 形式のバウンディングボックス。人物が検出されない場合はNone。
    """
    model = YOLO(model_path)
    results = model(img, classes=[0], verbose=False)  # class 0 = person

    if len(results) == 0 or len(results[0].boxes) == 0:
        return None

    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]


def detect_person_from_zarr(zarr_path: str, model_path: str = "yolov8m.pt") -> list[int] | None:
    """zarrファイルから画像を読み込み、人物検出を行う"""
    img = load_image_from_zarr(zarr_path)
    return detect_person(img, model_path)
