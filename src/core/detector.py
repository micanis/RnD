from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """単一フレームの検出結果"""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float


@dataclass
class TrackedPerson:
    """トラッキング付き検出結果"""

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float


class PersonDetector:
    """人物検出・トラッキングのベースクラス"""

    def __init__(self, model_path: str = "yolov8m.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        単一フレームの人物検出（トラッキングなし）

        Parameters
        ----------
        frame : np.ndarray
            RGB画像 (H, W, 3)

        Returns
        -------
        list[Detection]
            検出された人物のリスト
        """
        results = self.model.predict(frame, classes=[0], verbose=False, conf=0.6)

        if results[0].boxes is None:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        return [Detection(bbox=box, confidence=conf) for box, conf in zip(boxes, confs)]

    def track(self, frame: np.ndarray) -> list[TrackedPerson]:
        """
        単一フレームの人物トラッキング（persist=True）

        連続フレームで呼び出すことでIDが維持される。
        新しいシーケンスを開始する場合は reset() を呼ぶこと。

        Parameters
        ----------
        frame : np.ndarray
            RGB画像 (H, W, 3)

        Returns
        -------
        list[TrackedPerson]
            トラッキングされた人物のリスト
        """
        results = self.model.track(
            frame, persist=True, classes=[0], verbose=False, conf=0.6
        )

        if results[0].boxes is None or results[0].boxes.id is None:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        return [
            TrackedPerson(track_id=int(tid), bbox=box, confidence=conf)
            for tid, box, conf in zip(ids, boxes, confs)
        ]

    def reset(self) -> None:
        """トラッカーの状態をリセット（新しいシーケンス開始時に呼ぶ）

        YOLO モデル自体には reset_tracker() は存在しない。
        tracker オブジェクト（BYTETracker / BOTSORT）は model.track() を初回呼び出した
        後に model.predictor.trackers として初期化されるため、存在確認を行ってからリセットする。
        """
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return
        trackers = getattr(predictor, "trackers", None)
        if not trackers:
            return
        for tracker in trackers:
            tracker.reset()
