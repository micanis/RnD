import cv2
import sys
import questionary
from pathlib import Path
from tqdm import tqdm
from abc import ABC, abstractmethod
from utils.paths import PATHS


# ============================================================
# 抽象基底クラス
# ============================================================

class VideoProcessor(ABC):
    """動画をフレーム画像に変換する抽象クラス"""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.cap = None
        self.save_dir = PATHS.output / "from_video" / self.video_path.stem
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def open_video(self):
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"❌ 読み込み失敗: {self.video_path.name}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.digit = len(str(self.total_frames))

    def release(self):
        if self.cap:
            self.cap.release()

    def run(self):
        """動画→画像出力の共通処理"""
        print(f"🚀 {self.video_path.name} -> {self.save_dir}")
        self.open_video()
        idx = 0

        with tqdm(total=self.total_frames, unit="frame") as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.process_frame(frame, idx)
                idx += 1
                pbar.update(1)
        self.release()

    @abstractmethod
    def process_frame(self, frame, idx: int):
        """フレームごとの処理（サブクラスで実装）"""
        pass


# ============================================================
# 通常動画クラス
# ============================================================

class NormalVideoProcessor(VideoProcessor):
    """通常動画のフレームをそのまま保存"""

    def process_frame(self, frame, idx: int):
        save_path = self.save_dir / f"{str(idx).zfill(self.digit)}.jpg"
        cv2.imwrite(str(save_path), frame)


# ============================================================
# 魚眼動画クラス
# ============================================================

class FisheyeVideoProcessor(VideoProcessor):
    """魚眼動画（左右に分割して保存）"""

    def __init__(self, video_path: Path):
        super().__init__(video_path)
        self.left_dir = self.save_dir / "left"
        self.right_dir = self.save_dir / "right"
        self.left_dir.mkdir(exist_ok=True)
        self.right_dir.mkdir(exist_ok=True)

    def process_frame(self, frame, idx: int):
        h, w, _ = frame.shape
        half_w = w // 2
        left = frame[:, :half_w, :]
        right = frame[:, half_w:, :]

        cv2.imwrite(str(self.left_dir / f"{str(idx).zfill(self.digit)}.jpg"), left)
        cv2.imwrite(str(self.right_dir / f"{str(idx).zfill(self.digit)}.jpg"), right)


# ============================================================
# ユーティリティ関数
# ============================================================

def select_videos(video_dir: Path) -> list[Path]:
    """input/video 内の動画を選択"""
    EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

    if not video_dir.exists():
        video_dir.mkdir(parents=True)
        print(f"📁 {video_dir} を作成しました。ここに動画を入れてください。")
        sys.exit()

    videos = [
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    if not videos:
        print(f"⚠️ {video_dir} に動画が見つかりません。")
        sys.exit()

    choices = [questionary.Choice(p.name, value=p) for p in videos]
    selected = questionary.checkbox(
        "処理する動画を選択してください (スペースで選択/解除 -> Enterで決定):",
        choices=choices
    ).ask()
    return selected


# ============================================================
# メイン処理
# ============================================================

def main():
    video_input_dir = PATHS.input / "video"
    target_videos = select_videos(video_input_dir)

    if not target_videos:
        print("キャンセルされました。")
        return

    mode = questionary.select(
        "動画の形式を選択してください:",
        choices=[
            "通常動画 (1映像)",
            "魚眼動画 (左右2映像)"
        ]
    ).ask()

    print(f"\n📹 {len(target_videos)} 本の動画を処理します... ({mode})\n")

    for video in target_videos:
        try:
            if mode == "通常動画 (1映像)":
                processor = NormalVideoProcessor(video)
            else:
                processor = FisheyeVideoProcessor(video)
            processor.run()
        except Exception as e:
            print(f"⚠️ {video.name} の処理中にエラー: {e}")

    print("\n✅ すべて完了しました！")


if __name__ == "__main__":
    main()
