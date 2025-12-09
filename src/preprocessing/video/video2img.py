import sys
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import questionary
from tqdm import tqdm

try:
    from src.utils.paths import PATHS, RESOLVE
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from utils.paths import PATHS, RESOLVE


def parse_video_info(video_path: Path, video_root: Path) -> tuple[str, str, str]:
    """data/raw/video/<camera>/<subject>/<condition>.* からメタ情報を抽出"""
    try:
        rel = video_path.relative_to(video_root)
    except ValueError:
        raise ValueError(f"{video_path} は {video_root} 配下ではありません。")

    parts = rel.parts
    if len(parts) < 3:
        raise ValueError(
            f"想定パス data/raw/video/<camera>/<person>/<condition>.* に合いません: {video_path}"
        )

    camera, subject = parts[0], parts[1]
    condition = Path(parts[-1]).stem
    return camera, subject, condition


def choose_dir(prompt: str, dirs: list[Path]) -> Path | None:
    if not dirs:
        return None
    dirs = sorted(dirs)
    selection = questionary.select(
        prompt,
        choices=[questionary.Choice(d.name, value=d) for d in dirs],
    ).ask()
    return selection


# ============================================================
# 抽象基底クラス
# ============================================================

class VideoProcessor(ABC):
    """動画をフレーム画像に変換する抽象クラス"""

    def __init__(self, video_path: Path, camera: str, subject: str, condition: str):
        self.video_path = video_path
        self.camera = camera
        self.subject = subject
        self.condition = condition
        self.cap = None
        self.save_dir: Path | None = None

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

    def __init__(self, video_path: Path, camera: str, subject: str, condition: str):
        super().__init__(video_path, camera, subject, condition)
        self.save_dir = RESOLVE.frames_dir(camera, subject, condition, "single")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def process_frame(self, frame, idx: int):
        save_path = self.save_dir / f"{str(idx).zfill(self.digit)}.jpg"
        cv2.imwrite(str(save_path), frame)


# ============================================================
# 魚眼動画クラス
# ============================================================

class FisheyeVideoProcessor(VideoProcessor):
    """魚眼動画（左右に分割して保存）"""

    def __init__(self, video_path: Path, camera: str, subject: str, condition: str):
        super().__init__(video_path, camera, subject, condition)
        self.left_dir = RESOLVE.frames_dir(camera, subject, condition, "left")
        self.right_dir = RESOLVE.frames_dir(camera, subject, condition, "right")
        self.left_dir.mkdir(parents=True, exist_ok=True)
        self.right_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = self.left_dir.parent

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

def select_videos(video_root: Path) -> list[Path]:
    """data/raw/video 以下を階層ごとに選択（camera -> person -> video file）"""
    EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

    if not video_root.exists():
        print(f"⚠️ 入力ディレクトリがありません: {video_root}")
        return []

    camera_dir = choose_dir("カメラディレクトリを選択してください:", [
        p for p in video_root.iterdir() if p.is_dir()
    ])
    if camera_dir is None:
        print(f"⚠️ {video_root} 配下にサブディレクトリがありません。")
        return []

    person_dir = choose_dir("人物/シナリオディレクトリを選択してください:", [
        p for p in camera_dir.iterdir() if p.is_dir()
    ])
    if person_dir is None:
        print(f"⚠️ {camera_dir} 配下に人物ディレクトリがありません。")
        return []

    videos = sorted(
        p for p in person_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    if not videos:
        print(f"⚠️ {person_dir} に動画が見つかりません。")
        return []

    choices = [questionary.Choice(p.name, value=p) for p in videos]
    selected = questionary.checkbox(
        "処理する動画を選択してください (スペースで選択/解除 -> Enterで決定):",
        choices=choices
    ).ask()
    return selected or []


# ============================================================
# メイン処理
# ============================================================

def main():
    video_root = PATHS.raw / "video"
    target_videos = select_videos(video_root)

    if not target_videos:
        print("キャンセルされました。")
        return

    print(f"\n📹 {len(target_videos)} 本の動画を処理します...\n")

    for video in target_videos:
        try:
            camera, subject, condition = parse_video_info(video, video_root)
            if camera.lower() == "fisheye":
                processor = FisheyeVideoProcessor(video, camera, subject, condition)
            else:
                processor = NormalVideoProcessor(video, camera, subject, condition)
            processor.run()
        except Exception as e:
            print(f"⚠️ {video.name} の処理中にエラー: {e}")

    print("\n✅ すべて完了しました！")


if __name__ == "__main__":
    main()
