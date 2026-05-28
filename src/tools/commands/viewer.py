import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import questionary
import typer
import zarr
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

app = typer.Typer(help="View Zarr image data frame by frame")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "."))
PROCESSED_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "image"
PROCESSED_DETECT_DIR = PROJECT_ROOT / "data" / "processed" / "detect"
PROCESSED_EXTRACT_DIR = PROJECT_ROOT / "data" / "processed" / "extract"
SCREENSHOT_DIR = PROJECT_ROOT / "data" / "screenshots"

DataType = Literal["image", "detect", "extract"]
DetectMode = Literal["overlay", "bbox_only"]


class ZarrViewer:
    def __init__(self):
        self.zoom = 1.0
        self.frame_idx = 0
        self.person_idx = 0
        self.detect_mode: DetectMode = "overlay"
        self.window_name = "Zarr Viewer"

    def view_image(self, zarr_path: str, side: str | None = None) -> None:
        """image.zarr.zipを閲覧"""
        store = zarr.ZipStore(zarr_path, mode="r")
        root = zarr.group(store=store)

        if "frames" in root:
            frames = root["frames"]
            side_label = ""
        elif side and side in root:
            frames = root[side]
            side_label = f"[{side}] "
        else:
            typer.echo("対応するフレームが見つかりません")
            return

        total = len(frames)
        self.frame_idx = 0

        while True:
            frame = np.array(frames[self.frame_idx])
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            title = f"{side_label}Frame {self.frame_idx + 1}/{total}"
            key = self._show_frame(frame_bgr, title)

            if key == ord("q"):
                break
            elif self._is_right_key(key):
                self.frame_idx = min(self.frame_idx + 1, total - 1)
            elif self._is_left_key(key):
                self.frame_idx = max(self.frame_idx - 1, 0)
            elif key == ord("s"):
                self._save_frame(frame_bgr, f"image_{self.frame_idx:04d}")
            elif key == ord("+") or key == ord("="):
                self.zoom = min(self.zoom * 1.2, 5.0)
            elif key == ord("-"):
                self.zoom = max(self.zoom / 1.2, 0.2)

        cv2.destroyAllWindows()

    def view_detect(
        self,
        detect_zarr_path: str,
        image_zarr_path: str | None = None,
        side: str | None = None,
    ) -> None:
        """detect.zarrを閲覧（bbox表示）"""
        detect_store = zarr.DirectoryStore(detect_zarr_path)
        detect_root = zarr.group(store=detect_store)

        image_root = None
        frames = None
        if image_zarr_path:
            image_store = zarr.ZipStore(image_zarr_path, mode="r")
            image_root = zarr.group(store=image_store)

        # side判定
        if side and side in detect_root:
            detect_group = detect_root[side]
            if image_root and side in image_root:
                frames = image_root[side]
            side_label = f"[{side}] "
        elif "left" in detect_root or "right" in detect_root:
            # dual_fisheyeだがsideが指定されていない
            typer.echo("dual_fisheyeの場合はsideを指定してください")
            return
        else:
            detect_group = detect_root
            if image_root and "frames" in image_root:
                frames = image_root["frames"]
            side_label = ""

        # 人物リスト取得
        person_names = sorted([k for k in detect_group.keys() if k.startswith("person")])
        if not person_names:
            typer.echo("検出された人物がいません")
            return

        # all_tracksからフレーム範囲を取得
        if "all_tracks" in detect_group:
            all_tracks = np.array(detect_group["all_tracks"])
            frame_indices = np.unique(all_tracks[:, 1].astype(int))
            total_frames = int(frame_indices.max()) + 1
        else:
            total_frames = 0
            for pn in person_names:
                coords = np.array(detect_group[pn]["coords"])
                total_frames = max(total_frames, int(coords[:, 0].max()) + 1)

        # 画像サイズ取得
        if frames is not None:
            h, w = frames.shape[1:3]
        else:
            # all_tracksからbboxの最大値で推定
            h, w = 960, 1920  # fallback

        self.frame_idx = 0
        self.person_idx = 0

        while True:
            # 現在フレームのbboxを収集
            bboxes = []
            for pn in person_names:
                coords = np.array(detect_group[pn]["coords"])
                mask = coords[:, 0].astype(int) == self.frame_idx
                for row in coords[mask]:
                    bbox = row[1:5]
                    conf = row[5]
                    pid = int(pn.replace("person", ""))
                    bboxes.append((pid, bbox, conf))

            # フレーム描画
            if self.detect_mode == "overlay" and frames is not None:
                frame = np.array(frames[self.frame_idx])
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = np.zeros((h, w, 3), dtype=np.uint8)

            # bbox描画
            for pid, bbox, conf in bboxes:
                x1, y1, x2, y2 = bbox.astype(int)
                color = self._get_color(pid)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{pid} {conf:.2f}"
                cv2.putText(
                    frame_bgr, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

            mode_str = "overlay" if self.detect_mode == "overlay" else "bbox"
            title = f"{side_label}Frame {self.frame_idx + 1}/{total_frames} [{mode_str}]"
            key = self._show_frame(frame_bgr, title)

            if key == ord("q"):
                break
            elif self._is_right_key(key):
                self.frame_idx = min(self.frame_idx + 1, total_frames - 1)
            elif self._is_left_key(key):
                self.frame_idx = max(self.frame_idx - 1, 0)
            elif key == ord("m"):
                self.detect_mode = "bbox_only" if self.detect_mode == "overlay" else "overlay"
            elif key == ord("s"):
                self._save_frame(frame_bgr, f"detect_{self.frame_idx:04d}")
            elif key == ord("+") or key == ord("="):
                self.zoom = min(self.zoom * 1.2, 5.0)
            elif key == ord("-"):
                self.zoom = max(self.zoom / 1.2, 0.2)

        cv2.destroyAllWindows()

    def view_extract(self, zarr_path: str, side: str | None = None) -> None:
        """extract.zarrを閲覧（人物ごとの補正画像）"""
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store)

        # side判定
        if side and side in root:
            extract_group = root[side]
            side_label = f"[{side}] "
        elif "left" in root or "right" in root:
            typer.echo("dual_fisheyeの場合はsideを指定してください")
            return
        else:
            extract_group = root
            side_label = ""

        person_names = sorted([k for k in extract_group.keys() if k.startswith("person")])
        if not person_names:
            typer.echo("抽出された人物がいません")
            return

        self.person_idx = 0
        self.frame_idx = 0

        while True:
            person_name = person_names[self.person_idx]
            person_group = extract_group[person_name]
            frames = person_group["frames"]
            total = len(frames)

            self.frame_idx = min(self.frame_idx, total - 1)

            frame = np.array(frames[self.frame_idx])
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            title = f"{side_label}{person_name} Frame {self.frame_idx + 1}/{total} (↑↓: person)"
            key = self._show_frame(frame_bgr, title)

            if key == ord("q"):
                break
            elif self._is_right_key(key):
                self.frame_idx = min(self.frame_idx + 1, total - 1)
            elif self._is_left_key(key):
                self.frame_idx = max(self.frame_idx - 1, 0)
            elif self._is_up_key(key):
                self.person_idx = (self.person_idx - 1) % len(person_names)
                self.frame_idx = 0
            elif self._is_down_key(key):
                self.person_idx = (self.person_idx + 1) % len(person_names)
                self.frame_idx = 0
            elif key == ord("s"):
                self._save_frame(frame_bgr, f"extract_{person_name}_{self.frame_idx:04d}")
            elif key == ord("+") or key == ord("="):
                self.zoom = min(self.zoom * 1.2, 5.0)
            elif key == ord("-"):
                self.zoom = max(self.zoom / 1.2, 0.2)

        cv2.destroyAllWindows()

    def _is_left_key(self, key: int) -> bool:
        # macOS: 63234, Linux: 81, letter: 'a'
        return key in (63234, 81, 2, ord("a"))

    def _is_right_key(self, key: int) -> bool:
        # macOS: 63235, Linux: 83, letter: 'd'
        return key in (63235, 83, 3, ord("d"))

    def _is_up_key(self, key: int) -> bool:
        # macOS: 63232, Linux: 82, letter: 'w'
        return key in (63232, 82, 0, ord("w"))

    def _is_down_key(self, key: int) -> bool:
        # macOS: 63233, Linux: 84, letter: 'x'
        return key in (63233, 84, 1, ord("x"))

    def _show_frame(self, frame: np.ndarray, title: str) -> int:
        """フレームを表示してキー入力を返す"""
        h, w = frame.shape[:2]
        new_w = int(w * self.zoom)
        new_h = int(h * self.zoom)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, new_w, new_h)
        cv2.imshow(self.window_name, resized)
        cv2.setWindowTitle(self.window_name, f"{title} (zoom: {self.zoom:.1f}x)")
        return cv2.waitKeyEx(0)

    def _save_frame(self, frame: np.ndarray, prefix: str) -> None:
        """フレームを保存"""
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        filepath = SCREENSHOT_DIR / filename
        cv2.imwrite(str(filepath), frame)
        typer.echo(f"保存: {filepath}")

    def _get_color(self, idx: int) -> tuple[int, int, int]:
        """ID別の色を取得"""
        colors = [
            (0, 255, 0),    # green
            (255, 0, 0),    # blue
            (0, 0, 255),    # red
            (255, 255, 0),  # cyan
            (255, 0, 255),  # magenta
            (0, 255, 255),  # yellow
        ]
        return colors[idx % len(colors)]


def get_zarr_files(base_dir: Path, category: str, pattern: str) -> list[Path]:
    """指定パターンのZarrファイルを取得"""
    category_dir = base_dir / category
    if not category_dir.exists():
        return []
    return sorted(category_dir.glob(pattern))


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return

    # データ種類選択
    data_type = questionary.select(
        "Select data type:",
        choices=["image", "detect", "extract"],
    ).ask()

    if data_type is None:
        raise typer.Exit(0)

    # ディレクトリ設定
    if data_type == "image":
        base_dir = PROCESSED_IMAGE_DIR
        pattern = "*.zarr.zip"
    elif data_type == "detect":
        base_dir = PROCESSED_DETECT_DIR
        pattern = "*.zarr"
    else:
        base_dir = PROCESSED_EXTRACT_DIR
        pattern = "*.zarr"

    # カテゴリ選択
    categories = []
    for cat in ["perspective_projection", "dual_fisheye"]:
        if get_zarr_files(base_dir, cat, pattern):
            categories.append(cat)

    if not categories:
        typer.echo(f"No {data_type} files found")
        raise typer.Exit(1)

    category = questionary.select(
        "Select category:",
        choices=categories,
    ).ask()

    if category is None:
        raise typer.Exit(0)

    # ファイル選択
    files = get_zarr_files(base_dir, category, pattern)
    file_choices = [f.name for f in files]

    selected = questionary.select(
        "Select file:",
        choices=file_choices,
    ).ask()

    if selected is None:
        raise typer.Exit(0)

    file_path = base_dir / category / selected

    # dual_fisheyeの場合side選択
    side = None
    if category == "dual_fisheye":
        side = questionary.select(
            "Select side:",
            choices=["left", "right"],
        ).ask()
        if side is None:
            raise typer.Exit(0)

    viewer = ZarrViewer()

    if data_type == "image":
        viewer.view_image(str(file_path), side)

    elif data_type == "detect":
        # 対応するimage.zarrを探す
        stem = Path(selected).stem.replace("_detect", "").replace(".zarr", "")
        image_candidates = list((PROCESSED_IMAGE_DIR / category).glob(f"{stem}*.zarr.zip"))

        image_path = None
        if image_candidates:
            use_image = questionary.confirm(
                "元画像を重畳表示しますか？",
                default=True,
            ).ask()
            if use_image:
                if len(image_candidates) == 1:
                    image_path = str(image_candidates[0])
                else:
                    img_selected = questionary.select(
                        "Select image file:",
                        choices=[f.name for f in image_candidates],
                    ).ask()
                    if img_selected:
                        image_path = str(PROCESSED_IMAGE_DIR / category / img_selected)

        viewer.view_detect(str(file_path), image_path, side)

    else:  # extract
        viewer.view_extract(str(file_path), side)

    typer.echo("Done!")


if __name__ == "__main__":
    app()
