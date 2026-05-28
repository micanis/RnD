import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import questionary
import typer
import zarr
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.detector import PersonDetector

load_dotenv()

app = typer.Typer(help="Detect and track persons in Zarr video data")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "."))
PROCESSED_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "image"
PROCESSED_DETECT_DIR = PROJECT_ROOT / "data" / "processed" / "detect"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "yolov8m.pt"

SideOption = Literal["left", "right", "both", "frames"]


class ZarrTrackingSystem:
    def __init__(self, model_path: str = "yolov8m.pt"):
        self.detector = PersonDetector(model_path)

    def _detect_input_type(self, root: zarr.Group) -> SideOption:
        """入力Zarrのキーから映像タイプを自動判別する"""
        keys = set(root.keys())
        if "frames" in keys:
            return "frames"
        elif "left" in keys or "right" in keys:
            return (
                "both"
                if ("left" in keys and "right" in keys)
                else ("left" if "left" in keys else "right")
            )
        else:
            raise ValueError(f"未対応のZarr構造です。検出されたキー: {keys}")

    def _track_side(self, frames_array: zarr.Array, side_label: str) -> dict:
        """1つのside（またはframes）に対してトラッキングを実行し、raw_tracksを返す"""
        raw_tracks = defaultdict(list)
        total = len(frames_array)

        for i in tqdm(range(total), desc=f"[{side_label}] Tracking"):
            frame = np.array(frames_array[i])
            persons = self.detector.track(frame)

            for person in persons:
                # [フレーム番号, x1, y1, x2, y2, スコア]
                raw_tracks[person.track_id].append([i, *person.bbox, person.confidence])

        return raw_tracks

    def _write_tracks(self, root: zarr.Group, raw_tracks: dict, group_prefix: str = ""):
        """
        raw_tracks を Zarr に書き込む。
        group_prefix が空なら root 直下に personXX/ を作成。
        group_prefix が 'left' 等なら left/personXX/ に作成。
        """
        parent = root if not group_prefix else root.require_group(group_prefix)

        combined_list = []
        for obj_id, track_data in raw_tracks.items():
            group_name = f"person{obj_id:02d}"
            person_group = parent.create_group(group_name)

            data_array = np.array(track_data, dtype=np.float32)
            person_group.create_dataset(
                "coords",
                data=data_array,
                chunks=(100, 6),
                dtype=np.float32,
            )
            person_group.attrs["total_frames"] = len(track_data)

            for row in track_data:
                combined_list.append([obj_id, *row])

        if combined_list:
            parent.create_dataset(
                "all_tracks",
                data=np.array(combined_list, dtype=np.float32),
                chunks=(500, 7),
            )

    def run_tracking(
        self,
        input_zarr_path: str,
        side: SideOption | None = None,
    ) -> dict[str, dict]:
        """
        入力Zarrを読み込んでトラッキングを実行する。

        Parameters
        ----------
        input_zarr_path : str
            入力 .zarr.zip のパス。
        side : {"left", "right", "both", "frames"} | None
            処理するside。None の場合は入力構造から自動判別。
            - dual_fisheye で片側だけ処理したい場合に "left" / "right" を指定。
            - "both" または None（dual_fisheye検出時）は両側を処理。

        Returns
        -------
        dict[str, dict]
            {"left": raw_tracks, "right": raw_tracks} または {"frames": raw_tracks}
        """
        store = zarr.ZipStore(input_zarr_path, mode="r")
        root = zarr.group(store=store)

        detected_type = self._detect_input_type(root)

        # sideの解決: 指定がなければ自動判別結果を使う
        resolved_side = side if side is not None else detected_type

        # perspective_projection
        if resolved_side == "frames":
            raw = self._track_side(root["frames"], "frames")
            return {"frames": raw}

        # dual_fisheye
        sides_to_process = (
            ["left", "right"] if resolved_side == "both" else [resolved_side]
        )
        results = {}
        for s in sides_to_process:
            if s not in root:
                typer.echo(f"警告: '{s}' キーが見つかりません。スキップします。")
                continue
            # persist=True のIDはmodel内部に保持されるため、sideをまたぐ前にリセット
            self.detector.reset()
            results[s] = self._track_side(root[s], s)

        return results

    def save_to_zarr(self, raw_results: dict[str, dict], output_path: str):
        """
        run_tracking の戻り値を Zarr に保存する。

        出力構造:
          perspective_projection の場合:
            output.zarr/
            ├── person01/coords, attrs
            ├── ...
            └── all_tracks

          dual_fisheye (both) の場合:
            output.zarr/
            ├── left/
            │   ├── person01/coords, attrs
            │   └── all_tracks
            └── right/
                ├── person01/coords, attrs
                └── all_tracks

          dual_fisheye (片側) の場合:
            output.zarr/
            ├── person01/coords, attrs
            └── all_tracks
        """
        store = zarr.DirectoryStore(output_path)
        root = zarr.group(store=store)

        keys = list(raw_results.keys())

        if keys == ["frames"]:
            # perspective_projection: root直下に展開
            self._write_tracks(root, raw_results["frames"], group_prefix="")
        elif len(keys) > 1:
            # dual_fisheye both: left/ right/ のサブグループに分ける
            for s, raw_tracks in raw_results.items():
                self._write_tracks(root, raw_tracks, group_prefix=s)
        else:
            # dual_fisheye 片側: root直下に展開
            self._write_tracks(root, raw_results[keys[0]], group_prefix="")

        typer.echo(f"保存完了: {output_path}")

    def process(
        self, input_zarr_path: str, output_path: str, side: SideOption | None = None
    ):
        """run_tracking + save_to_zarr をまとめて実行するショートカット"""
        raw_results = self.run_tracking(input_zarr_path, side=side)
        self.save_to_zarr(raw_results, output_path)


def get_zarr_files(category: str) -> list[Path]:
    """指定カテゴリのZarrファイル一覧を取得"""
    category_dir = PROCESSED_IMAGE_DIR / category
    if not category_dir.exists():
        return []
    return sorted(category_dir.glob("*.zarr.zip"))


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return

    categories = []
    if get_zarr_files("perspective_projection"):
        categories.append("perspective_projection")
    if get_zarr_files("dual_fisheye"):
        categories.append("dual_fisheye")

    if not categories:
        typer.echo("No zarr files found in data/processed/image/")
        raise typer.Exit(1)

    category = questionary.select(
        "Select video category:",
        choices=categories,
    ).ask()

    if category is None:
        raise typer.Exit(0)

    zarr_files = get_zarr_files(category)
    file_choices = [f.name for f in zarr_files]

    selected = questionary.checkbox(
        "Select zarr files to process (Space to select, Enter to confirm):",
        choices=file_choices,
    ).ask()

    if not selected:
        typer.echo("No files selected")
        raise typer.Exit(0)

    model_path = str(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else "yolov8n.pt"
    system = ZarrTrackingSystem(model_path=model_path)

    output_dir = PROCESSED_DETECT_DIR / category
    output_dir.mkdir(parents=True, exist_ok=True)

    for zarr_name in selected:
        input_path = PROCESSED_IMAGE_DIR / category / zarr_name
        output_path = output_dir / f"{Path(zarr_name).stem.replace('.zarr', '')}_detect.zarr"

        if output_path.exists():
            overwrite = questionary.confirm(
                f"{output_path.name} already exists. Overwrite?",
                default=False,
            ).ask()
            if not overwrite:
                continue
            shutil.rmtree(output_path)

        typer.echo(f"\nProcessing: {zarr_name}")
        system.process(str(input_path), str(output_path))

    typer.echo("\nDone!")


if __name__ == "__main__":
    app()
