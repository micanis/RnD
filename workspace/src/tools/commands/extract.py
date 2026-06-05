import os
import shutil
import sys
from pathlib import Path

import numpy as np
import questionary
import typer
import zarr
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.fisheye import FisheyeCorrector

load_dotenv()

app = typer.Typer(help="Extract perspective-corrected views for each detected person")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[4]))
PROCESSED_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "image"
PROCESSED_DETECT_DIR = PROJECT_ROOT / "data" / "processed" / "detect"
PROCESSED_EXTRACT_DIR = PROJECT_ROOT / "data" / "processed" / "extract"

DEFAULT_FOV = 90


def extract_person_views(
    image_zarr_path: str,
    detect_zarr_path: str,
    output_path: str,
    fov: float = DEFAULT_FOV,
) -> None:
    """
    検出結果に基づいて人物ごとの補正画像を抽出

    Parameters
    ----------
    image_zarr_path : str
        video2img出力 (.zarr.zip)
    detect_zarr_path : str
        detector出力 (.zarr)
    output_path : str
        出力先 (.zarr)
    fov : float
        切り出す視野角（度）
    """
    image_store = zarr.ZipStore(image_zarr_path, mode="r")
    image_root = zarr.group(store=image_store)

    detect_store = zarr.DirectoryStore(detect_zarr_path)
    detect_root = zarr.group(store=detect_store)

    output_store = zarr.DirectoryStore(output_path)
    output_root = zarr.group(store=output_store)

    # 入力タイプ判定
    if "frames" in image_root:
        _process_single(image_root["frames"], detect_root, output_root, fov)
    elif "left" in image_root or "right" in image_root:
        _process_dual(image_root, detect_root, output_root, fov)
    else:
        raise ValueError(f"未対応のZarr構造です: {list(image_root.keys())}")

    typer.echo(f"保存完了: {output_path}")


def _process_single(
    frames: zarr.Array,
    detect_root: zarr.Group,
    output_root: zarr.Group,
    fov: float,
) -> None:
    """perspective_projection用の処理"""
    h, w = frames.shape[1:3]
    corrector = FisheyeCorrector(w, h)

    # 出力サイズは入力解像度ベース（アスペクト比維持）
    out_size = (w, h)

    person_groups = [k for k in detect_root.keys() if k.startswith("person")]

    for person_name in tqdm(person_groups, desc="Extracting persons"):
        person_detect = detect_root[person_name]
        coords = person_detect["coords"][:].astype(np.float32)  # (N, 6): [frame, x1, y1, x2, y2, conf]

        n_frames = len(coords)

        person_out = output_root.create_group(person_name)
        person_frames = person_out.create_dataset(
            "frames",
            shape=(n_frames, out_size[1], out_size[0], 3),
            chunks=(16, out_size[1], out_size[0], 3),
            dtype=np.uint8,
        )

        for i, row in enumerate(tqdm(coords, desc=f"  {person_name}", leave=False)):
            frame_idx = int(row[0])
            bbox = row[1:5]  # [x1, y1, x2, y2]

            frame = np.array(frames[frame_idx])
            view = corrector.extract_view_from_bbox(frame, bbox, out_size, fov)
            person_frames[i] = view

        person_out.attrs["frame_indices"] = coords[:, 0].astype(int).tolist()
        person_out.attrs["total_frames"] = n_frames


def _process_dual(
    image_root: zarr.Group,
    detect_root: zarr.Group,
    output_root: zarr.Group,
    fov: float,
) -> None:
    """dual_fisheye用の処理"""
    for side in ["left", "right"]:
        if side not in image_root:
            continue
        if side not in detect_root:
            typer.echo(f"警告: detect_zarrに'{side}'がありません。スキップします。")
            continue

        frames = image_root[side]
        h, w = frames.shape[1:3]
        corrector = FisheyeCorrector(w, h)

        # 出力サイズは入力解像度ベース
        out_size = (w, h)

        side_detect = detect_root[side]
        side_out = output_root.create_group(side)

        person_groups = [k for k in side_detect.keys() if k.startswith("person")]

        for person_name in tqdm(person_groups, desc=f"[{side}] Extracting"):
            person_detect = side_detect[person_name]
            coords = person_detect["coords"][:].astype(np.float32)

            n_frames = len(coords)

            person_out = side_out.create_group(person_name)
            person_frames = person_out.create_dataset(
                "frames",
                shape=(n_frames, out_size[1], out_size[0], 3),
                chunks=(16, out_size[1], out_size[0], 3),
                dtype=np.uint8,
            )

            for i, row in enumerate(tqdm(coords, desc=f"  {person_name}", leave=False)):
                frame_idx = int(row[0])
                bbox = row[1:5]  # [x1, y1, x2, y2]

                frame = np.array(frames[frame_idx])
                view = corrector.extract_view_from_bbox(frame, bbox, out_size, fov)
                person_frames[i] = view

            person_out.attrs["frame_indices"] = coords[:, 0].astype(int).tolist()
            person_out.attrs["total_frames"] = n_frames


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return

    # カテゴリ選択
    categories = []
    for cat in ["perspective_projection", "dual_fisheye"]:
        if (PROCESSED_IMAGE_DIR / cat).exists() and (
            PROCESSED_DETECT_DIR / cat
        ).exists():
            categories.append(cat)

    if not categories:
        typer.echo("No matching image/detect pairs found")
        raise typer.Exit(1)

    category = questionary.select(
        "Select category:",
        choices=categories,
    ).ask()

    if category is None:
        raise typer.Exit(0)

    # 画像Zarrの選択
    image_dir = PROCESSED_IMAGE_DIR / category
    detect_dir = PROCESSED_DETECT_DIR / category

    image_files = sorted(image_dir.glob("*.zarr.zip"))
    if not image_files:
        typer.echo(f"No image zarr files in {image_dir}")
        raise typer.Exit(1)

    image_choices = [f.name for f in image_files]
    selected_image = questionary.select(
        "Select image zarr:",
        choices=image_choices,
    ).ask()

    if selected_image is None:
        raise typer.Exit(0)

    # 対応するdetect Zarrを探す
    stem = Path(selected_image).stem.replace(".zarr", "")
    detect_files = sorted(detect_dir.glob(f"{stem}*.zarr"))

    if not detect_files:
        typer.echo(f"No detect zarr found for {selected_image}")
        raise typer.Exit(1)

    detect_choices = [f.name for f in detect_files]
    selected_detect = questionary.select(
        "Select detect zarr:",
        choices=detect_choices,
    ).ask()

    if selected_detect is None:
        raise typer.Exit(0)

    # 出力パス
    output_dir = PROCESSED_EXTRACT_DIR / category
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{stem}_extract.zarr"
    output_path = output_dir / output_name

    if output_path.exists():
        overwrite = questionary.confirm(
            f"{output_name} already exists. Overwrite?",
            default=False,
        ).ask()
        if not overwrite:
            raise typer.Exit(0)
        shutil.rmtree(output_path)

    image_path = image_dir / selected_image
    detect_path = detect_dir / selected_detect

    typer.echo(f"\nImage:  {image_path}")
    typer.echo(f"Detect: {detect_path}")
    typer.echo(f"Output: {output_path}\n")

    extract_person_views(str(image_path), str(detect_path), str(output_path))

    typer.echo("\nDone!")


if __name__ == "__main__":
    app()
