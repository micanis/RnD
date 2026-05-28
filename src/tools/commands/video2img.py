import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import questionary
import typer
import zarr
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

app = typer.Typer(help="Convert video files to Zarr ZipStore format")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "."))
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "image"

CHUNK_FRAMES = 16


def get_video_files(category: str) -> list[Path]:
    category_dir = RAW_DIR / category
    if not category_dir.exists():
        return []
    return sorted(list(category_dir.glob("*.mp4")) + list(category_dir.glob("*.MP4")))


def convert_perspective_projection(video_path: Path, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_zarr = Path(tmpdir) / "temp.zarr"
        store = zarr.DirectoryStore(str(tmp_zarr))
        root = zarr.group(store=store, overwrite=True)

        frames = root.create_dataset(
            "frames",
            shape=(total_frames, height, width, 3),
            chunks=(CHUNK_FRAMES, height, width, 3),
            dtype=np.uint8,
            compressor=zarr.Zstd(level=3),
        )

        for i in tqdm(range(total_frames), desc="Reading frames"):
            ret, frame = cap.read()
            if not ret:
                break
            frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        root.attrs.update({
            "source": video_path.name,
            "fps": fps,
            "total_frames": total_frames,
        })

        cap.release()
        store.close()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zarr.ZipStore(str(output_path), mode="w") as zip_store:
            zarr.copy_store(store, zip_store)


def convert_dual_fisheye(video_path: Path, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    half_width = width // 2

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_zarr = Path(tmpdir) / "temp.zarr"
        store = zarr.DirectoryStore(str(tmp_zarr))
        root = zarr.group(store=store, overwrite=True)

        left = root.create_dataset(
            "left",
            shape=(total_frames, height, half_width, 3),
            chunks=(CHUNK_FRAMES, height, half_width, 3),
            dtype=np.uint8,
            compressor=zarr.Zstd(level=3),
        )
        right = root.create_dataset(
            "right",
            shape=(total_frames, height, half_width, 3),
            chunks=(CHUNK_FRAMES, height, half_width, 3),
            dtype=np.uint8,
            compressor=zarr.Zstd(level=3),
        )

        for i in tqdm(range(total_frames), desc="Reading frames"):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            left[i] = frame_rgb[:, :half_width]
            right[i] = frame_rgb[:, half_width:]

        root.attrs.update({
            "source": video_path.name,
            "fps": fps,
            "total_frames": total_frames,
            "frame_index": "temporal index, same across left and right",
        })

        cap.release()
        store.close()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zarr.ZipStore(str(output_path), mode="w") as zip_store:
            zarr.copy_store(store, zip_store)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return

    categories = []
    if get_video_files("perspective_projection"):
        categories.append("perspective_projection")
    if get_video_files("dual_fisheye"):
        categories.append("dual_fisheye")

    if not categories:
        typer.echo("No video files found in data/raw/")
        raise typer.Exit(1)

    category = questionary.select(
        "Select video category:",
        choices=categories,
    ).ask()

    if category is None:
        raise typer.Exit(0)

    videos = get_video_files(category)
    video_choices = [v.name for v in videos]

    selected = questionary.checkbox(
        "Select videos to convert (Space to select, Enter to confirm):",
        choices=video_choices,
    ).ask()

    if not selected:
        typer.echo("No videos selected")
        raise typer.Exit(0)

    convert_fn = convert_dual_fisheye if category == "dual_fisheye" else convert_perspective_projection
    output_dir = PROCESSED_DIR / category

    for video_name in selected:
        video_path = RAW_DIR / category / video_name
        output_path = output_dir / f"{video_path.stem}.zarr.zip"

        if output_path.exists():
            overwrite = questionary.confirm(
                f"{output_path.name} already exists. Overwrite?",
                default=False,
            ).ask()
            if not overwrite:
                continue

        typer.echo(f"\nConverting: {video_name}")
        convert_fn(video_path, output_path)
        typer.echo(f"Saved: {output_path}")

    typer.echo("\nDone!")


if __name__ == "__main__":
    app()
