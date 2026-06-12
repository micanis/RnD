from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np
import zarr


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
CEPDOF_DIR = PROJECT_ROOT / "data" / "raw" / "CEPDOF" / "Lunch1"
REAL_ZARR_PATH = (
    PROJECT_ROOT / "data" / "processed" / "image" / "dual_fisheye" / "test2.zarr.zip"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "images"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_cepdof_images(cepdof_dir: Path) -> list[Path]:
    if not cepdof_dir.exists():
        raise FileNotFoundError(f"CEPDOF Lunch1 directory does not exist: {cepdof_dir}")

    return sorted(
        path
        for path in cepdof_dir.glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_jpg(image_bgr: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError(f"failed to write image: {output_path}")


def export_cepdof_images(
    cepdof_images: Iterable[Path],
    output_dir: Path,
    frame_index: int,
    cepdof_dir: Path,
) -> list[Path]:
    images = list(cepdof_images)
    if len(images) <= frame_index:
        raise FileNotFoundError(
            f"CEPDOF frame index {frame_index} is unavailable; "
            f"found {len(images)} images under {cepdof_dir}"
        )

    source_path = images[frame_index]
    image_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read CEPDOF image: {source_path}")

    output_path = output_dir / f"cepdof_lunch1_{frame_index:04d}.jpg"
    write_jpg(image_bgr, output_path)
    return [output_path]


def read_real_first_frame(zarr_path: Path) -> np.ndarray:
    if not zarr_path.exists():
        raise FileNotFoundError(f"real environment zarr does not exist: {zarr_path}")

    with zarr.ZipStore(str(zarr_path), mode="r") as store:
        root = zarr.group(store=store)
        if "right" not in root:
            raise KeyError(f"'right' dataset does not exist in: {zarr_path}")

        right = root["right"]
        if len(right) == 0:
            raise ValueError(f"'right' dataset has no frames: {zarr_path}")

        return np.asarray(right[0])


def export_real_first_frame(zarr_path: Path, output_dir: Path) -> Path:
    frame_rgb = read_real_first_frame(zarr_path)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    output_path = output_dir / "real_test2_000.jpg"
    write_jpg(frame_bgr, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one CEPDOF/RAPiD image and the first real-environment "
            "right frame as jpg files."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory to write jpg files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cepdof-dir",
        type=Path,
        default=CEPDOF_DIR,
        help=f"directory containing CEPDOF Lunch1 frames (default: {CEPDOF_DIR})",
    )
    parser.add_argument(
        "--real-zarr",
        type=Path,
        default=REAL_ZARR_PATH,
        help=f"real-environment zarr.zip path (default: {REAL_ZARR_PATH})",
    )
    parser.add_argument(
        "--only",
        choices=("all", "cepdof", "real"),
        default="all",
        help="subset to export",
    )
    parser.add_argument(
        "--cepdof-frame-index",
        type=int,
        default=500,
        help="0-based CEPDOF Lunch1 frame index to export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    only: Literal["all", "cepdof", "real"] = args.only

    written: list[Path] = []
    if only in ("all", "cepdof"):
        cepdof_images = find_cepdof_images(args.cepdof_dir)
        written.extend(
            export_cepdof_images(
                cepdof_images=cepdof_images,
                output_dir=args.output_dir,
                frame_index=args.cepdof_frame_index,
                cepdof_dir=args.cepdof_dir,
            )
        )

    if only in ("all", "real"):
        written.append(export_real_first_frame(args.real_zarr, args.output_dir))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
