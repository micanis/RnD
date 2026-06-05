import os
import sys
from pathlib import Path

import cv2
import zarr
from PIL import Image
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from tools.commands.video2img import PROJECT_ROOT, convert_dual_fisheye  # noqa: E402


def run_detection():
    """
    Converts a video to Zarr, runs object detection on the first frame,
    and saves the result using YOLOv11.
    """
    # Define paths
    video_path = PROJECT_ROOT / "data" / "raw" / "dual_fisheye" / "test2.MP4"
    temp_zarr_path = PROJECT_ROOT / "tmp" / "test_yolo.zarr.zip"
    output_image_path = PROJECT_ROOT / "tmp" / "detected_by_yolo.jpg"

    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        print("Please make sure the video file exists.")
        return

    # Ensure tmp directory exists
    (PROJECT_ROOT / "tmp").mkdir(exist_ok=True)

    try:
        # Convert video to Zarr format
        print(f"Converting {video_path.name} to Zarr format...")
        convert_dual_fisheye(video_path, temp_zarr_path)
        print(f"Saved temporary Zarr file to {temp_zarr_path}")

        # Open the Zarr archive
        store = zarr.ZipStore(str(temp_zarr_path), mode="r")
        root = zarr.group(store=store)

        # Get the first frame from the 'left' dataset for dual fisheye
        if "left" in root:
            first_frame_np = root["right"][0]
        else:
            print("Error: Could not find 'left' dataset in the Zarr archive.")
            return

        # The frame is an RGB numpy array. Convert to PIL Image.
        image = Image.fromarray(first_frame_np)

        # Load YOLOv11 model
        print("Loading YOLOv11 model (yolov11m.pt)...")
        model = YOLO("yolov11m.pt")  # Load a YOLOv11m model
        print("Model loaded.")

        # Perform detection
        print("Running object detection...")
        # YOLO predict method can take a PIL Image or numpy array
        # It returns a list of Results objects
        results = model.predict(image, conf=0.25)  # conf is confidence threshold
        print("Detection finished.")

        # Process results and draw bounding boxes
        # The plot() method of Results object can draw directly on the image
        for r in results:
            # r.plot() returns a numpy array with detections drawn
            im_bgr = r.plot()
            # Convert back to RGB for PIL if needed, but cv2.imwrite expects BGR
            # So we can directly save im_bgr
            cv2.imwrite(str(output_image_path), im_bgr)
            print(f"Successfully saved detection result to {output_image_path}")
            break  # Only process the first result for now

    finally:
        # Clean up the temporary Zarr file
        if temp_zarr_path.exists():
            os.remove(temp_zarr_path)
            print(f"Removed temporary file: {temp_zarr_path}")


if __name__ == "__main__":
    run_detection()
