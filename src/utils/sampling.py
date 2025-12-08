import random
from pathlib import Path

def sample_frames(img_dir: Path, n_samples: int) -> list[Path]:
    images = sorted([p for p in img_dir.glob("*.jpg")])
    if len(images) <= n_samples:
        return images
    return random.sample(images, n_samples)