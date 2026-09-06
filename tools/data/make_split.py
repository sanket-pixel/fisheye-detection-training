# src/data/make_split.py
"""
Generate reproducible train/val splits for the WoodScape person detection dataset.

Stratified by camera (FV/MVL/MVR/RV) so each split holds a representative mix
of viewpoints and distortion characteristics. Seeded for reproducibility.

Outputs newline-delimited image basenames to data/splits/{train,val}.txt
"""
import os
import random
from collections import defaultdict

DATA_ROOT = "data/woodscape"
IMG_DIR = os.path.join(DATA_ROOT, "rgb_images")
SPLIT_DIR = "data/splits"

SEED = 42
VAL_FRACTION = 0.1


def camera_of(filename: str) -> str:
    """Extract camera id from e.g. '00123_MVL.png' -> 'MVL'."""
    stem = os.path.splitext(filename)[0]
    return stem.split("_")[-1]


def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    images = sorted(
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    # Group by camera for stratification
    by_camera = defaultdict(list)
    for f in images:
        by_camera[camera_of(f)].append(f)

    train, val = [], []
    for cam, files in sorted(by_camera.items()):
        files = sorted(files)          # deterministic starting order
        rng.shuffle(files)             # seeded shuffle
        n_val = round(len(files) * VAL_FRACTION)
        val.extend(files[:n_val])
        train.extend(files[n_val:])
        print(f"{cam}: {len(files)} total -> {len(files) - n_val} train, {n_val} val")

    train.sort()
    val.sort()

    # Sanity: no overlap, nothing lost
    assert not set(train) & set(val), "train/val overlap detected"
    assert len(train) + len(val) == len(images), "images lost during split"

    for name, split in (("train", train), ("val", val)):
        path = os.path.join(SPLIT_DIR, f"{name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(split) + "\n")
        print(f"wrote {len(split)} entries -> {path}")


if __name__ == "__main__":
    main()