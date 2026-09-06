# src/data/convert_to_yolo.py
"""
Convert WoodScape box2d annotations -> YOLO format, single-class (person).

WoodScape:  class_name,class_idx,xmin,ymin,xmax,ymax   (absolute pixels, corners)
YOLO:       class_idx x_center y_center width height    (normalized [0,1], center+size)

Builds the directory layout Ultralytics expects:
    data/yolo/
      images/{train,val}/*.png
      labels/{train,val}/*.txt

Images are symlinked, not copied — no need to duplicate 14GB on disk.
Person-free images still get an (empty) label file: that is how YOLO
represents a valid hard-negative frame, and dropping them would silently
change the dataset defined in the manifest.
"""
import os
from pathlib import Path

from PIL import Image

DATA_ROOT = Path("data/woodscape")
IMG_DIR = DATA_ROOT / "rgb_images"
ANN_DIR = DATA_ROOT / "box_2d_annotations"
SPLIT_DIR = Path("data/splits")
OUT_ROOT = Path("data/yolo")

TARGET_CLASS = "person"
TARGET_CLASS_IDX = 0  # single-class dataset -> person becomes class 0

EXPECTED_FIELDS = 6


def convert_box(xmin, ymin, xmax, ymax, img_w, img_h):
    """Corner pixels -> normalized center format, clamped to [0, 1]."""
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    # Clamp defensively: boxes sitting exactly on the image edge can round
    # a hair outside [0,1] and Ultralytics will reject the label file.
    clamp = lambda v: max(0.0, min(1.0, v))
    return clamp(x_center), clamp(y_center), clamp(width), clamp(height)


def convert_split(split_name):
    img_out = OUT_ROOT / "images" / split_name
    lbl_out = OUT_ROOT / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_DIR / f"{split_name}.txt") as f:
        images = [line.strip() for line in f if line.strip()]

    n_boxes = 0
    n_empty = 0

    for img_name in images:
        src_img = (IMG_DIR / img_name).resolve()
        dst_img = img_out / img_name

        # Symlink instead of copy — saves ~14GB of duplication
        if not dst_img.exists():
            dst_img.symlink_to(src_img)

        with Image.open(src_img) as im:
            img_w, img_h = im.size

        ann_path = ANN_DIR / (Path(img_name).stem + ".txt")
        yolo_lines = []

        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != EXPECTED_FIELDS:
                    continue
                if parts[0] != TARGET_CLASS:
                    continue  # single-class: ignore everything but person

                xmin, ymin, xmax, ymax = map(float, parts[2:])
                xc, yc, w, h = convert_box(xmin, ymin, xmax, ymax, img_w, img_h)

                if w <= 0 or h <= 0:
                    continue  # degenerate after clamping

                yolo_lines.append(f"{TARGET_CLASS_IDX} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        n_boxes += len(yolo_lines)
        if not yolo_lines:
            n_empty += 1

        # Always write the label file, even when empty (= hard negative)
        with open(lbl_out / (Path(img_name).stem + ".txt"), "w") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    print(
        f"{split_name}: {len(images)} images, {n_boxes} person boxes, "
        f"{n_empty} images with no person ({n_empty / len(images):.1%})"
    )


def main():
    for split in ("train", "val"):
        convert_split(split)
    print(f"\nWrote YOLO dataset to {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()