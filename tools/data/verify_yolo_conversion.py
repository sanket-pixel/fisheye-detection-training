# src/data/verify_yolo_conversion.py
"""
Round-trip check: load the CONVERTED YOLO labels (not the originals) back
into FiftyOne and render them. If the conversion swapped an axis, botched
normalization, or shifted center-vs-corner, it will be immediately visible
here rather than after hours of training on garbage.
"""
import os
from pathlib import Path

import fiftyone as fo
from PIL import Image

YOLO_ROOT = Path("data/yolo")
SPLIT = "val"  # smaller, faster to eyeball

IMG_DIR = YOLO_ROOT / "images" / SPLIT
LBL_DIR = YOLO_ROOT / "labels" / SPLIT


def load_yolo_labels(lbl_path):
    """YOLO normalized center format -> FiftyOne relative top-left format."""
    detections = []
    if not lbl_path.exists():
        return detections

    with open(lbl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls_idx, xc, yc, w, h = line.split()
            xc, yc, w, h = map(float, (xc, yc, w, h))

            # FiftyOne wants [top-left-x, top-left-y, width, height], relative
            x = xc - w / 2
            y = yc - h / 2

            detections.append(
                fo.Detection(label="person", bounding_box=[x, y, w, h])
            )
    return detections


def main(max_samples=200):
    name = f"woodscape_yolo_{SPLIT}_verify"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)

    dataset = fo.Dataset(name)
    samples = []

    images = sorted(os.listdir(IMG_DIR))[:max_samples]
    total_boxes = 0

    for img_name in images:
        img_path = IMG_DIR / img_name
        lbl_path = LBL_DIR / (Path(img_name).stem + ".txt")

        dets = load_yolo_labels(lbl_path)
        total_boxes += len(dets)

        sample = fo.Sample(filepath=str(img_path.resolve()))
        sample["converted_labels"] = fo.Detections(detections=dets)
        samples.append(sample)

    dataset.add_samples(samples)
    print(f"loaded {len(samples)} samples, {total_boxes} boxes")

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()