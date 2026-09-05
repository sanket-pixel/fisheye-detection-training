# src/data/inspect_woodscape.py
"""
Load WoodScape (rgb_images + box_2d_annotations) into a FiftyOne dataset
for visual inspection, filtered to the 'person' class for our fisheye
person-detection project.
"""
import os
import fiftyone as fo

DATA_ROOT = "data/woodscape"
IMG_DIR = os.path.join(DATA_ROOT, "rgb_images")
ANN_DIR = os.path.join(DATA_ROOT, "box_2d_annotations")

CLASSES = ["vehicles", "person", "bicycle", "traffic_light", "traffic_sign"]


def parse_annotation_file(txt_path, img_width, img_height):
    """Parse one WoodScape box2d .txt file into FiftyOne Detection objects."""
    detections = []
    if not os.path.exists(txt_path):
        return detections

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 6:
                continue  # malformed line, skip (we'll count these later)

            cls_name, cls_idx, xmin, ymin, xmax, ymax = parts
            xmin, ymin, xmax, ymax = map(float, (xmin, ymin, xmax, ymax))

            # FiftyOne expects relative [x, y, w, h] in [0, 1]
            w = (xmax - xmin) / img_width
            h = (ymax - ymin) / img_height
            x = xmin / img_width
            y = ymin / img_height

            detections.append(
                fo.Detection(
                    label=cls_name,
                    bounding_box=[x, y, w, h],
                )
            )
    return detections


def build_dataset(name="woodscape_person", max_samples=None):
    if fo.dataset_exists(name):
        fo.delete_dataset(name)

    dataset = fo.Dataset(name)
    samples = []

    image_files = sorted(os.listdir(IMG_DIR))
    if max_samples:
        image_files = image_files[:max_samples]

    for fname in image_files:
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(IMG_DIR, fname)
        base_name = os.path.splitext(fname)[0]
        ann_path = os.path.join(ANN_DIR, base_name + ".txt")

        sample = fo.Sample(filepath=img_path)

        # Need actual image dims for normalization — read once, lazily
        from PIL import Image
        with Image.open(img_path) as im:
            w, h = im.size

        detections = parse_annotation_file(ann_path, w, h)
        sample["ground_truth"] = fo.Detections(detections=detections)
        samples.append(sample)

    dataset.add_samples(samples)
    return dataset


if __name__ == "__main__":
    # Start small — inspect 300 images first, not the full 8k+
    dataset = build_dataset(max_samples=300)
    print(dataset)

    # Quick sanity stats
    dataset.compute_metadata()
    print(dataset.count_values("ground_truth.detections.label"))

    session = fo.launch_app(dataset)
    session.wait()