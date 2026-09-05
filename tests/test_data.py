# tests/test_data.py
"""
Data validation tests for the WoodScape person detection dataset.

These run in CI on every PR and are meant to fail loudly if the dataset
on disk is corrupt, incomplete, or inconsistent with the declared splits —
before any training run wastes GPU hours on bad data.
"""
import json
import os

import pytest
from PIL import Image

DATA_ROOT = "data/woodscape"
IMG_DIR = os.path.join(DATA_ROOT, "rgb_images")
ANN_DIR = os.path.join(DATA_ROOT, "box_2d_annotations")
INFO_JSON = os.path.join(DATA_ROOT, "box_2d_annotation_info.json")
SPLIT_DIR = "data/splits"

EXPECTED_FIELDS = 6  # class_name, class_idx, xmin, ymin, xmax, ymax


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def declared_classes():
    with open(INFO_JSON) as f:
        info = json.load(f)
    return set(info["classes"])


@pytest.fixture(scope="module")
def split_files():
    """All image basenames referenced by the split files, keyed by split name."""
    splits = {}
    for name in ("train", "val"):
        path = os.path.join(SPLIT_DIR, f"{name}.txt")
        with open(path) as f:
            splits[name] = [line.strip() for line in f if line.strip()]
    return splits


@pytest.fixture(scope="module")
def all_split_images(split_files):
    return split_files["train"] + split_files["val"]


# ---------- split integrity ----------

def test_splits_do_not_overlap(split_files):
    overlap = set(split_files["train"]) & set(split_files["val"])
    assert not overlap, f"{len(overlap)} images appear in both train and val"


def test_splits_cover_all_images(all_split_images):
    on_disk = {
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    }
    referenced = set(all_split_images)
    assert referenced == on_disk, (
        f"split/disk mismatch: {len(on_disk - referenced)} on disk but unreferenced, "
        f"{len(referenced - on_disk)} referenced but missing from disk"
    )


def test_no_duplicate_entries_within_split(split_files):
    for name, entries in split_files.items():
        assert len(entries) == len(set(entries)), f"duplicate entries in {name}.txt"


# ---------- file existence ----------

def test_every_image_exists(all_split_images):
    missing = [f for f in all_split_images if not os.path.exists(os.path.join(IMG_DIR, f))]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:5]}"


def test_every_image_has_annotation(all_split_images):
    missing = []
    for img in all_split_images:
        ann = os.path.splitext(img)[0] + ".txt"
        if not os.path.exists(os.path.join(ANN_DIR, ann)):
            missing.append(ann)
    assert not missing, f"{len(missing)} annotations missing, e.g. {missing[:5]}"


# ---------- annotation content ----------

def _parse_lines(ann_path):
    with open(ann_path) as f:
        return [line.strip() for line in f if line.strip()]


def test_annotation_lines_well_formed(all_split_images):
    """Every non-empty line has exactly 6 comma-separated fields, coords numeric."""
    bad = []
    for img in all_split_images:
        ann_path = os.path.join(ANN_DIR, os.path.splitext(img)[0] + ".txt")
        for i, line in enumerate(_parse_lines(ann_path)):
            parts = line.split(",")
            if len(parts) != EXPECTED_FIELDS:
                bad.append((img, i, f"expected {EXPECTED_FIELDS} fields, got {len(parts)}"))
                continue
            try:
                [float(p) for p in parts[1:]]
            except ValueError:
                bad.append((img, i, "non-numeric coordinate or class index"))
    assert not bad, f"{len(bad)} malformed lines, e.g. {bad[:5]}"


def test_class_names_are_declared(all_split_images, declared_classes):
    unknown = set()
    for img in all_split_images:
        ann_path = os.path.join(ANN_DIR, os.path.splitext(img)[0] + ".txt")
        for line in _parse_lines(ann_path):
            parts = line.split(",")
            if len(parts) == EXPECTED_FIELDS:
                unknown.add(parts[0]) if parts[0] not in declared_classes else None
    assert not unknown, f"undeclared class names found: {unknown}"


def test_boxes_have_positive_area(all_split_images):
    bad = []
    for img in all_split_images:
        ann_path = os.path.join(ANN_DIR, os.path.splitext(img)[0] + ".txt")
        for i, line in enumerate(_parse_lines(ann_path)):
            parts = line.split(",")
            if len(parts) != EXPECTED_FIELDS:
                continue
            xmin, ymin, xmax, ymax = map(float, parts[2:])
            if xmax <= xmin or ymax <= ymin:
                bad.append((img, i, (xmin, ymin, xmax, ymax)))
    assert not bad, f"{len(bad)} zero/negative-area boxes, e.g. {bad[:5]}"


def test_boxes_within_image_bounds(all_split_images):
    """
    Opens each image to get true dimensions. Slower than the other tests —
    this is the one that catches coordinate-system mistakes.
    """
    bad = []
    for img in all_split_images:
        img_path = os.path.join(IMG_DIR, img)
        ann_path = os.path.join(ANN_DIR, os.path.splitext(img)[0] + ".txt")

        with Image.open(img_path) as im:
            w, h = im.size

        for i, line in enumerate(_parse_lines(ann_path)):
            parts = line.split(",")
            if len(parts) != EXPECTED_FIELDS:
                continue
            xmin, ymin, xmax, ymax = map(float, parts[2:])
            if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
                bad.append((img, i, (xmin, ymin, xmax, ymax), (w, h)))
    assert not bad, f"{len(bad)} out-of-bounds boxes, e.g. {bad[:5]}"


def test_images_are_readable(all_split_images):
    """Catches truncated/corrupt image files that would crash a training run."""
    corrupt = []
    for img in all_split_images:
        try:
            with Image.open(os.path.join(IMG_DIR, img)) as im:
                im.verify()
        except Exception as e:
            corrupt.append((img, str(e)))
    assert not corrupt, f"{len(corrupt)} unreadable images, e.g. {corrupt[:5]}"