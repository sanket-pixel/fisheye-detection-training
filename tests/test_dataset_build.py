# tests/test_dataset_build.py
"""
Validation of the built dataset under data/build/<manifest name>/.

The build is what training actually reads, so these tests check the
converted artifacts rather than the raw source. They are also the guard
against a stale build: if the manifest has changed since the build was
produced, training must not proceed.

Requires the build to exist:

    python -m tools.data.build_dataset --manifest <path>
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from engine.data_manifest import load_manifest
from engine.provenance import hash_file

MANIFEST_PATH = Path("data/manifests/person_detection_woodscape_version_1.yaml")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
LABEL_FIELD_COUNT = 5  # class_index, centre_x, centre_y, width, height


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST_PATH)


@pytest.fixture(scope="module")
def build_directory(manifest):
    directory = manifest.build_directory
    if not directory.exists():
        pytest.skip(
            f"{directory} does not exist; run tools.data.build_dataset first"
        )
    return directory


@pytest.fixture(scope="module")
def build_info(build_directory):
    path = build_directory / "build_info.json"
    if not path.exists():
        pytest.fail(f"{path} is missing; the build is incomplete")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def partitions(manifest, build_directory):
    """Frame names per partition, read from the build's split files."""
    result = {}
    for name in manifest.split.partition_names:
        path = build_directory / "splits" / f"{name}.txt"
        result[name] = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return result


@pytest.fixture(scope="module")
def all_frames(partitions):
    return [(name, frame) for name, frames in partitions.items() for frame in frames]


# --------------------------------------------------------------------------
# Staleness — the most important test here
# --------------------------------------------------------------------------


def test_build_matches_current_manifest(build_info):
    """
    The build records the hash of the manifest that produced it. If the
    manifest has since changed, the build is stale and training on it would
    silently use data the manifest no longer describes.
    """
    current = hash_file(MANIFEST_PATH)
    recorded = build_info["manifest_hash"]
    assert current == recorded, (
        "build is stale: the manifest has changed since it was built. "
        "Rebuild with tools.data.build_dataset --force"
    )


def test_build_info_records_provenance(build_info):
    for field in ("manifest_name", "git_commit", "built_at", "classes", "partitions"):
        assert field in build_info, f"build_info.json is missing {field!r}"


def test_build_partition_counts_match_split_files(build_info, partitions):
    for name, frames in partitions.items():
        recorded = build_info["partitions"][name]["frames"]
        assert recorded == len(frames), (
            f"{name}: build_info records {recorded} frames but "
            f"splits/{name}.txt lists {len(frames)}"
        )


# --------------------------------------------------------------------------
# Split integrity
# --------------------------------------------------------------------------


def test_partitions_do_not_overlap(partitions):
    names = sorted(partitions)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = set(partitions[first]) & set(partitions[second])
            assert not overlap, (
                f"{len(overlap)} frames appear in both {first} and {second}"
            )


def test_no_duplicate_frames_within_a_partition(partitions):
    for name, frames in partitions.items():
        assert len(frames) == len(set(frames)), f"duplicate frames in {name}"


def test_partitions_cover_expected_frame_total(manifest, partitions):
    total = sum(len(frames) for frames in partitions.values())
    expected = manifest.statistics.total_frames - len(manifest.selection.excluded_frames)
    assert total == expected, (
        f"partitions hold {total} frames, expected {expected}"
    )


# --------------------------------------------------------------------------
# File presence and integrity
# --------------------------------------------------------------------------


def test_every_frame_has_an_image(build_directory, all_frames):
    missing = [
        (partition, frame)
        for partition, frame in all_frames
        if not (build_directory / "images" / partition / frame).exists()
    ]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:5]}"


def test_image_symlinks_resolve(build_directory, all_frames):
    """
    Images are symlinked into the build rather than copied. A broken link
    means the raw data moved and the build must be regenerated.
    """
    broken = []
    for partition, frame in all_frames:
        path = build_directory / "images" / partition / frame
        if path.is_symlink() and not path.resolve().exists():
            broken.append((partition, frame))
    assert not broken, f"{len(broken)} broken symlinks, e.g. {broken[:5]}"


def test_every_frame_has_a_label_file(build_directory, all_frames):
    """
    An empty label file is valid — it marks a hard negative. A missing one
    means the conversion skipped a frame.
    """
    missing = [
        (partition, frame)
        for partition, frame in all_frames
        if not (build_directory / "labels" / partition / f"{Path(frame).stem}.txt").exists()
    ]
    assert not missing, f"{len(missing)} label files missing, e.g. {missing[:5]}"


def test_images_are_readable(build_directory, all_frames):
    corrupt = []
    for partition, frame in all_frames:
        path = build_directory / "images" / partition / frame
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:  # noqa: BLE001 — report, do not swallow
            corrupt.append((frame, str(error)))
    assert not corrupt, f"{len(corrupt)} unreadable images, e.g. {corrupt[:5]}"


# --------------------------------------------------------------------------
# Label content
# --------------------------------------------------------------------------


def _read_label_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def test_label_lines_are_well_formed(build_directory, all_frames):
    bad = []
    for partition, frame in all_frames:
        path = build_directory / "labels" / partition / f"{Path(frame).stem}.txt"
        for index, line in enumerate(_read_label_lines(path)):
            fields = line.split()
            if len(fields) != LABEL_FIELD_COUNT:
                bad.append((frame, index, f"expected {LABEL_FIELD_COUNT} fields, got {len(fields)}"))
                continue
            try:
                int(fields[0])
                [float(value) for value in fields[1:]]
            except ValueError:
                bad.append((frame, index, "non-numeric field"))
    assert not bad, f"{len(bad)} malformed label lines, e.g. {bad[:5]}"


def test_label_values_are_normalised(build_directory, all_frames):
    """Every coordinate must lie in [0, 1] — the defining property of the format."""
    bad = []
    for partition, frame in all_frames:
        path = build_directory / "labels" / partition / f"{Path(frame).stem}.txt"
        for index, line in enumerate(_read_label_lines(path)):
            fields = line.split()
            if len(fields) != LABEL_FIELD_COUNT:
                continue
            values = [float(value) for value in fields[1:]]
            if any(value < 0.0 or value > 1.0 for value in values):
                bad.append((frame, index, values))
    assert not bad, f"{len(bad)} labels outside [0, 1], e.g. {bad[:5]}"


def test_label_boxes_have_positive_extent(build_directory, all_frames):
    bad = []
    for partition, frame in all_frames:
        path = build_directory / "labels" / partition / f"{Path(frame).stem}.txt"
        for index, line in enumerate(_read_label_lines(path)):
            fields = line.split()
            if len(fields) != LABEL_FIELD_COUNT:
                continue
            _, _, width, height = (float(value) for value in fields[1:])
            if width <= 0.0 or height <= 0.0:
                bad.append((frame, index, (width, height)))
    assert not bad, f"{len(bad)} zero-extent boxes, e.g. {bad[:5]}"


def test_class_indices_are_declared(manifest, build_directory, all_frames):
    valid_indices = set(manifest.classes.mapping)
    unknown = set()
    for partition, frame in all_frames:
        path = build_directory / "labels" / partition / f"{Path(frame).stem}.txt"
        for line in _read_label_lines(path):
            fields = line.split()
            if len(fields) == LABEL_FIELD_COUNT:
                index = int(fields[0])
                if index not in valid_indices:
                    unknown.add(index)
    assert not unknown, f"undeclared class indices in labels: {sorted(unknown)}"


def test_annotation_totals_match_build_info(build_info, build_directory, partitions):
    for name, frames in partitions.items():
        counted = 0
        for frame in frames:
            path = build_directory / "labels" / name / f"{Path(frame).stem}.txt"
            counted += len(_read_label_lines(path))
        recorded = build_info["partitions"][name]["annotations"]
        assert counted == recorded, (
            f"{name}: counted {counted} annotations, build_info records {recorded}"
        )