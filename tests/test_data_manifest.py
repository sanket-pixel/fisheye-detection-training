# tests/test_data_manifest.py
"""
Tests for manifest loading and validation.

The manifest is the single source of truth for a dataset, so a malformed
one must fail loudly at load time rather than silently producing a wrong
build. These tests construct deliberately broken manifests and assert that
each is rejected.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from engine.data_manifest import ManifestError, load_manifest

REAL_MANIFEST = Path("data/manifests/person_detection_woodscape_version_1.yaml")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def valid_manifest_dictionary() -> dict:
    """A minimal but complete manifest, used as the base for mutation tests."""
    return {
        "schema_version": 1,
        "identity": {
            "name": "test_dataset",
            "description": "Fixture manifest for validation tests.",
            "created": "2026-01-01",
            "author": "test",
        },
        "task": {
            "type": "object_detection",
            "modality": "monocular_camera",
            "camera_model": "fisheye",
        },
        "source": {
            "dataset": "woodscape",
            "location": "data/raw/woodscape",
            "images": "rgb_images",
            "annotations": "box_2d_annotations",
            "annotation_format": "woodscape_box_2d",
            "license": "non_commercial_research",
            "reference": None,
        },
        "classes": {
            "mapping": {0: "person"},
            "source_classes": ["person"],
            "ignored_source_classes": ["vehicles"],
        },
        "selection": {
            "include_frames_without_annotations": True,
            "minimum_annotation_area_pixels": None,
            "maximum_annotations_per_frame": None,
            "excluded_frames": [],
        },
        "split": {
            "method": "random_stratified",
            "stratify_by": "camera_position",
            "seed": 42,
            "fractions": {"train": 0.9, "validation": 0.1},
            "holdout": {"description": "none", "location": None},
        },
        "statistics": {
            "total_frames": 100,
            "frames_with_annotations": 70,
            "frames_without_annotations": 30,
            "total_annotations": 200,
            "frames_by_camera_position": {
                "front": 25,
                "mirror_left": 25,
                "mirror_right": 25,
                "rear": 25,
            },
        },
    }


@pytest.fixture
def write_manifest(tmp_path):
    """Write a manifest dictionary to disk and return its path."""

    def _write(dictionary: dict, name: str = "manifest.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(dictionary, sort_keys=False))
        return path

    return _write


# --------------------------------------------------------------------------
# The real manifest
# --------------------------------------------------------------------------


def test_project_manifest_loads():
    manifest = load_manifest(REAL_MANIFEST)
    assert manifest.identity.name == "person_detection_woodscape_version_1"
    assert manifest.classes.names == ["person"]
    assert manifest.split.partition_names == ["train", "validation"]


def test_build_directory_is_namespaced_by_manifest_name():
    manifest = load_manifest(REAL_MANIFEST)
    assert manifest.build_directory == Path("data/build") / manifest.identity.name


# --------------------------------------------------------------------------
# Happy path on the fixture
# --------------------------------------------------------------------------


def test_valid_manifest_loads(valid_manifest_dictionary, write_manifest):
    manifest = load_manifest(write_manifest(valid_manifest_dictionary))
    assert manifest.classes.count == 1
    assert manifest.classes.index_of("person") == 0
    assert manifest.source.images_directory == Path("data/raw/woodscape/rgb_images")


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "does_not_exist.yaml")


# --------------------------------------------------------------------------
# Structural validation
# --------------------------------------------------------------------------


def test_missing_schema_version_is_rejected(valid_manifest_dictionary, write_manifest):
    broken = copy.deepcopy(valid_manifest_dictionary)
    del broken["schema_version"]
    with pytest.raises(ManifestError, match="schema_version"):
        load_manifest(write_manifest(broken))


def test_unsupported_schema_version_is_rejected(
    valid_manifest_dictionary, write_manifest
):
    broken = copy.deepcopy(valid_manifest_dictionary)
    broken["schema_version"] = 99
    with pytest.raises(ManifestError, match="not supported"):
        load_manifest(write_manifest(broken))


@pytest.mark.parametrize(
    "section",
    ["identity", "task", "source", "classes", "selection", "split", "statistics"],
)
def test_missing_section_is_rejected(
    valid_manifest_dictionary, write_manifest, section
):
    broken = copy.deepcopy(valid_manifest_dictionary)
    del broken[section]
    with pytest.raises(ManifestError, match=f"missing section '{section}'"):
        load_manifest(write_manifest(broken))


def test_omitted_key_is_rejected_rather_than_defaulted(
    valid_manifest_dictionary, write_manifest
):
    """
    An absent value must be written as null. Omitting the key entirely is an
    error, so that a typo cannot silently fall back to a default.
    """
    broken = copy.deepcopy(valid_manifest_dictionary)
    del broken["selection"]["minimum_annotation_area_pixels"]
    with pytest.raises(ManifestError, match="minimum_annotation_area_pixels"):
        load_manifest(write_manifest(broken))


# --------------------------------------------------------------------------
# Class validation
# --------------------------------------------------------------------------


def test_empty_class_mapping_is_rejected(valid_manifest_dictionary, write_manifest):
    broken = copy.deepcopy(valid_manifest_dictionary)
    broken["classes"]["mapping"] = {}
    with pytest.raises(ManifestError, match="must not be empty"):
        load_manifest(write_manifest(broken))


def test_non_contiguous_class_indices_are_rejected(
    valid_manifest_dictionary, write_manifest
):
    broken = copy.deepcopy(valid_manifest_dictionary)
    broken["classes"]["mapping"] = {0: "person", 2: "vehicle"}
    with pytest.raises(ManifestError, match="contiguous"):
        load_manifest(write_manifest(broken))


def test_class_in_both_kept_and_ignored_is_rejected(
    valid_manifest_dictionary, write_manifest
):
    broken = copy.deepcopy(valid_manifest_dictionary)
    broken["classes"]["ignored_source_classes"] = ["person"]
    with pytest.raises(ManifestError, match="both"):
        load_manifest(write_manifest(broken))


def test_unknown_class_lookup_raises(valid_manifest_dictionary, write_manifest):
    manifest = load_manifest(write_manifest(valid_manifest_dictionary))
    with pytest.raises(KeyError):
        manifest.classes.index_of("bicycle")


# --------------------------------------------------------------------------
# Split validation
# --------------------------------------------------------------------------


def test_fractions_not_summing_to_one_are_rejected(
    valid_manifest_dictionary, write_manifest
):
    broken = copy.deepcopy(valid_manifest_dictionary)
    broken["split"]["fractions"] = {"train": 0.8, "validation": 0.1}
    with pytest.raises(ManifestError, match="sum to 1.0"):
        load_manifest(write_manifest(broken))


def test_zero_fraction_is_rejected(valid_manifest_dictionary, write_manifest):
    broken = copy.deepcopy(valid_manifest_dictionary)
    broken["split"]["fractions"] = {"train": 1.0, "validation": 0.0}
    with pytest.raises(ManifestError, match="must be in"):
        load_manifest(write_manifest(broken))


# --------------------------------------------------------------------------
# Statistics validation
# --------------------------------------------------------------------------


def test_inconsistent_frame_counts_are_rejected(
    valid_manifest_dictionary, write_manifest
):
    broken = copy.deepcopy(valid_manifest_dictionary)
    broken["statistics"]["frames_with_annotations"] = 60  # 60 + 30 != 100
    with pytest.raises(ManifestError, match="total_frames"):
        load_manifest(write_manifest(broken))


def test_camera_position_counts_must_sum_to_total(
    valid_manifest_dictionary, write_manifest
):
    broken = copy.deepcopy(valid_manifest_dictionary)
    broken["statistics"]["frames_by_camera_position"]["front"] = 10  # sums to 85
    with pytest.raises(ManifestError, match="frames_by_camera_position"):
        load_manifest(write_manifest(broken))