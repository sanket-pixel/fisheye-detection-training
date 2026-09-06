# engine/data_manifest.py
"""
Loading and validation for dataset manifests.

A manifest is the single source of truth for one dataset: what it contains,
where it comes from, how it is split, and what the data on disk should look
like. Everything under data/build/ is derived from it.

This module is deliberately free of task-specific knowledge. It validates
structure and provides typed access; interpreting a particular source format
(WoodScape, COCO, an internal recording session) belongs in src/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_SCHEMA_VERSIONS = {1}

# Every key that must be present in a valid manifest. Values may be null,
# but the key itself is never omitted — this keeps manifests diffable and
# means a typo produces a loud failure instead of a silent default.
REQUIRED_STRUCTURE: dict[str, list[str]] = {
    "identity": ["name", "description", "created", "author"],
    "task": ["type", "modality", "camera_model"],
    "source": [
        "dataset",
        "location",
        "images",
        "annotations",
        "annotation_format",
        "license",
        "reference",
    ],
    "classes": ["mapping", "source_classes", "ignored_source_classes"],
    "selection": [
        "include_frames_without_annotations",
        "minimum_annotation_area_pixels",
        "maximum_annotations_per_frame",
        "excluded_frames",
    ],
    "split": ["method", "stratify_by", "seed", "fractions", "holdout"],
    "statistics": [
        "total_frames",
        "frames_with_annotations",
        "frames_without_annotations",
        "total_annotations",
        "frames_by_camera_position",
    ],
}


class ManifestError(ValueError):
    """Raised when a manifest is structurally invalid."""


# --------------------------------------------------------------------------
# Typed views over the manifest sections
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Identity:
    name: str
    description: str
    created: str
    author: str


@dataclass(frozen=True)
class Task:
    type: str
    modality: str
    camera_model: str


@dataclass(frozen=True)
class Source:
    dataset: str
    location: Path
    images: str
    annotations: str
    annotation_format: str
    license: str
    reference: str | None

    @property
    def images_directory(self) -> Path:
        return self.location / self.images

    @property
    def annotations_directory(self) -> Path:
        return self.location / self.annotations


@dataclass(frozen=True)
class Classes:
    mapping: dict[int, str]
    source_classes: list[str]
    ignored_source_classes: list[str]

    @property
    def count(self) -> int:
        return len(self.mapping)

    @property
    def names(self) -> list[str]:
        """Class names ordered by index."""
        return [self.mapping[index] for index in sorted(self.mapping)]

    def index_of(self, name: str) -> int:
        for index, class_name in self.mapping.items():
            if class_name == name:
                return index
        raise KeyError(f"class {name!r} is not in this manifest's mapping")


@dataclass(frozen=True)
class Selection:
    include_frames_without_annotations: bool
    minimum_annotation_area_pixels: float | None
    maximum_annotations_per_frame: int | None
    excluded_frames: list[str]


@dataclass(frozen=True)
class Split:
    method: str
    stratify_by: str | None
    seed: int
    fractions: dict[str, float]
    holdout: dict[str, Any]

    @property
    def partition_names(self) -> list[str]:
        return sorted(self.fractions)


@dataclass(frozen=True)
class Statistics:
    total_frames: int
    frames_with_annotations: int
    frames_without_annotations: int
    total_annotations: int
    frames_by_camera_position: dict[str, int]


@dataclass(frozen=True)
class DataManifest:
    """A validated dataset manifest."""

    path: Path
    schema_version: int
    identity: Identity
    task: Task
    source: Source
    classes: Classes
    selection: Selection
    split: Split
    statistics: Statistics
    raw: dict[str, Any] = field(repr=False)

    @property
    def name(self) -> str:
        return self.identity.name

    @property
    def build_directory(self) -> Path:
        """Where derived artifacts for this manifest live."""
        return Path("data/build") / self.identity.name

    def __str__(self) -> str:
        return (
            f"{self.identity.name} "
            f"({self.task.type}, {self.classes.count} class(es), "
            f"{self.statistics.total_frames} frames)"
        )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _require_sections(raw: dict[str, Any], path: Path) -> None:
    if "schema_version" not in raw:
        raise ManifestError(f"{path}: missing 'schema_version'")

    version = raw["schema_version"]
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ManifestError(
            f"{path}: schema_version {version} is not supported "
            f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
        )

    for section, keys in REQUIRED_STRUCTURE.items():
        if section not in raw:
            raise ManifestError(f"{path}: missing section '{section}'")
        if not isinstance(raw[section], dict):
            raise ManifestError(f"{path}: section '{section}' must be a mapping")

        missing = [key for key in keys if key not in raw[section]]
        if missing:
            raise ManifestError(
                f"{path}: section '{section}' is missing required key(s): "
                f"{', '.join(missing)}. Write an explicit null rather than "
                f"omitting a key."
            )


def _validate_classes(classes: Classes, path: Path) -> None:
    if not classes.mapping:
        raise ManifestError(f"{path}: classes.mapping must not be empty")

    indices = sorted(classes.mapping)
    if indices != list(range(len(indices))):
        raise ManifestError(
            f"{path}: classes.mapping indices must be contiguous from zero, "
            f"got {indices}"
        )

    overlap = set(classes.source_classes) & set(classes.ignored_source_classes)
    if overlap:
        raise ManifestError(
            f"{path}: class(es) appear in both source_classes and "
            f"ignored_source_classes: {sorted(overlap)}"
        )


def _validate_split(split: Split, path: Path) -> None:
    if not split.fractions:
        raise ManifestError(f"{path}: split.fractions must not be empty")

    total = sum(split.fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ManifestError(
            f"{path}: split.fractions must sum to 1.0, got {total} "
            f"({split.fractions})"
        )

    for name, fraction in split.fractions.items():
        if not 0.0 < fraction < 1.0:
            raise ManifestError(
                f"{path}: split fraction '{name}' must be in (0, 1), got {fraction}"
            )


def _validate_statistics(statistics: Statistics, path: Path) -> None:
    annotated = statistics.frames_with_annotations
    unannotated = statistics.frames_without_annotations
    if annotated + unannotated != statistics.total_frames:
        raise ManifestError(
            f"{path}: frames_with_annotations ({annotated}) + "
            f"frames_without_annotations ({unannotated}) != "
            f"total_frames ({statistics.total_frames})"
        )

    by_camera = statistics.frames_by_camera_position
    if by_camera and sum(by_camera.values()) != statistics.total_frames:
        raise ManifestError(
            f"{path}: frames_by_camera_position sums to "
            f"{sum(by_camera.values())}, expected {statistics.total_frames}"
        )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_manifest(path: str | Path) -> DataManifest:
    """Load and validate a manifest. Raises ManifestError on any problem."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")

    with open(path) as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ManifestError(f"{path}: manifest must be a YAML mapping")

    _require_sections(raw, path)

    identity = Identity(**raw["identity"])
    task = Task(**raw["task"])

    source_section = dict(raw["source"])
    source_section["location"] = Path(source_section["location"])
    source = Source(**source_section)

    classes_section = dict(raw["classes"])
    classes_section["mapping"] = {
        int(index): name for index, name in classes_section["mapping"].items()
    }
    classes = Classes(**classes_section)

    selection = Selection(**raw["selection"])
    split = Split(**raw["split"])
    statistics = Statistics(**raw["statistics"])

    _validate_classes(classes, path)
    _validate_split(split, path)
    _validate_statistics(statistics, path)

    return DataManifest(
        path=path,
        schema_version=raw["schema_version"],
        identity=identity,
        task=task,
        source=source,
        classes=classes,
        selection=selection,
        split=split,
        statistics=statistics,
        raw=raw,
    )


def verify_source_matches_statistics(manifest: DataManifest) -> None:
    """
    Check the data on disk against the counts recorded in the manifest.

    This is what catches a silently changed dataset — frames added to or
    removed from the source directory without the manifest being updated.
    Call this in the build step, not on every load.
    """
    images_directory = manifest.source.images_directory
    if not images_directory.exists():
        raise ManifestError(f"source images directory not found: {images_directory}")

    image_suffixes = {".png", ".jpg", ".jpeg"}
    frames_on_disk = sum(
        1 for entry in images_directory.iterdir() if entry.suffix.lower() in image_suffixes
    )

    if frames_on_disk != manifest.statistics.total_frames:
        raise ManifestError(
            f"{manifest.path}: manifest records "
            f"{manifest.statistics.total_frames} frames but "
            f"{frames_on_disk} are present in {images_directory}. "
            f"The manifest is stale, or the source data has changed."
        )