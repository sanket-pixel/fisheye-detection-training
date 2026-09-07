# tools/data/inspect_dataset.py
"""
Visual inspection of a dataset in FiftyOne.

    # Raw source annotations, as they exist before conversion
    python -m tools.data.inspect_dataset \
        --manifest data/manifests/person_detection_woodscape_version_1.yaml \
        --stage source --limit 300

    # Converted labels from the build — the round-trip check
    python -m tools.data.inspect_dataset \
        --manifest data/manifests/person_detection_woodscape_version_1.yaml \
        --stage build --partition train --limit 300

Both stages render into the same field so they can be compared directly.
Inspecting the build is the check that matters: it renders what training
actually reads, so a conversion bug (swapped axes, wrong normalisation,
centre-versus-corner confusion) is visible immediately rather than after
hours of training.

Every sample carries metadata fields (camera position, box area, radial
distance from the image centre) so the same tool supports sliced failure
analysis once predictions exist.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import fiftyone as fo
from PIL import Image

from engine.data_manifest import DataManifest, load_manifest
from src.data.source_readers import get_reader

# WoodScape encodes camera position in the filename suffix. Duplicated from
# the build tool deliberately: this is source-format knowledge, and pulling
# it into engine/ would make the engine dataset-aware.
CAMERA_POSITION_BY_SUFFIX = {
    "FV": "front",
    "MVL": "mirror_left",
    "MVR": "mirror_right",
    "RV": "rear",
}

LABEL_FIELD = "ground_truth"
LABEL_FIELD_COUNT = 5


def camera_position_of(frame_name: str) -> str | None:
    suffix = Path(frame_name).stem.split("_")[-1]
    return CAMERA_POSITION_BY_SUFFIX.get(suffix)


def radial_distance_of(centre_x: float, centre_y: float) -> float:
    """
    Distance of a box centre from the image centre, normalised so that the
    image centre is 0.0 and a corner is 1.0.

    This is the fisheye-specific axis worth slicing on: distortion grows
    with radius, so recall at high radial distance is the number that
    actually tells you whether edge distortion is hurting the model.
    """
    offset_x = centre_x - 0.5
    offset_y = centre_y - 0.5
    return math.hypot(offset_x, offset_y) / math.hypot(0.5, 0.5)


# --------------------------------------------------------------------------
# Source stage
# --------------------------------------------------------------------------


def detections_from_source(
    manifest: DataManifest, frame_name: str, image_width: int, image_height: int
) -> list[fo.Detection]:
    reader = get_reader(manifest.source.annotation_format)
    annotation_path = (
        manifest.source.annotations_directory / f"{Path(frame_name).stem}.txt"
    )

    detections = []
    for annotation in reader(annotation_path):
        # FiftyOne wants relative [top-left x, top-left y, width, height]
        width = annotation.width / image_width
        height = annotation.height / image_height
        left = annotation.x_minimum / image_width
        top = annotation.y_minimum / image_height

        centre_x = left + width / 2
        centre_y = top + height / 2

        detections.append(
            fo.Detection(
                label=annotation.class_name,
                bounding_box=[left, top, width, height],
                area_pixels=annotation.area,
                radial_distance=radial_distance_of(centre_x, centre_y),
            )
        )
    return detections


def frames_for_source(manifest: DataManifest) -> list[tuple[str, Path]]:
    directory = manifest.source.images_directory
    excluded = set(manifest.selection.excluded_frames)
    return [
        (entry.name, entry)
        for entry in sorted(directory.iterdir())
        if entry.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and entry.name not in excluded
    ]


# --------------------------------------------------------------------------
# Build stage
# --------------------------------------------------------------------------


def detections_from_build(
    manifest: DataManifest,
    build_directory: Path,
    partition: str,
    frame_name: str,
    image_width: int,
    image_height: int,
) -> list[fo.Detection]:
    label_path = build_directory / "labels" / partition / f"{Path(frame_name).stem}.txt"
    if not label_path.exists():
        return []

    class_names = manifest.classes.mapping
    detections = []

    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != LABEL_FIELD_COUNT:
            continue

        class_index = int(fields[0])
        centre_x, centre_y, width, height = (float(value) for value in fields[1:])

        detections.append(
            fo.Detection(
                label=class_names.get(class_index, f"unknown_{class_index}"),
                bounding_box=[centre_x - width / 2, centre_y - height / 2, width, height],
                area_pixels=width * image_width * height * image_height,
                radial_distance=radial_distance_of(centre_x, centre_y),
            )
        )
    return detections


def frames_for_build(
    manifest: DataManifest, build_directory: Path, partition: str
) -> list[tuple[str, Path]]:
    split_path = build_directory / "splits" / f"{partition}.txt"
    if not split_path.exists():
        raise SystemExit(
            f"{split_path} not found; run tools.data.build_dataset first"
        )

    images_directory = build_directory / "images" / partition
    return [
        (name, images_directory / name)
        for name in (
            line.strip() for line in split_path.read_text().splitlines() if line.strip()
        )
    ]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_fiftyone_dataset(
    manifest: DataManifest,
    stage: str,
    partition: str | None,
    limit: int | None,
    dataset_name: str,
) -> fo.Dataset:
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    build_directory = manifest.build_directory

    if stage == "source":
        frames = frames_for_source(manifest)
    else:
        if partition is None:
            raise SystemExit("--partition is required when --stage build")
        frames = frames_for_build(manifest, build_directory, partition)

    if limit is not None:
        frames = frames[:limit]

    samples = []
    for frame_name, image_path in frames:
        resolved = image_path.resolve()
        with Image.open(resolved) as image:
            image_width, image_height = image.size

        if stage == "source":
            detections = detections_from_source(
                manifest, frame_name, image_width, image_height
            )
        else:
            detections = detections_from_build(
                manifest, build_directory, partition, frame_name, image_width, image_height
            )

        sample = fo.Sample(filepath=str(resolved))
        sample[LABEL_FIELD] = fo.Detections(detections=detections)

        # Sample-level metadata for sliced analysis
        sample["camera_position"] = camera_position_of(frame_name)
        sample["annotation_count"] = len(detections)
        sample["stage"] = stage
        if partition is not None:
            sample["partition"] = partition

        samples.append(sample)

    dataset.add_samples(samples)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--stage",
        choices=("source", "build"),
        default="build",
        help="Inspect raw source annotations or the converted build output.",
    )
    parser.add_argument(
        "--partition",
        default="train",
        help="Which build partition to inspect. Ignored when --stage source.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of frames to load. Omit the flag's default with --limit 0 for all.",
    )
    parser.add_argument("--name", default=None, help="FiftyOne dataset name.")
    parser.add_argument(
        "--no-app",
        action="store_true",
        help="Build and summarise without launching the app.",
    )
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    partition = arguments.partition if arguments.stage == "build" else None
    limit = None if arguments.limit == 0 else arguments.limit

    dataset_name = arguments.name or (
        f"{manifest.identity.name}_{arguments.stage}"
        + (f"_{partition}" if partition else "")
    )

    dataset = build_fiftyone_dataset(
        manifest, arguments.stage, partition, limit, dataset_name
    )

    print(f"manifest: {manifest}")
    print(f"stage:    {arguments.stage}" + (f" ({partition})" if partition else ""))
    print(f"samples:  {len(dataset)}")
    print(f"labels:   {dataset.count_values(f'{LABEL_FIELD}.detections.label')}")
    print(f"cameras:  {dataset.count_values('camera_position')}")

    if arguments.no_app:
        return

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()