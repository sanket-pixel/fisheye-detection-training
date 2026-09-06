# src/data/source_readers.py
"""
Readers for source annotation formats.

One reader per `source.annotation_format` value in a manifest. Each returns
a canonical list of annotations so that everything downstream — conversion,
statistics, inspection — is format-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceAnnotation:
    """One annotation in absolute pixel corner coordinates."""

    class_name: str
    x_minimum: float
    y_minimum: float
    x_maximum: float
    y_maximum: float

    @property
    def width(self) -> float:
        return self.x_maximum - self.x_minimum

    @property
    def height(self) -> float:
        return self.y_maximum - self.y_minimum

    @property
    def area(self) -> float:
        return self.width * self.height


def read_woodscape_box_2d(annotation_path: Path) -> list[SourceAnnotation]:
    """
    WoodScape 2D box format: one annotation per line,

        class_name,class_index,x_minimum,y_minimum,x_maximum,y_maximum

    with absolute pixel coordinates. Malformed lines are skipped here; the
    data validation tests are responsible for failing loudly on them.
    """
    if not annotation_path.exists():
        return []

    annotations: list[SourceAnnotation] = []
    with open(annotation_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) != 6:
                continue
            try:
                coordinates = [float(value) for value in fields[2:]]
            except ValueError:
                continue
            annotations.append(
                SourceAnnotation(fields[0], *coordinates)
            )
    return annotations


READERS = {
    "woodscape_box_2d": read_woodscape_box_2d,
}


def get_reader(annotation_format: str):
    if annotation_format not in READERS:
        raise KeyError(
            f"no reader registered for annotation_format {annotation_format!r}; "
            f"available: {sorted(READERS)}"
        )
    return READERS[annotation_format]