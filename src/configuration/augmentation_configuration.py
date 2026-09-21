"""
Augmentation configuration schema for fisheye person detection.

Project-specific: a tracker or depth network defines entirely different
augmentation fields. Defaults apply to keys omitted from the YAML; unknown
keys are rejected by the engine's loader.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.configuration import ConfigurationError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


@dataclass(frozen=True)
class AugmentationConfiguration:
    """
    Box-aware augmentation for fisheye detection.

    Fisheye distortion is roughly radially symmetric about the optical
    centre, which decides which transforms produce physically plausible
    images:

        colour jitter              always plausible
        horizontal flip            plausible when the optical centre is near
                                   the image centre, as it is for WoodScape
        scale about the centre     approximately plausible
        rotation about the centre  approximately plausible, but loosens
                                   axis-aligned boxes
        translation                moves the optical centre, so distortion no
                                   longer matches position; keep small
        mosaic                     four lens geometries stitched together;
                                   impossible at the seams

    Mosaic and mixup are the "strong" augmentations. YOLOX switches them off
    for the final epochs so the model settles on realistic images.
    """

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.0
    hsv_hue_gain: float = 0.015
    hsv_saturation_gain: float = 0.7
    hsv_value_gain: float = 0.4
    rotation_degrees: float = 0.0
    translation_fraction: float = 0.1
    scale_gain: float = 0.5
    mosaic_probability: float = 0.0
    mixup_probability: float = 0.0
    strong_augmentation_off_final_epochs: int = 15
    minimum_box_side_pixels: float = 2.0
    minimum_visible_fraction: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "horizontal_flip_probability",
            "vertical_flip_probability",
            "mosaic_probability",
            "mixup_probability",
            "minimum_visible_fraction",
        ):
            value = getattr(self, name)
            _require(0.0 <= value <= 1.0, f"augmentation.{name} must be in [0, 1], got {value}")

        for name in ("hsv_saturation_gain", "hsv_value_gain", "scale_gain", "minimum_box_side_pixels"):
            value = getattr(self, name)
            _require(value >= 0.0, f"augmentation.{name} must be non-negative, got {value}")

        _require(
            0.0 <= self.hsv_hue_gain <= 0.5,
            f"augmentation.hsv_hue_gain must be in [0, 0.5], got {self.hsv_hue_gain}",
        )
        _require(
            0.0 <= self.rotation_degrees <= 180.0,
            f"augmentation.rotation_degrees must be in [0, 180], got {self.rotation_degrees}",
        )
        _require(
            0.0 <= self.translation_fraction <= 0.5,
            f"augmentation.translation_fraction must be in [0, 0.5], got {self.translation_fraction}",
        )
        _require(
            self.scale_gain < 1.0,
            f"augmentation.scale_gain must be below 1, got {self.scale_gain}",
        )
        _require(
            self.strong_augmentation_off_final_epochs >= 0,
            "augmentation.strong_augmentation_off_final_epochs must be non-negative",
        )

    @property
    def uses_strong_augmentation(self) -> bool:
        return self.mosaic_probability > 0.0 or self.mixup_probability > 0.0