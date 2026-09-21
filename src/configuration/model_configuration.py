"""
Model configuration schema for YOLOX person detection.

The engine owns the loader and validation machinery; this schema is
project-specific and is what a different project (a tracker, a depth
network) replaces. Defaults apply to keys omitted from the YAML; unknown
keys are rejected by the engine's loader.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.configuration import ConfigurationError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def _require_choice(value: Any, choices: set, name: str) -> None:
    _require(value in choices, f"{name} must be one of {sorted(choices)}, got {value!r}")


@dataclass(frozen=True)
class ModelConfiguration:
    """
    YOLOX: CSPDarknet backbone, PAFPN neck, decoupled anchor-free head.

    Published variants, by depth and width multiplier:

        nano   0.33  0.25    depthwise convolutions, 416 input
        tiny   0.33  0.375   416 input
        s      0.33  0.50
        m      0.67  0.75
        l      1.00  1.00
        x      1.33  1.25

    The number of classes is not configured here; it is derived from the
    dataset manifest so it cannot disagree with the data.
    """

    architecture: str = "yolox"
    depth_multiplier: float = 0.33
    width_multiplier: float = 0.25
    use_depthwise_convolutions: bool = True
    activation: str = "silu"
    feature_strides: tuple[int, ...] = (8, 16, 32)
    input_height: int = 640
    input_width: int = 640
    pretrained_weights: str | None = None

    def __post_init__(self) -> None:
        _require_choice(self.architecture, {"yolox"}, "model.architecture")
        _require(self.depth_multiplier > 0, "model.depth_multiplier must be positive")
        _require(self.width_multiplier > 0, "model.width_multiplier must be positive")
        _require_choice(self.activation, {"silu", "relu", "leaky_relu"}, "model.activation")

        strides = self.feature_strides
        _require(len(strides) > 0, "model.feature_strides must not be empty")
        _require(all(stride > 0 for stride in strides), "model.feature_strides must be positive")
        _require(
            list(strides) == sorted(set(strides)),
            f"model.feature_strides must be strictly increasing, got {list(strides)}",
        )

        largest = strides[-1]
        _require(
            self.input_height % largest == 0 and self.input_width % largest == 0,
            f"model input dimensions must be multiples of the largest feature "
            f"stride ({largest}); got {self.input_height}x{self.input_width}",
        )

    @property
    def number_of_feature_levels(self) -> int:
        return len(self.feature_strides)