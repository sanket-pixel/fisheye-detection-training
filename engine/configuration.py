# engine/configuration.py
"""
Configuration loading, overriding, validation, and hashing.

A run is fully specified by one training configuration file, which
references the dataset manifest and the model and augmentation
configuration files:

    configs/training/yolox_nano_baseline.yaml
        manifest:                    data/manifests/<name>.yaml
        model_configuration:         configs/model/yolox_nano.yaml
        augmentation_configuration:  configs/augmentation/fisheye_conservative.yaml
        epochs, optimizer, schedule, ...

Ownership is split along what varies between projects:

    engine (this file)   The loading machinery, and TrainingConfiguration:
                         optimisation, precision, checkpointing, tracking.
                         Identical for detection, tracking, depth, fusion.

    project (src/)       The model and augmentation schemas, which mean
                         nothing outside one task. They are passed in as
                         dataclass types; the engine never imports them.

    configuration = load_configuration(
        "configs/training/yolox_nano_baseline.yaml",
        model_schema=ModelConfiguration,
        augmentation_schema=AugmentationConfiguration,
        overrides=["training.optimizer.learning_rate=0.001"],
    )

Rules:
  - Keys missing from a file take the schema default.
  - Unknown keys are always rejected, in files and in overrides, so a typo
    such as `learing_rate` crashes at startup instead of being ignored.
  - Types are checked. Integers are accepted for floats; booleans are never
    accepted for numbers.
  - Overrides have three roots: training, model, augmentation. Training
    overrides are applied first, so `training.model_configuration=<path>`
    swaps the model file for one run.
  - Hashes cover the values that change what a run produces and exclude
    labels and file locations. Renaming a file, renaming the run, or
    reaching the same values through an override leaves the hash unchanged.
    Editing the manifest changes it.
"""
from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml

from engine.data_manifest import load_manifest
from engine.provenance import hash_dict, hash_file

CONFIGURATION_ROOTS = ("training", "model", "augmentation")

# Training keys that label or locate things rather than change the result.
# Excluded from the training hash; data identity enters through the
# manifest's content hash instead of its path.
NON_SEMANTIC_TRAINING_KEYS = frozenset(
    {
        "name",
        "manifest",
        "model_configuration",
        "augmentation_configuration",
        "log_interval_steps",
        "tracker",
    }
)

ModelSchema = TypeVar("ModelSchema")
AugmentationSchema = TypeVar("AugmentationSchema")


class ConfigurationError(ValueError):
    """Raised when a configuration file or override is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def _require_choice(value: Any, choices: set, name: str) -> None:
    _require(value in choices, f"{name} must be one of {sorted(choices)}, got {value!r}")


# --------------------------------------------------------------------------
# Training schema — shared by every project
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizerConfiguration:
    type: str = "sgd"
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    nesterov: bool = True

    def __post_init__(self) -> None:
        _require_choice(self.type, {"sgd", "adamw"}, "training.optimizer.type")
        _require(self.learning_rate > 0, "training.optimizer.learning_rate must be positive")
        _require(self.weight_decay >= 0, "training.optimizer.weight_decay must be non-negative")


@dataclass(frozen=True)
class ScheduleConfiguration:
    type: str = "cosine"
    warmup_epochs: float = 3.0
    final_learning_rate_fraction: float = 0.05

    def __post_init__(self) -> None:
        _require_choice(self.type, {"cosine", "linear", "constant"}, "training.schedule.type")
        _require(self.warmup_epochs >= 0, "training.schedule.warmup_epochs must be non-negative")
        _require(
            0.0 <= self.final_learning_rate_fraction <= 1.0,
            "training.schedule.final_learning_rate_fraction must be in [0, 1]",
        )


@dataclass(frozen=True)
class CheckpointConfiguration:
    metric_name: str = "mean_average_precision_50"
    metric_mode: str = "maximise"
    keep_last_n_epochs: int = 0

    def __post_init__(self) -> None:
        _require(bool(self.metric_name), "training.checkpoint.metric_name must not be empty")
        _require_choice(
            self.metric_mode, {"maximise", "minimise"}, "training.checkpoint.metric_mode"
        )
        _require(
            self.keep_last_n_epochs >= 0,
            "training.checkpoint.keep_last_n_epochs must be non-negative",
        )


@dataclass(frozen=True)
class TrackerConfiguration:
    kind: str = "none"
    project: str | None = None
    entity: str | None = None
    mode: str = "online"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_choice(self.kind, {"none", "weights_and_biases"}, "training.tracker.kind")
        _require_choice(self.mode, {"online", "offline", "disabled"}, "training.tracker.mode")
        if self.kind == "weights_and_biases":
            _require(
                self.project is not None,
                "training.tracker.project is required for weights_and_biases",
            )


@dataclass(frozen=True)
class TrainingConfiguration:
    manifest: str
    model_configuration: str
    augmentation_configuration: str
    name: str = "unnamed_run"
    epochs: int = 100
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    number_of_data_loader_workers: int = 4
    seed: int = 42
    mixed_precision: str = "bfloat16"
    gradient_clipping_norm: float | None = 10.0
    exponential_moving_average_decay: float | None = 0.9998
    evaluation_interval_epochs: int = 1
    log_interval_steps: int = 50
    optimizer: OptimizerConfiguration = field(default_factory=OptimizerConfiguration)
    schedule: ScheduleConfiguration = field(default_factory=ScheduleConfiguration)
    checkpoint: CheckpointConfiguration = field(default_factory=CheckpointConfiguration)
    tracker: TrackerConfiguration = field(default_factory=TrackerConfiguration)

    def __post_init__(self) -> None:
        for name in ("manifest", "model_configuration", "augmentation_configuration"):
            _require(bool(getattr(self, name)), f"training.{name} must not be empty")
        _require(self.epochs > 0, "training.epochs must be positive")
        _require(self.batch_size > 0, "training.batch_size must be positive")
        _require(
            self.gradient_accumulation_steps >= 1,
            "training.gradient_accumulation_steps must be at least 1",
        )
        _require(
            self.number_of_data_loader_workers >= 0,
            "training.number_of_data_loader_workers must be non-negative",
        )
        _require_choice(
            self.mixed_precision,
            {"float32", "float16", "bfloat16"},
            "training.mixed_precision",
        )
        if self.gradient_clipping_norm is not None:
            _require(
                self.gradient_clipping_norm > 0,
                "training.gradient_clipping_norm must be positive",
            )
        if self.exponential_moving_average_decay is not None:
            _require(
                0.0 < self.exponential_moving_average_decay < 1.0,
                "training.exponential_moving_average_decay must be in (0, 1)",
            )
        _require(
            self.evaluation_interval_epochs >= 1,
            "training.evaluation_interval_epochs must be at least 1",
        )
        _require(self.log_interval_steps >= 1, "training.log_interval_steps must be at least 1")
        _require(
            self.schedule.warmup_epochs < self.epochs,
            f"training.schedule.warmup_epochs ({self.schedule.warmup_epochs}) "
            f"must be less than training.epochs ({self.epochs})",
        )

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


# --------------------------------------------------------------------------
# The resolved configuration handed to the trainer
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedConfiguration(Generic[ModelSchema, AugmentationSchema]):
    training: TrainingConfiguration
    model: ModelSchema
    augmentation: AugmentationSchema
    class_names: tuple[str, ...]
    manifest_hash: str
    overrides: tuple[str, ...]
    training_configuration_path: str

    @property
    def number_of_classes(self) -> int:
        return len(self.class_names)

    def _semantic_training_values(self) -> dict[str, Any]:
        values = dataclasses.asdict(self.training)
        for key in NON_SEMANTIC_TRAINING_KEYS:
            values.pop(key, None)
        return values

    def data_dictionary(self) -> dict[str, Any]:
        return {
            "manifest": self.training.manifest,
            "manifest_hash": self.manifest_hash,
            "class_names": list(self.class_names),
            "number_of_classes": self.number_of_classes,
        }

    @property
    def training_hash(self) -> str:
        return hash_dict(self._semantic_training_values())

    @property
    def model_hash(self) -> str:
        return hash_dict(dataclasses.asdict(self.model))

    @property
    def augmentation_hash(self) -> str:
        return hash_dict(dataclasses.asdict(self.augmentation))

    @property
    def hash(self) -> str:
        """Identity of the run: everything that affects what it produces."""
        return hash_dict(
            {
                "training": self._semantic_training_values(),
                "model": dataclasses.asdict(self.model),
                "augmentation": dataclasses.asdict(self.augmentation),
                "data": {
                    "manifest_hash": self.manifest_hash,
                    "class_names": list(self.class_names),
                },
            }
        )

    def to_dictionary(self) -> dict[str, Any]:
        """The form written into runs/<run_id>/configuration.json."""
        return {
            "training": dataclasses.asdict(self.training),
            "model": dataclasses.asdict(self.model),
            "augmentation": dataclasses.asdict(self.augmentation),
            "data": self.data_dictionary(),
            "overrides": list(self.overrides),
            "sources": {
                "training_configuration_path": self.training_configuration_path,
                "model_configuration_path": self.training.model_configuration,
                "augmentation_configuration_path": self.training.augmentation_configuration,
            },
            "hashes": {
                "configuration": self.hash,
                "training": self.training_hash,
                "model": self.model_hash,
                "augmentation": self.augmentation_hash,
            },
        }


# --------------------------------------------------------------------------
# Building typed objects from dictionaries — works for any dataclass schema
# --------------------------------------------------------------------------


def _type_error(location: str, expected: str, value: Any) -> ConfigurationError:
    message = f"{location}: expected {expected}, got {type(value).__name__} {value!r}"
    if expected == "a number" and isinstance(value, str):
        message += (
            ". Note that YAML reads scientific notation without a decimal point "
            "(1e-3) as a string; write 1.0e-3 instead"
        )
    return ConfigurationError(message)


def _coerce(value: Any, hint: Any, location: str) -> Any:
    origin = typing.get_origin(hint)
    arguments = typing.get_args(hint)

    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in arguments:
            return None
        remaining = [argument for argument in arguments if argument is not type(None)]
        if len(remaining) != 1:
            raise ConfigurationError(f"{location}: unsupported union type {hint}")
        return _coerce(value, remaining[0], location)

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise _type_error(location, "a list", value)
        element_hint = arguments[0] if arguments else Any
        return tuple(
            _coerce(element, element_hint, f"{location}[{index}]")
            for index, element in enumerate(value)
        )

    if hint is Any:
        return value

    if value is None:
        raise ConfigurationError(f"{location}: null is not allowed here")

    if hint is bool:
        if not isinstance(value, bool):
            raise _type_error(location, "true or false", value)
        return value

    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _type_error(location, "an integer", value)
        return value

    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _type_error(location, "a number", value)
        return float(value)

    if hint is str:
        if not isinstance(value, str):
            raise _type_error(location, "a string", value)
        return value

    raise ConfigurationError(f"{location}: unsupported type {hint}")


def _build(schema: type, data: Any, location: str) -> Any:
    """Construct a schema dataclass from a dictionary, checking keys and types."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"{location}: expected a mapping, got {type(data).__name__}"
        )

    schema_fields = {item.name: item for item in dataclasses.fields(schema)}

    unknown = sorted(set(data) - set(schema_fields))
    if unknown:
        raise ConfigurationError(
            f"{location}: unknown key(s) {unknown}. Known keys: {sorted(schema_fields)}"
        )

    missing = sorted(
        name
        for name, item in schema_fields.items()
        if item.default is dataclasses.MISSING
        and item.default_factory is dataclasses.MISSING
        and name not in data
    )
    if missing:
        raise ConfigurationError(f"{location}: missing required key(s) {missing}")

    hints = typing.get_type_hints(schema)
    values: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        child_location = f"{location}.{name}"
        if dataclasses.is_dataclass(hint):
            values[name] = _build(hint, value, child_location)
        else:
            values[name] = _coerce(value, hint, child_location)

    return schema(**values)


def _require_schema(schema: Any, name: str) -> None:
    if not (isinstance(schema, type) and dataclasses.is_dataclass(schema)):
        raise TypeError(f"{name} must be a dataclass type, got {schema!r}")


# --------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------


def parse_override(text: str) -> tuple[list[str], Any]:
    """
    Parse `root.key.subkey=value`.

    The value is parsed as YAML, so `0.001` becomes a float, `true` a
    boolean, `null` None, and `[a, b]` a list.
    """
    if "=" not in text:
        raise ConfigurationError(
            f"override {text!r} must have the form section.key=value"
        )
    path, raw_value = text.split("=", 1)
    keys = path.strip().split(".")

    if len(keys) < 2 or any(not key for key in keys):
        raise ConfigurationError(
            f"override {text!r} must name a key inside a section, e.g. "
            "training.epochs=50"
        )
    if keys[0] not in CONFIGURATION_ROOTS:
        raise ConfigurationError(
            f"override {text!r} must start with one of {list(CONFIGURATION_ROOTS)}"
        )

    return keys, yaml.safe_load(raw_value)


def _apply_override(section: dict[str, Any], keys: list[str], value: Any, text: str) -> None:
    """Set a value inside one section. `keys` excludes the root."""
    node = section
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        elif not isinstance(child, dict):
            raise ConfigurationError(
                f"override {text!r}: {key!r} is a value, not a section"
            )
        node = child
    node[keys[-1]] = value


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _load_yaml(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"{description} not found: {path}")
    with open(path) as handle:
        content = yaml.safe_load(handle)
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ConfigurationError(f"{path}: expected a mapping at the top level")
    return content


def load_configuration(
    training_configuration_path: str | Path,
    *,
    model_schema: type[ModelSchema],
    augmentation_schema: type[AugmentationSchema],
    overrides: list[str] | tuple[str, ...] = (),
) -> ResolvedConfiguration[ModelSchema, AugmentationSchema]:
    """
    Load the training configuration and everything it references, apply
    overrides, validate against the schemas, and derive from the manifest.
    """
    _require_schema(model_schema, "model_schema")
    _require_schema(augmentation_schema, "augmentation_schema")

    overrides_by_root: dict[str, list[tuple[str, list[str], Any]]] = {
        root: [] for root in CONFIGURATION_ROOTS
    }
    for text in overrides:
        keys, value = parse_override(text)
        overrides_by_root[keys[0]].append((text, keys[1:], value))

    # Training first: its overrides may change which model or augmentation
    # file is loaded.
    training_path = Path(training_configuration_path)
    raw_training = _load_yaml(training_path, "training configuration")
    for text, keys, value in overrides_by_root["training"]:
        _apply_override(raw_training, keys, value, text)
    training = _build(TrainingConfiguration, raw_training, "training")

    raw_model = _load_yaml(Path(training.model_configuration), "model configuration")
    for text, keys, value in overrides_by_root["model"]:
        _apply_override(raw_model, keys, value, text)
    model = _build(model_schema, raw_model, "model")

    raw_augmentation = _load_yaml(
        Path(training.augmentation_configuration), "augmentation configuration"
    )
    for text, keys, value in overrides_by_root["augmentation"]:
        _apply_override(raw_augmentation, keys, value, text)
    augmentation = _build(augmentation_schema, raw_augmentation, "augmentation")

    manifest = load_manifest(training.manifest)

    return ResolvedConfiguration(
        training=training,
        model=model,
        augmentation=augmentation,
        class_names=tuple(manifest.classes.names),
        manifest_hash=hash_file(training.manifest),
        overrides=tuple(overrides),
        training_configuration_path=str(training_path),
    )