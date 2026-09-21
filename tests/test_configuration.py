# tests/test_configuration.py
"""
Tests for the engine's configuration machinery and training schema.

Deliberately uses small stub schemas defined here rather than the YOLOX
ones, so these tests would pass unchanged in a tracking or depth project.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from engine.configuration import (
    ConfigurationError,
    TrainingConfiguration,
    load_configuration,
)

MANIFEST = "data/manifests/person_detection_woodscape_version_1.yaml"


@dataclass(frozen=True)
class StubModel:
    width: int = 4
    scale: float = 1.0
    optional_path: str | None = None
    sizes: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class StubAugmentation:
    probability: float = 0.5


def _load(training_path, overrides=()):
    return load_configuration(
        training_path,
        model_schema=StubModel,
        augmentation_schema=StubAugmentation,
        overrides=overrides,
    )


@pytest.fixture
def make_files(tmp_path):
    """
    Write a training file plus the model and augmentation files it references.
    Anything not passed takes its default.
    """

    def _make(training=None, model=None, augmentation=None, directory_name="configuration"):
        directory = tmp_path / directory_name
        directory.mkdir(exist_ok=True)

        model_path = directory / "model.yaml"
        model_path.write_text(yaml.safe_dump(model or {}))
        augmentation_path = directory / "augmentation.yaml"
        augmentation_path.write_text(yaml.safe_dump(augmentation or {}))

        content = {
            "manifest": MANIFEST,
            "model_configuration": str(model_path),
            "augmentation_configuration": str(augmentation_path),
        }
        content.update(training or {})

        training_path = directory / "training.yaml"
        training_path.write_text(yaml.safe_dump(content))
        return training_path

    return _make


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_minimal_configuration_uses_defaults(make_files):
    configuration = _load(make_files())
    defaults = TrainingConfiguration(
        manifest=MANIFEST, model_configuration="unused", augmentation_configuration="unused"
    )
    assert configuration.training.epochs == defaults.epochs
    assert configuration.training.optimizer == defaults.optimizer
    assert configuration.model == StubModel()
    assert configuration.augmentation == StubAugmentation()


def test_class_names_are_derived_from_the_manifest(make_files):
    configuration = _load(make_files())
    assert configuration.class_names == ("person",)
    assert configuration.number_of_classes == 1


def test_missing_training_file_is_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="training configuration not found"):
        _load(tmp_path / "absent.yaml")


def test_missing_referenced_model_file_is_rejected(make_files, tmp_path):
    training = make_files(training={"model_configuration": str(tmp_path / "absent.yaml")})
    with pytest.raises(ConfigurationError, match="model configuration not found"):
        _load(training)


def test_missing_required_reference_is_rejected(tmp_path):
    training = tmp_path / "training.yaml"
    training.write_text(yaml.safe_dump({"manifest": MANIFEST}))
    with pytest.raises(ConfigurationError, match="missing required key"):
        _load(training)


def test_non_dataclass_schema_is_rejected(make_files):
    with pytest.raises(TypeError, match="model_schema"):
        load_configuration(
            make_files(), model_schema=dict, augmentation_schema=StubAugmentation
        )


# --------------------------------------------------------------------------
# Unknown keys — the protection against silent typos
# --------------------------------------------------------------------------


def test_unknown_training_key_is_rejected(make_files):
    with pytest.raises(ConfigurationError, match="unknown key"):
        _load(make_files(training={"epoch": 50}))


def test_unknown_nested_training_key_is_rejected(make_files):
    with pytest.raises(ConfigurationError, match="training.optimizer"):
        _load(make_files(training={"optimizer": {"learing_rate": 0.1}}))


def test_unknown_key_in_project_schema_is_rejected(make_files):
    with pytest.raises(ConfigurationError, match="model"):
        _load(make_files(model={"widht": 8}))


def test_override_typo_is_rejected(make_files):
    with pytest.raises(ConfigurationError, match="unknown key"):
        _load(make_files(), overrides=["training.optimizer.learing_rate=0.001"])


# --------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------


def test_training_override_changes_value(make_files):
    configuration = _load(make_files(), overrides=["training.optimizer.learning_rate=0.001"])
    assert configuration.training.optimizer.learning_rate == pytest.approx(0.001)


def test_model_override_changes_value(make_files):
    configuration = _load(make_files(), overrides=["model.width=16"])
    assert configuration.model.width == 16


def test_augmentation_override_changes_value(make_files):
    configuration = _load(make_files(), overrides=["augmentation.probability=0.1"])
    assert configuration.augmentation.probability == pytest.approx(0.1)


def test_override_can_swap_the_model_file(make_files, tmp_path):
    alternative = tmp_path / "alternative_model.yaml"
    alternative.write_text(yaml.safe_dump({"width": 8}))
    configuration = _load(
        make_files(), overrides=[f"training.model_configuration={alternative}"]
    )
    assert configuration.model.width == 8


@pytest.mark.parametrize(
    "override",
    [
        "training.epochs",              # no value
        "optimizer.learning_rate=0.1",  # unknown root
        "data.manifest=other.yaml",     # unknown root
        "training=5",                   # no key inside the section
        "training..epochs=5",           # empty key
    ],
)
def test_malformed_override_is_rejected(make_files, override):
    with pytest.raises(ConfigurationError):
        _load(make_files(), overrides=[override])


def test_override_into_a_value_is_rejected(make_files):
    with pytest.raises(ConfigurationError, match="is a value, not a section"):
        _load(make_files(training={"epochs": 10}), overrides=["training.epochs.value=3"])


def test_scientific_notation_without_decimal_gets_a_helpful_error(make_files):
    with pytest.raises(ConfigurationError, match="1.0e-3"):
        _load(make_files(), overrides=["training.optimizer.learning_rate=1e-3"])


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


def test_string_for_integer_is_rejected(make_files):
    with pytest.raises(ConfigurationError, match="integer"):
        _load(make_files(training={"epochs": "ten"}))


def test_boolean_for_integer_is_rejected(make_files):
    with pytest.raises(ConfigurationError, match="integer"):
        _load(make_files(training={"epochs": True}))


def test_integer_is_accepted_for_float(make_files):
    configuration = _load(make_files(model={"scale": 2}))
    assert isinstance(configuration.model.scale, float)


def test_null_for_required_value_is_rejected(make_files):
    with pytest.raises(ConfigurationError, match="null is not allowed"):
        _load(make_files(), overrides=["training.epochs=null"])


def test_null_is_accepted_for_optional_value(make_files):
    configuration = _load(make_files(model={"optional_path": None}))
    assert configuration.model.optional_path is None


def test_list_becomes_tuple(make_files):
    configuration = _load(make_files(model={"sizes": [3, 4]}))
    assert configuration.model.sizes == (3, 4)


def test_wrong_element_type_in_list_is_rejected(make_files):
    with pytest.raises(ConfigurationError, match=r"sizes\[1\]"):
        _load(make_files(model={"sizes": [3, "four"]}))


# --------------------------------------------------------------------------
# Training schema validation
# --------------------------------------------------------------------------


def test_invalid_choice_is_rejected(make_files):
    with pytest.raises(ConfigurationError, match="mixed_precision"):
        _load(make_files(), overrides=["training.mixed_precision=float8"])


def test_warmup_longer_than_training_is_rejected(make_files):
    with pytest.raises(ConfigurationError, match="warmup_epochs"):
        _load(make_files(), overrides=["training.epochs=2"])


def test_weights_and_biases_requires_a_project(make_files):
    with pytest.raises(ConfigurationError, match="tracker.project"):
        _load(make_files(), overrides=["training.tracker.kind=weights_and_biases"])


def test_effective_batch_size(make_files):
    configuration = _load(
        make_files(training={"batch_size": 8, "gradient_accumulation_steps": 4})
    )
    assert configuration.training.effective_batch_size == 32


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def test_override_changes_hash(make_files):
    training = make_files()
    assert _load(training).hash != _load(training, overrides=["training.epochs=50"]).hash


def test_same_values_by_different_routes_give_same_hash(make_files):
    from_file = _load(make_files(training={"epochs": 50}, directory_name="from_file"))
    from_override = _load(
        make_files(directory_name="from_override"), overrides=["training.epochs=50"]
    )
    assert from_file.hash == from_override.hash


def test_renaming_the_run_does_not_change_hash(make_files):
    training = make_files()
    assert _load(training).hash == _load(training, overrides=["training.name=renamed"]).hash


def test_tracker_settings_do_not_change_hash(make_files):
    training = make_files()
    tracked = _load(
        training,
        overrides=[
            "training.tracker.kind=weights_and_biases",
            "training.tracker.project=some_project",
        ],
    )
    assert _load(training).hash == tracked.hash


def test_section_hashes_are_independent(make_files):
    training = make_files()
    baseline = _load(training)
    changed = _load(training, overrides=["model.width=16"])
    assert baseline.training_hash == changed.training_hash
    assert baseline.augmentation_hash == changed.augmentation_hash
    assert baseline.model_hash != changed.model_hash


def test_editing_the_manifest_changes_hash(make_files, tmp_path):
    original = _load(make_files())

    manifest = yaml.safe_load(Path(MANIFEST).read_text())
    manifest["identity"]["description"] = "edited"
    edited_manifest = tmp_path / "edited_manifest.yaml"
    edited_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False))

    edited = _load(
        make_files(training={"manifest": str(edited_manifest)}, directory_name="edited")
    )
    assert original.hash != edited.hash


def test_to_dictionary_is_json_serialisable(make_files):
    configuration = _load(make_files(), overrides=["training.epochs=50"])
    dictionary = configuration.to_dictionary()
    json.dumps(dictionary)

    assert dictionary["overrides"] == ["training.epochs=50"]
    assert dictionary["data"]["class_names"] == ["person"]
    assert set(dictionary["hashes"]) == {"configuration", "training", "model", "augmentation"}
    assert set(dictionary["sources"]) == {
        "training_configuration_path",
        "model_configuration_path",
        "augmentation_configuration_path",
    }