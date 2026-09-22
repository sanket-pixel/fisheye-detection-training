# tests/test_experiment_logging.py
"""Tests for run logging. Weights & Biases itself is not exercised here."""
from __future__ import annotations

import json

import pytest
import torch

from engine.experiment_logging import (
    MetricAverager,
    NullTracker,
    RunLogger,
    WeightsAndBiasesTracker,
    create_tracker,
)


def _read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_metrics_are_written_to_jsonl(tmp_path):
    with RunLogger(tmp_path / "run") as logger:
        logger.start("test_run", {"learning_rate": 0.01})
        logger.log_metrics({"loss": 1.0}, step=1, epoch=0)
        logger.log_metrics({"loss": 0.5}, step=2, epoch=0)

    records = _read_records(tmp_path / "run" / "metrics.jsonl")
    assert [record["loss"] for record in records] == [1.0, 0.5]
    assert [record["step"] for record in records] == [1, 2]
    assert all(record["epoch"] == 0 for record in records)


def test_configuration_is_written(tmp_path):
    configuration = {"learning_rate": 0.01, "provenance": {"git_commit": "abc1234"}}
    with RunLogger(tmp_path / "run") as logger:
        logger.start("test_run", configuration)

    written = json.loads((tmp_path / "run" / "configuration.json").read_text())
    assert written == configuration


def test_tensor_values_are_converted(tmp_path):
    with RunLogger(tmp_path / "run") as logger:
        logger.start("test_run", {})
        logger.log_metrics({"loss": torch.tensor(0.25)}, step=1)

    records = _read_records(tmp_path / "run" / "metrics.jsonl")
    assert records[0]["loss"] == pytest.approx(0.25)


def test_tracker_is_finished_when_training_crashes(tmp_path):
    class RecordingTracker(NullTracker):
        finished = False

        def finish(self):
            RecordingTracker.finished = True

    with pytest.raises(RuntimeError):
        with RunLogger(tmp_path / "run", tracker=RecordingTracker()) as logger:
            logger.start("test_run", {})
            raise RuntimeError("simulated crash")

    assert RecordingTracker.finished


def test_averager_weights_by_count():
    averager = MetricAverager()
    averager.update({"loss": 1.0}, count=1)
    averager.update({"loss": 4.0}, count=3)
    assert averager.compute()["loss"] == pytest.approx(3.25)

    averager.reset()
    assert averager.compute() == {}


def test_unknown_tracker_is_rejected():
    with pytest.raises(ValueError, match="unknown tracker"):
        create_tracker("tensorboard")


def test_invalid_weights_and_biases_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        WeightsAndBiasesTracker(project="test", mode="sideways")


def test_none_tracker_is_null():
    assert isinstance(create_tracker("none"), NullTracker)

def test_console_output_is_saved_with_the_run(tmp_path):
    with RunLogger(tmp_path / "run") as logger:
        logger.start("test_run", {})
        logger.info("a message worth keeping")

    content = (tmp_path / "run" / "console.log").read_text()
    assert "a message worth keeping" in content
    assert "finished in" in content


def test_crash_is_recorded_and_logs_stay_separate(tmp_path):
    with pytest.raises(RuntimeError):
        with RunLogger(tmp_path / "first") as logger:
            logger.start("first", {})
            raise RuntimeError("simulated crash")

    with RunLogger(tmp_path / "second") as logger:
        logger.start("second", {})
        logger.info("only in the second run")

    first = (tmp_path / "first" / "console.log").read_text()
    assert "simulated crash" in first
    assert "only in the second run" not in first