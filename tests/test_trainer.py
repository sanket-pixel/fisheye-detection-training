# tests/test_trainer.py
"""
The toy gate: the real trainer, end to end, on a problem small enough that
the only way to fail is a bug in the training machinery.

Ten fixed random input/target pairs and a small network with roughly a
thousand parameters. Memorising forty numbers with a thousand parameters is
trivial, so a correct training loop must drive the loss to near zero. If it
cannot, the loop is broken — and it is far cheaper to learn that here than
in a YOLOX run, where a bad result could be blamed on the data, the model,
or the loss.

Also verified: the run directory holds what it should, and a run that
crashes halfway and is resumed ends in exactly the state of a run that
never stopped.

    make toy-gate    # this file, with training output visible
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset

from engine.checkpoint import CheckpointManager
from engine.configuration import (
    CheckpointConfiguration,
    OptimizerConfiguration,
    ScheduleConfiguration,
    TrainingConfiguration,
)
from engine.experiment_logging import RunLogger
from engine.exponential_moving_average import ModelExponentialMovingAverage
from engine.trainer import (
    EpochSeededRandomSampler,
    Trainer,
    TrainerError,
    build_optimizer,
    learning_rate_factor,
)

NUMBER_OF_SAMPLES = 10
INPUT_SIZE = 8
HIDDEN_SIZE = 64
OUTPUT_SIZE = 4
DATA_SEED = 1234


# --------------------------------------------------------------------------
# The toy problem, written against the same contract a real project uses
# --------------------------------------------------------------------------


class ToyDataset(Dataset):
    """Ten fixed random pairs, returned in the engine's canonical sample format."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(DATA_SEED)
        self.inputs = torch.randn(NUMBER_OF_SAMPLES, INPUT_SIZE, generator=generator)
        self.targets = torch.randn(NUMBER_OF_SAMPLES, OUTPUT_SIZE, generator=generator)

    def __len__(self) -> int:
        return NUMBER_OF_SAMPLES

    def __getitem__(self, index: int) -> dict:
        return {
            "inputs": self.inputs[index],
            "targets": self.targets[index],
            "meta": {"sample_id": f"toy_{index:02d}"},
        }


def build_toy_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
        nn.LayerNorm(HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
    )


def toy_training_step(model: nn.Module, batch: dict) -> dict[str, torch.Tensor]:
    predictions = model(batch["inputs"])
    mean_squared_error = functional.mse_loss(predictions.float(), batch["targets"])
    return {"loss": mean_squared_error, "mean_squared_error": mean_squared_error}


def toy_evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    total = 0.0
    count = 0
    for batch in loader:
        predictions = model(batch["inputs"].to(device))
        targets = batch["targets"].to(device)
        total += functional.mse_loss(predictions, targets, reduction="sum").item()
        count += targets.numel()
    return {"validation_loss": total / count}


def toy_configuration(**changes) -> TrainingConfiguration:
    configuration = TrainingConfiguration(
        manifest="unused",
        model_configuration="unused",
        augmentation_configuration="unused",
        name="toy_gate",
        epochs=300,
        batch_size=5,
        number_of_data_loader_workers=0,
        seed=7,
        mixed_precision="float32",
        gradient_clipping_norm=10.0,
        exponential_moving_average_decay=0.9998,
        evaluation_interval_epochs=10,
        log_interval_steps=20,
        optimizer=OptimizerConfiguration(type="adamw", learning_rate=0.01, weight_decay=0.0),
        schedule=ScheduleConfiguration(
            type="cosine", warmup_epochs=5.0, final_learning_rate_fraction=0.01
        ),
        checkpoint=CheckpointConfiguration(metric_name="validation_loss", metric_mode="minimise"),
    )
    return replace(configuration, **changes)


def run_training(
    run_directory,
    configuration,
    *,
    model_seed=0,
    resume_from=None,
    epoch_start_callbacks=(),
    training_step=toy_training_step,
):
    """Wire everything together the way tools/train.py will."""
    dataset = ToyDataset()
    train_loader = DataLoader(
        dataset,
        batch_size=configuration.batch_size,
        sampler=EpochSeededRandomSampler(len(dataset), configuration.seed),
        num_workers=0,
    )
    # Validation deliberately sees the same ten samples: the gate measures
    # memorisation, which is exactly what a working loop must achieve.
    validation_loader = DataLoader(dataset, batch_size=NUMBER_OF_SAMPLES)

    checkpoint_manager = CheckpointManager(
        run_directory / "checkpoints",
        metric_name=configuration.checkpoint.metric_name,
        metric_mode=configuration.checkpoint.metric_mode,
    )

    with RunLogger(run_directory) as run_logger:
        run_logger.start(configuration.name, {"training": asdict(configuration)})
        trainer = Trainer(
            configuration=configuration,
            model=build_toy_model(model_seed),
            training_step=training_step,
            train_loader=train_loader,
            run_logger=run_logger,
            checkpoint_manager=checkpoint_manager,
            device="cpu",
            evaluate=toy_evaluate,
            validation_loader=validation_loader,
            checkpoint_metadata={"training": asdict(configuration)},
            epoch_start_callbacks=epoch_start_callbacks,
        )
        result = trainer.fit(resume_from=resume_from)

    return trainer, result


class SimulatedCrash(RuntimeError):
    pass


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_toy_model_overfits_to_near_zero_loss(tmp_path):
    _, result = run_training(tmp_path / "run", toy_configuration())

    training_loss = result.final_training_metrics["loss"]
    validation_loss = result.final_evaluation_metrics["validation_loss"]
    print(f"\nfinal training loss {training_loss:.2e}   validation loss {validation_loss:.2e}")

    assert training_loss < 1e-3, f"training loss only reached {training_loss:.2e}"
    assert validation_loss < 1e-3, f"validation loss only reached {validation_loss:.2e}"


def test_run_directory_contains_expected_files(tmp_path):
    run_directory = tmp_path / "run"
    run_training(run_directory, toy_configuration(epochs=20))

    for relative in (
        "configuration.json",
        "metrics.jsonl",
        "console.log",
        "checkpoints/last.pt",
        "checkpoints/best.pt",
    ):
        assert (run_directory / relative).exists(), f"missing {relative}"


def test_metrics_are_logged_with_increasing_steps(tmp_path):
    run_directory = tmp_path / "run"
    run_training(run_directory, toy_configuration(epochs=20))

    records = [
        json.loads(line)
        for line in (run_directory / "metrics.jsonl").read_text().splitlines()
    ]
    steps = [record["step"] for record in records]
    assert steps == sorted(steps), "steps must never go backwards"
    assert any("train/loss" in record for record in records)
    assert any("train/learning_rate" in record for record in records)
    assert any("validation/validation_loss" in record for record in records)


def test_resume_matches_uninterrupted_training(tmp_path):
    configuration = toy_configuration(epochs=40)
    crash_epoch = 20

    uninterrupted_trainer, uninterrupted = run_training(
        tmp_path / "uninterrupted", configuration
    )

    def crash(epoch: int) -> None:
        if epoch == crash_epoch:
            raise SimulatedCrash(f"simulated crash at the start of epoch {epoch + 1}")

    interrupted_directory = tmp_path / "interrupted"
    with pytest.raises(SimulatedCrash):
        run_training(interrupted_directory, configuration, epoch_start_callbacks=[crash])

    resumed_trainer, resumed = run_training(
        interrupted_directory,
        configuration,
        # Different initial weights: everything must come from the checkpoint.
        model_seed=999,
        resume_from=interrupted_directory / "checkpoints" / "last.pt",
    )

    assert resumed.global_step == uninterrupted.global_step
    assert resumed.best_metric == pytest.approx(uninterrupted.best_metric)
    assert resumed.final_training_metrics["loss"] == pytest.approx(
        uninterrupted.final_training_metrics["loss"], rel=1e-5
    )

    for original, restored in zip(
        uninterrupted_trainer.model.parameters(), resumed_trainer.model.parameters()
    ):
        assert torch.allclose(original, restored, atol=1e-6)

    for original, restored in zip(
        uninterrupted_trainer.exponential_moving_average.module.parameters(),
        resumed_trainer.exponential_moving_average.module.parameters(),
    ):
        assert torch.allclose(original, restored, atol=1e-6)


# --------------------------------------------------------------------------
# Failing loudly
# --------------------------------------------------------------------------


def test_non_finite_loss_stops_training(tmp_path):
    def exploding_step(model, batch):
        return {"loss": toy_training_step(model, batch)["loss"] * float("nan")}

    with pytest.raises(TrainerError, match="non-finite loss"):
        run_training(tmp_path / "run", toy_configuration(epochs=10), training_step=exploding_step)


def test_missing_checkpoint_metric_fails_loudly(tmp_path):
    configuration = toy_configuration(
        epochs=10,
        checkpoint=CheckpointConfiguration(metric_name="accuracy", metric_mode="maximise"),
    )
    with pytest.raises(TrainerError, match="accuracy"):
        run_training(tmp_path / "run", configuration)


def test_training_step_must_return_total_loss(tmp_path):
    def unnamed_step(model, batch):
        return {"mean_squared_error": toy_training_step(model, batch)["loss"]}

    with pytest.raises(TrainerError, match="'loss'"):
        run_training(tmp_path / "run", toy_configuration(epochs=10), training_step=unnamed_step)


def test_gradient_accumulation_reduces_optimiser_steps(tmp_path):
    # 10 samples in batches of 2 is 5 batches per epoch. Grouped in twos that
    # is 3 optimiser steps per epoch, the last group being a single batch.
    configuration = toy_configuration(epochs=10, batch_size=2, gradient_accumulation_steps=2)
    _, result = run_training(tmp_path / "run", configuration)
    assert result.global_step == 30


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def test_learning_rate_warms_up_then_decays_to_final_fraction():
    factors = [learning_rate_factor(step, 100, 10, "cosine", 0.05) for step in range(100)]
    assert factors[0] == pytest.approx(0.1)
    assert factors[9] == pytest.approx(1.0)
    assert factors[-1] == pytest.approx(0.05, abs=1e-3)
    after_warmup = factors[10:]
    assert all(earlier >= later for earlier, later in zip(after_warmup, after_warmup[1:]))


def test_sampler_order_depends_only_on_seed_and_epoch():
    first = EpochSeededRandomSampler(10, seed=3)
    second = EpochSeededRandomSampler(10, seed=3)
    first.set_epoch(4)
    second.set_epoch(4)
    assert list(first) == list(second)
    assert sorted(first) == list(range(10))

    second.set_epoch(5)
    assert list(first) != list(second)


def test_weight_decay_skips_biases_and_normalisation():
    optimizer = build_optimizer(
        build_toy_model(), OptimizerConfiguration(type="sgd", weight_decay=0.1)
    )
    decayed, not_decayed = optimizer.param_groups
    assert decayed["weight_decay"] == 0.1
    assert all(parameter.ndim > 1 for parameter in decayed["params"])
    assert not_decayed["weight_decay"] == 0.0
    assert all(parameter.ndim <= 1 for parameter in not_decayed["params"])


def test_moving_average_is_frozen_and_ramps_up():
    average = ModelExponentialMovingAverage(build_toy_model(), decay=0.9998, warmup_updates=2000)
    assert all(not parameter.requires_grad for parameter in average.module.parameters())

    average.updates = 0
    assert average.current_decay() == 0.0
    average.updates = 100_000
    assert average.current_decay() == pytest.approx(0.9998)


def test_moving_average_follows_the_model_early_in_training():
    model = build_toy_model()
    average = ModelExponentialMovingAverage(model, decay=0.9998, warmup_updates=2000)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    average.update(model)

    for averaged, current in zip(average.module.parameters(), model.parameters()):
        assert torch.allclose(averaged, current, atol=1e-2)