# tests/test_checkpoint.py
"""Tests for checkpoint saving, resumption, and metric tracking."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from engine.checkpoint import (
    CheckpointError,
    CheckpointManager,
    describe,
    load_weights_only,
)


@pytest.fixture
def model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


@pytest.fixture
def manager(tmp_path) -> CheckpointManager:
    return CheckpointManager(tmp_path / "checkpoints", metric_name="fitness")


def test_save_writes_last(manager, model):
    manager.save(model=model, epoch=0, global_step=10, metrics={"fitness": 0.5})
    assert manager.last_path.exists()


def test_best_is_written_on_improvement(manager, model):
    manager.save(model=model, epoch=0, global_step=1, metrics={"fitness": 0.5})
    assert manager.best_metric == 0.5

    manager.save(model=model, epoch=1, global_step=2, metrics={"fitness": 0.7})
    assert manager.best_metric == 0.7

    manager.save(model=model, epoch=2, global_step=3, metrics={"fitness": 0.6})
    assert manager.best_metric == 0.7, "best must not regress"


def test_minimise_mode_prefers_lower(tmp_path, model):
    manager = CheckpointManager(
        tmp_path / "checkpoints", metric_name="loss", metric_mode="minimise"
    )
    manager.save(model=model, epoch=0, global_step=1, metrics={"loss": 1.0})
    manager.save(model=model, epoch=1, global_step=2, metrics={"loss": 0.4})
    assert manager.best_metric == 0.4


def test_invalid_metric_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="metric_mode"):
        CheckpointManager(tmp_path, metric_mode="sideways")


def test_resume_restores_weights_and_position(manager, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Take a step so weights and optimiser state are non-trivial
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    optimizer.step()

    manager.save(
        model=model,
        optimizer=optimizer,
        epoch=7,
        global_step=700,
        metrics={"fitness": 0.42},
    )

    restored_model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)

    state = manager.resume(model=restored_model, optimizer=restored_optimizer)

    assert state.epoch == 7
    assert state.global_step == 700
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.allclose(original, restored)


def test_resume_from_missing_file_raises(manager, model):
    with pytest.raises(CheckpointError, match="not found"):
        manager.resume(model=model, path=manager.directory / "absent.pt")


def test_epoch_checkpoints_are_pruned(tmp_path, model):
    manager = CheckpointManager(tmp_path / "checkpoints", keep_last_n_epochs=2)
    for epoch in range(5):
        manager.save(
            model=model, epoch=epoch, global_step=epoch, metrics={"fitness": 0.1}
        )

    remaining = sorted(p.name for p in manager.directory.glob("epoch_*.pt"))
    assert remaining == ["epoch_0003.pt", "epoch_0004.pt"]


def test_load_weights_only_prefers_moving_average(manager, model):
    averaged = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with torch.no_grad():
        for parameter in averaged.parameters():
            parameter.fill_(0.123)

    manager.save(
        model=model,
        model_exponential_moving_average=averaged,
        epoch=0,
        global_step=1,
        metrics={"fitness": 0.5},
    )

    target = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    load_weights_only(target, manager.last_path)

    for parameter in target.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 0.123))


def test_provenance_survives_the_round_trip(manager, model):
    provenance = {"git_commit": "abc1234", "manifest_hash": "deadbeef"}
    manager.save(
        model=model,
        epoch=0,
        global_step=1,
        metrics={"fitness": 0.5},
        provenance=provenance,
    )
    summary = describe(manager.last_path)
    assert summary["provenance"] == provenance
    assert summary["epoch"] == 0