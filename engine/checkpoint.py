# engine/checkpoint.py
"""
Checkpoint saving, loading, and resumption.

A checkpoint captures everything needed to continue a run exactly where it
stopped: model weights, optimiser state, learning rate schedule position,
gradient scaler state, the exponential moving average shadow weights, and
the epoch counter. Omitting any of these produces a resume that silently
diverges from an uninterrupted run — most commonly by resetting the
optimiser's momentum buffers or restarting the schedule from step zero.

Provenance is written alongside the weights so that a checkpoint file found
on disk months later can be traced back to the code, config, and data that
produced it.

This module is task-agnostic: it treats the model as an object with a state
dictionary and nothing more.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_SUFFIX = ".pt"
LAST_CHECKPOINT_NAME = f"last{CHECKPOINT_SUFFIX}"
BEST_CHECKPOINT_NAME = f"best{CHECKPOINT_SUFFIX}"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be written or is unusable."""


@dataclass
class ResumeState:
    """What a resumed run needs to know about where it left off."""

    epoch: int
    global_step: int
    best_metric: float | None
    metrics: dict[str, float]


def _state_dict_of(module: Any) -> dict[str, Any] | None:
    """
    Extract a state dictionary, unwrapping DistributedDataParallel.

    Saving the wrapped state dictionary would prefix every key with
    'module.', producing a checkpoint that cannot be loaded into a
    single-device model without string surgery.
    """
    if module is None:
        return None
    inner = getattr(module, "module", module)
    return inner.state_dict()


def _load_into(module: Any, state: dict[str, Any] | None, name: str, strict: bool) -> None:
    if module is None or state is None:
        return
    inner = getattr(module, "module", module)
    try:
        inner.load_state_dict(state, strict=strict)
    except TypeError:
        # Optimisers, schedulers, and scalers take no `strict` argument.
        inner.load_state_dict(state)
    except Exception as error:  # noqa: BLE001 — surface which component failed
        raise CheckpointError(f"failed to load {name} state: {error}") from error


class CheckpointManager:
    """
    Writes and reads checkpoints for one run.

    Two checkpoints are always maintained: `last` for resumption, and `best`
    for the highest-scoring model seen so far. Keeping both means an
    interrupted run resumes correctly even if its most recent epoch was
    worse than an earlier one.
    """

    def __init__(
        self,
        directory: str | Path,
        metric_name: str = "fitness",
        metric_mode: str = "maximise",
        keep_last_n_epochs: int = 0,
    ) -> None:
        if metric_mode not in {"maximise", "minimise"}:
            raise ValueError(
                f"metric_mode must be 'maximise' or 'minimise', got {metric_mode!r}"
            )

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.keep_last_n_epochs = keep_last_n_epochs
        self.best_metric: float | None = None

    # -- paths ---------------------------------------------------------

    @property
    def last_path(self) -> Path:
        return self.directory / LAST_CHECKPOINT_NAME

    @property
    def best_path(self) -> Path:
        return self.directory / BEST_CHECKPOINT_NAME

    def epoch_path(self, epoch: int) -> Path:
        return self.directory / f"epoch_{epoch:04d}{CHECKPOINT_SUFFIX}"

    # -- comparison ----------------------------------------------------

    def is_better(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.metric_mode == "maximise":
            return metric > self.best_metric
        return metric < self.best_metric

    # -- writing -------------------------------------------------------

    def save(
        self,
        *,
        model: Any,
        epoch: int,
        global_step: int,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        model_exponential_moving_average: Any = None,
        metrics: dict[str, float] | None = None,
        provenance: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write `last`, and `best` when the tracked metric has improved.

        Returns the path of the checkpoint written for this epoch.
        """
        metrics = metrics or {}
        payload: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "model": _state_dict_of(model),
            "optimizer": _state_dict_of(optimizer),
            "scheduler": _state_dict_of(scheduler),
            "scaler": _state_dict_of(scaler),
            "model_exponential_moving_average": _state_dict_of(
                model_exponential_moving_average
            ),
            "metrics": metrics,
            "provenance": provenance or {},
            "extra": extra or {},
        }

        # Write to a temporary path and move into place, so an interrupted
        # write cannot leave a truncated checkpoint that fails on resume.
        temporary_path = self.last_path.with_suffix(".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(self.last_path)

        written = self.last_path

        if self.keep_last_n_epochs > 0:
            epoch_path = self.epoch_path(epoch)
            shutil.copyfile(self.last_path, epoch_path)
            self._prune_epoch_checkpoints(epoch)
            written = epoch_path

        tracked = metrics.get(self.metric_name)
        if tracked is not None and self.is_better(tracked):
            self.best_metric = tracked
            payload["best_metric"] = tracked
            temporary_best = self.best_path.with_suffix(".tmp")
            torch.save(payload, temporary_best)
            temporary_best.replace(self.best_path)

        return written

    def _prune_epoch_checkpoints(self, current_epoch: int) -> None:
        epoch_checkpoints = sorted(
            self.directory.glob(f"epoch_*{CHECKPOINT_SUFFIX}")
        )
        excess = len(epoch_checkpoints) - self.keep_last_n_epochs
        for path in epoch_checkpoints[:excess]:
            path.unlink(missing_ok=True)

    # -- reading -------------------------------------------------------

    @staticmethod
    def load(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise CheckpointError(f"checkpoint not found: {path}")
        # weights_only=False: checkpoints carry provenance dictionaries, not
        # just tensors. Only load checkpoints you produced.
        return torch.load(path, map_location=map_location, weights_only=False)

    def resume(
        self,
        *,
        model: Any,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        model_exponential_moving_average: Any = None,
        path: str | Path | None = None,
        map_location: str = "cpu",
        strict: bool = True,
    ) -> ResumeState:
        """Restore every component in place and report where the run stopped."""
        checkpoint_path = Path(path) if path is not None else self.last_path
        payload = self.load(checkpoint_path, map_location=map_location)

        _load_into(model, payload.get("model"), "model", strict)
        _load_into(optimizer, payload.get("optimizer"), "optimizer", strict)
        _load_into(scheduler, payload.get("scheduler"), "scheduler", strict)
        _load_into(scaler, payload.get("scaler"), "scaler", strict)
        _load_into(
            model_exponential_moving_average,
            payload.get("model_exponential_moving_average"),
            "exponential moving average",
            strict,
        )

        self.best_metric = payload.get("best_metric", self.best_metric)

        return ResumeState(
            epoch=payload.get("epoch", 0),
            global_step=payload.get("global_step", 0),
            best_metric=self.best_metric,
            metrics=payload.get("metrics", {}),
        )


def load_weights_only(
    model: Any,
    path: str | Path,
    map_location: str = "cpu",
    strict: bool = True,
    prefer_exponential_moving_average: bool = True,
) -> None:
    """
    Load weights for inference or fine-tuning, ignoring training state.

    Prefers the exponential moving average weights when present, since those
    are what should be evaluated and exported — the raw weights are the last
    optimiser step, which is noisier.
    """
    payload = CheckpointManager.load(path, map_location=map_location)

    state = None
    if prefer_exponential_moving_average:
        state = payload.get("model_exponential_moving_average")
    if state is None:
        state = payload.get("model")
    if state is None:
        raise CheckpointError(f"{path} contains no model weights")

    _load_into(model, state, "model", strict)


def describe(path: str | Path) -> dict[str, Any]:
    """Summarise a checkpoint without loading tensors into a model."""
    payload = CheckpointManager.load(path)
    return {
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "metrics": payload.get("metrics", {}),
        "best_metric": payload.get("best_metric"),
        "provenance": payload.get("provenance", {}),
        "has_exponential_moving_average": payload.get(
            "model_exponential_moving_average"
        )
        is not None,
    }