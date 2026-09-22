# engine/trainer.py
"""
The training loop, shared by every project.

The trainer knows how to optimise, not what it is optimising. The project
supplies:

    model           an nn.Module
    training_step   (model, batch) -> dict of loss tensors. "loss" is the
                    total that is backpropagated; other entries are logged.
    evaluate        (model, loader, device) -> dict of float metrics

A batch is whatever the project's collate function returns. Every tensor in
it is moved to the device and the batch is otherwise passed through
untouched, so detection, tracking, depth, and fusion inputs all use the same
loop.

What the trainer owns, driven by TrainingConfiguration:
  - optimiser construction, with weight decay on weights but not on biases
    or normalisation parameters
  - linear warmup, then a cosine, linear, or constant schedule, stepped once
    per optimiser step
  - mixed precision: bfloat16, float16 with loss scaling, or float32
  - gradient accumulation and clipping
  - an exponential moving average of the weights, which is what gets
    evaluated and exported
  - periodic evaluation, checkpointing, and exact resumption, including
    random number generator state
  - failing loudly on a non-finite loss or gradient
"""
from __future__ import annotations

import contextlib
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from engine.checkpoint import CheckpointManager
from engine.configuration import (
    OptimizerConfiguration,
    ScheduleConfiguration,
    TrainingConfiguration,
)
from engine.experiment_logging import MetricAverager, RunLogger
from engine.exponential_moving_average import ModelExponentialMovingAverage

TrainingStep = Callable[[nn.Module, Any], dict[str, torch.Tensor]]
EvaluationFunction = Callable[[nn.Module, DataLoader, torch.device], dict[str, float]]
EpochCallback = Callable[[int], None]

TOTAL_LOSS_KEY = "loss"
EXPONENTIAL_MOVING_AVERAGE_WARMUP_UPDATES = 2000


class TrainerError(RuntimeError):
    """Raised when training cannot continue correctly."""


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class EpochSeededRandomSampler(Sampler[int]):
    """
    Shuffles with a permutation determined only by (seed, epoch).

    A standard shuffling DataLoader draws from a generator whose state
    depends on everything that happened before, so a resumed run sees a
    different order than an uninterrupted one. Here the trainer calls
    set_epoch() before each epoch, and the order is identical either way.
    """

    def __init__(self, dataset_length: int, seed: int) -> None:
        self.dataset_length = dataset_length
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.dataset_length, generator=generator).tolist())

    def __len__(self) -> int:
        return self.dataset_length


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def move_to_device(value: Any, device: torch.device) -> Any:
    """Move every tensor in a nested batch to the device; leave the rest alone."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def build_optimizer(
    model: nn.Module, configuration: OptimizerConfiguration
) -> torch.optim.Optimizer:
    """
    Weight decay applies to weight matrices and kernels only. Biases and
    normalisation parameters are one-dimensional, and decaying them pulls
    them towards zero for no benefit — the standard YOLOX and ResNet recipe.
    """
    decayed: list[nn.Parameter] = []
    not_decayed: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (not_decayed if parameter.ndim <= 1 else decayed).append(parameter)

    if not decayed and not not_decayed:
        raise TrainerError("model has no trainable parameters")

    groups = [
        group
        for group in (
            {"params": decayed, "weight_decay": configuration.weight_decay},
            {"params": not_decayed, "weight_decay": 0.0},
        )
        if group["params"]
    ]

    if configuration.type == "sgd":
        return torch.optim.SGD(
            groups,
            lr=configuration.learning_rate,
            momentum=configuration.momentum,
            nesterov=configuration.nesterov and configuration.momentum > 0,
        )
    if configuration.type == "adamw":
        return torch.optim.AdamW(groups, lr=configuration.learning_rate)
    raise TrainerError(f"unsupported optimizer {configuration.type!r}")


def learning_rate_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
    schedule_type: str,
    final_fraction: float,
) -> float:
    """Multiplier on the base learning rate at a given optimiser step."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule_type == "constant":
        return 1.0

    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))

    if schedule_type == "cosine":
        return final_fraction + (1.0 - final_fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule_type == "linear":
        return 1.0 - (1.0 - final_fraction) * progress
    raise TrainerError(f"unsupported schedule {schedule_type!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    configuration: ScheduleConfiguration,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_factor(
            step,
            total_steps,
            warmup_steps,
            configuration.type,
            configuration.final_learning_rate_fraction,
        ),
    )


def resolve_precision(mixed_precision: str, device: torch.device) -> torch.dtype | None:
    """The autocast dtype, or None for full float32."""
    if mixed_precision == "float32":
        return None
    if mixed_precision == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise TrainerError(
                "bfloat16 is not supported on this GPU; set training.mixed_precision=float16"
            )
        return torch.bfloat16
    if mixed_precision == "float16":
        if device.type != "cuda":
            raise TrainerError(
                "float16 mixed precision needs a CUDA device; use bfloat16 or float32"
            )
        return torch.float16
    raise TrainerError(f"unsupported mixed precision {mixed_precision!r}")


# --------------------------------------------------------------------------
# The trainer
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingResult:
    epochs_completed: int
    global_step: int
    best_metric: float | None
    final_training_metrics: dict[str, float]
    final_evaluation_metrics: dict[str, float]


class Trainer:
    def __init__(
        self,
        *,
        configuration: TrainingConfiguration,
        model: nn.Module,
        training_step: TrainingStep,
        train_loader: DataLoader,
        run_logger: RunLogger,
        checkpoint_manager: CheckpointManager,
        device: torch.device | str,
        evaluate: EvaluationFunction | None = None,
        validation_loader: DataLoader | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
        epoch_start_callbacks: Iterable[EpochCallback] = (),
    ) -> None:
        if (evaluate is None) != (validation_loader is None):
            raise TrainerError("evaluate and validation_loader must be given together")
        if len(train_loader) == 0:
            raise TrainerError("train_loader yields no batches")

        self.configuration = configuration
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.training_step = training_step
        self.train_loader = train_loader
        self.evaluate = evaluate
        self.validation_loader = validation_loader
        self.run_logger = run_logger
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.epoch_start_callbacks = list(epoch_start_callbacks)

        self.optimizer = build_optimizer(self.model, configuration.optimizer)

        accumulation = configuration.gradient_accumulation_steps
        self.steps_per_epoch = math.ceil(len(train_loader) / accumulation)
        self.total_steps = self.steps_per_epoch * configuration.epochs
        self.warmup_steps = round(configuration.schedule.warmup_epochs * self.steps_per_epoch)
        self.scheduler = build_scheduler(
            self.optimizer, configuration.schedule, self.total_steps, self.warmup_steps
        )

        self.precision_dtype = resolve_precision(configuration.mixed_precision, self.device)
        # Loss scaling is only needed for float16; a disabled scaler is a
        # transparent pass-through, which keeps the step code uniform.
        self.scaler = torch.amp.GradScaler(
            device=self.device.type, enabled=self.precision_dtype is torch.float16
        )

        decay = configuration.exponential_moving_average_decay
        self.exponential_moving_average = (
            ModelExponentialMovingAverage(
                self.model, decay, EXPONENTIAL_MOVING_AVERAGE_WARMUP_UPDATES
            )
            if decay is not None
            else None
        )

        self.global_step = 0
        self.start_epoch = 0

    # -- public ------------------------------------------------------------

    def fit(self, resume_from: str | Path | None = None) -> TrainingResult:
        seed_everything(self.configuration.seed)
        if resume_from is not None:
            # After seeding, so the restored random state takes precedence.
            self._resume(Path(resume_from))

        self._log_setup()

        total_epochs = self.configuration.epochs
        training_metrics: dict[str, float] = {}
        evaluation_metrics: dict[str, float] = {}

        for epoch in range(self.start_epoch, total_epochs):
            epoch_started = time.monotonic()

            for callback in self.epoch_start_callbacks:
                callback(epoch)
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            training_metrics = self._train_one_epoch(epoch)

            evaluation_metrics = {}
            if self._should_evaluate(epoch):
                evaluation_metrics = self._evaluate()
                self._require_checkpoint_metric(evaluation_metrics)

            self.run_logger.log_metrics(
                {
                    **{f"train_epoch/{name}": value for name, value in training_metrics.items()},
                    **{f"validation/{name}": value for name, value in evaluation_metrics.items()},
                },
                step=self.global_step,
                epoch=epoch,
            )
            self.run_logger.log_epoch_summary(
                epoch,
                total_epochs,
                {
                    **{f"train_{name}": value for name, value in training_metrics.items()},
                    **evaluation_metrics,
                    "seconds": time.monotonic() - epoch_started,
                },
            )

            self._save_checkpoint(epoch, evaluation_metrics)

        return TrainingResult(
            epochs_completed=total_epochs,
            global_step=self.global_step,
            best_metric=self.checkpoint_manager.best_metric,
            final_training_metrics=training_metrics,
            final_evaluation_metrics=evaluation_metrics,
        )

    # -- one epoch ---------------------------------------------------------

    def _autocast(self):
        if self.precision_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.precision_dtype)

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        averager = MetricAverager()
        accumulation = self.configuration.gradient_accumulation_steps
        number_of_batches = len(self.train_loader)

        self.optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(self.train_loader):
            batch = move_to_device(batch, self.device)

            with self._autocast():
                losses = self.training_step(self.model, batch)

            loss = self._validated_total_loss(losses, epoch)
            self.scaler.scale(loss / accumulation).backward()

            detached = {name: value.detach().float() for name, value in losses.items()}
            averager.update(detached)

            # The last group of an epoch may be shorter than `accumulation`;
            # it is stepped anyway so no gradient is carried across epochs.
            is_step_boundary = (
                (batch_index + 1) % accumulation == 0 or batch_index + 1 == number_of_batches
            )
            if not is_step_boundary:
                continue

            gradient_norm = self._optimizer_step()
            self.global_step += 1

            if self.global_step % self.configuration.log_interval_steps == 0:
                self.run_logger.log_metrics(
                    {
                        **{f"train/{name}": value for name, value in detached.items()},
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train/gradient_norm": gradient_norm,
                    },
                    step=self.global_step,
                    epoch=epoch,
                )

        return averager.compute()

    def _validated_total_loss(self, losses: dict[str, torch.Tensor], epoch: int) -> torch.Tensor:
        if TOTAL_LOSS_KEY not in losses:
            raise TrainerError(
                f"training_step must return a dictionary containing {TOTAL_LOSS_KEY!r}; "
                f"got keys {sorted(losses)}"
            )
        loss = losses[TOTAL_LOSS_KEY]
        if loss.ndim != 0:
            raise TrainerError(f"the total loss must be a scalar, got shape {tuple(loss.shape)}")
        if not torch.isfinite(loss):
            raise TrainerError(
                f"non-finite loss {loss.item()} at epoch {epoch + 1}, "
                f"after optimiser step {self.global_step}"
            )
        return loss

    def _optimizer_step(self) -> float:
        self.scaler.unscale_(self.optimizer)

        parameters = [p for p in self.model.parameters() if p.grad is not None]
        maximum_norm = self.configuration.gradient_clipping_norm
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, maximum_norm if maximum_norm is not None else float("inf")
        )

        # Under float16 an occasional infinite gradient is expected: the
        # scaler skips that step and lowers its scale. Elsewhere it means the
        # weights are about to be corrupted.
        if not self.scaler.is_enabled() and not torch.isfinite(gradient_norm):
            raise TrainerError(f"non-finite gradient norm at optimiser step {self.global_step}")

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        if self.exponential_moving_average is not None:
            self.exponential_moving_average.update(self.model)

        return float(gradient_norm)

    # -- evaluation --------------------------------------------------------

    def _should_evaluate(self, epoch: int) -> bool:
        if self.evaluate is None:
            return False
        is_last = epoch == self.configuration.epochs - 1
        return is_last or (epoch + 1) % self.configuration.evaluation_interval_epochs == 0

    @torch.no_grad()
    def _evaluate(self) -> dict[str, float]:
        """Evaluates the moving-average weights when they exist — those are what ships."""
        target = (
            self.exponential_moving_average.module
            if self.exponential_moving_average is not None
            else self.model
        )
        target.eval()
        metrics = self.evaluate(target, self.validation_loader, self.device)
        return {name: float(value) for name, value in metrics.items()}

    def _require_checkpoint_metric(self, metrics: dict[str, float]) -> None:
        name = self.configuration.checkpoint.metric_name
        if name not in metrics:
            raise TrainerError(
                f"training.checkpoint.metric_name is {name!r}, but evaluate returned "
                f"{sorted(metrics)}. Set it to one of those."
            )

    # -- checkpointing -----------------------------------------------------

    def _checkpointed_components(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            # A disabled scaler has an empty state and refuses to load one.
            "scaler": self.scaler if self.scaler.is_enabled() else None,
            "model_exponential_moving_average": self.exponential_moving_average,
        }

    def _save_checkpoint(self, epoch: int, evaluation_metrics: dict[str, float]) -> None:
        self.checkpoint_manager.save(
            **self._checkpointed_components(),
            epoch=epoch,
            global_step=self.global_step,
            metrics=evaluation_metrics,
            provenance=self.checkpoint_metadata,
            extra={
                "random_state": capture_random_state(),
                "exponential_moving_average_updates": (
                    self.exponential_moving_average.updates
                    if self.exponential_moving_average is not None
                    else 0
                ),
            },
        )

    def _resume(self, path: Path) -> None:
        state = self.checkpoint_manager.resume(
            **self._checkpointed_components(),
            path=path,
            map_location=str(self.device),
        )

        # Random states must be loaded onto the CPU; torch refuses to restore
        # a generator state that lives on a GPU.
        extra = CheckpointManager.load(path, map_location="cpu").get("extra", {})
        if self.exponential_moving_average is not None:
            self.exponential_moving_average.updates = int(
                extra.get("exponential_moving_average_updates", 0)
            )
        if "random_state" in extra:
            restore_random_state(extra["random_state"])

        self.global_step = state.global_step
        self.start_epoch = state.epoch + 1
        self.run_logger.info(
            f"resumed from {path}: continuing at epoch {self.start_epoch + 1}, "
            f"optimiser step {self.global_step}"
        )

    # -- reporting ---------------------------------------------------------

    def _log_setup(self) -> None:
        total_parameters = sum(p.numel() for p in self.model.parameters())
        trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        configuration = self.configuration
        self.run_logger.info(
            f"parameters {total_parameters:,} ({trainable_parameters:,} trainable)"
        )
        self.run_logger.info(
            f"device {self.device}  precision {configuration.mixed_precision}  "
            f"effective batch {configuration.effective_batch_size}"
        )
        self.run_logger.info(
            f"steps per epoch {self.steps_per_epoch}  total {self.total_steps}  "
            f"warmup {self.warmup_steps}"
        )
        self.run_logger.info(
            "moving average "
            + (
                f"on, decay {configuration.exponential_moving_average_decay}"
                if self.exponential_moving_average is not None
                else "off"
            )
        )