# engine/exponential_moving_average.py
"""
Exponential moving average of model weights.

Keeps a shadow copy of the model whose weights are a running average of the
training weights. It is evaluated and exported instead of the raw weights:
the raw weights are wherever the last optimiser step happened to land, while
the average smooths out step-to-step noise — for detectors typically worth a
point or more of mAP.

The decay ramps up from zero, following YOLOX:

    decay(updates) = decay * (1 - exp(-updates / warmup_updates))

so early in training, when weights change fastest, the average tracks the
model closely instead of staying anchored to the random initialisation.

The averaged model is exposed as `.module`. The checkpoint manager unwraps
`.module` when saving, so a checkpoint's moving-average entry is a plain
model state dictionary, loadable directly into the architecture for
evaluation or export. The update counter is stored separately by the trainer.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model from a DistributedDataParallel wrapper."""
    return getattr(model, "module", model)


class ModelExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float, warmup_updates: int = 2000) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        if warmup_updates <= 0:
            raise ValueError(f"warmup_updates must be positive, got {warmup_updates}")

        self.module = copy.deepcopy(unwrap_model(model)).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

        self.decay = decay
        self.warmup_updates = warmup_updates
        self.updates = 0

    def current_decay(self) -> float:
        return self.decay * (1.0 - math.exp(-self.updates / self.warmup_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.current_decay()
        source = unwrap_model(model).state_dict()

        # state_dict() returns the live tensors, so in-place updates modify
        # the averaged module directly.
        for name, averaged in self.module.state_dict().items():
            current = source[name].detach()
            if averaged.dtype.is_floating_point:
                averaged.mul_(decay).add_(current.to(averaged.dtype), alpha=1.0 - decay)
            else:
                # Integer buffers such as batch-norm's num_batches_tracked
                # are copied, not averaged.
                averaged.copy_(current)