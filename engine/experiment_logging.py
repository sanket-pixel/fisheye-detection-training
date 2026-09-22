# engine/experiment_logging.py
"""
Run logging: console output, a local metrics record, and optional
experiment tracking.

Three sinks behind one interface:

  console         always on; human-readable progress
  metrics.jsonl   always on; the local record of every logged value,
                  written to the run directory. Survives tracker outages
                  and needs no account or network.
  tracker         optional; Weights & Biases today. Switching to MLflow
                  later means adding one class here and nothing else.

Provenance is passed to the tracker as structured configuration rather
than encoded in the run name, so lineage is a queryable field.

Named experiment_logging to avoid shadowing the standard library logging
module, and to avoid confusion with object tracking.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Protocol

LOGGER_NAME = "training"
METRICS_FILE_NAME = "metrics.jsonl"
CONFIGURATION_FILE_NAME = "configuration.json"
CONSOLE_LOG_FILE_NAME = "console.log"
CONSOLE_HANDLER_NAME = "training_console"

def _to_float(value: Any) -> float:
    """Accept Python numbers and zero-dimensional tensors without importing torch."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


# --------------------------------------------------------------------------
# Console
# --------------------------------------------------------------------------


def configure_console_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not any(handler.get_name() == CONSOLE_HANDLER_NAME for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name(CONSOLE_HANDLER_NAME)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)

    return logger

# --------------------------------------------------------------------------
# Experiment trackers
# --------------------------------------------------------------------------


class ExperimentTracker(Protocol):
    run_url: str | None

    def start(
        self, run_name: str, configuration: dict[str, Any], directory: Path
    ) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def finish(self) -> None: ...


class NullTracker:
    """Tracker that records nothing. The default for debugging runs."""

    run_url: str | None = None

    def start(self, run_name: str, configuration: dict[str, Any], directory: Path) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        pass

    def finish(self) -> None:
        pass


class WeightsAndBiasesTracker:
    """
    Weights & Biases backend.

    The library is imported only when a run starts, so it is not a hard
    dependency of the engine. Steps passed to log_metrics must increase
    monotonically — Weights & Biases silently drops out-of-order steps.
    """

    VALID_MODES = {"online", "offline", "disabled"}

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        mode: str = "online",
        tags: list[str] | None = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(self.VALID_MODES)}, got {mode!r}"
            )
        self.project = project
        self.entity = entity
        self.mode = mode
        self.tags = tags or []
        self._run: Any = None

    @property
    def run_url(self) -> str | None:
        return getattr(self._run, "url", None) if self._run is not None else None

    def start(self, run_name: str, configuration: dict[str, Any], directory: Path) -> None:
        import wandb

        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            config=configuration,
            dir=str(directory),
            mode=self.mode,
            tags=self.tags,
        )

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def create_tracker(kind: str, **options: Any) -> ExperimentTracker:
    """Construct a tracker from configuration."""
    if kind == "none":
        return NullTracker()
    if kind == "weights_and_biases":
        return WeightsAndBiasesTracker(**options)
    raise ValueError(
        f"unknown tracker {kind!r}; expected 'none' or 'weights_and_biases'"
    )


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


class MetricAverager:
    """
    Running mean of per-batch values for epoch-level reporting.

    Weighted by sample count, so a small final batch does not count as much
    as a full one.
    """

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, metrics: dict[str, Any], count: int = 1) -> None:
        for name, value in metrics.items():
            self._totals[name] = self._totals.get(name, 0.0) + _to_float(value) * count
            self._counts[name] = self._counts.get(name, 0) + count

    def compute(self) -> dict[str, float]:
        return {name: self._totals[name] / self._counts[name] for name in self._totals}

    def reset(self) -> None:
        self._totals.clear()
        self._counts.clear()


# --------------------------------------------------------------------------
# The logger the trainer talks to
# --------------------------------------------------------------------------


class RunLogger:
    """
    Single entry point for everything a run reports.

    Use as a context manager so the tracker is always finished, including
    when training crashes — otherwise Weights & Biases leaves the run
    marked as running indefinitely.
    """

    def __init__(
            self,
            run_directory: str | Path,
            tracker: ExperimentTracker | None = None,
            console: logging.Logger | None = None,
    ) -> None:
        self.run_directory = Path(run_directory)
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_directory / METRICS_FILE_NAME
        self.console_log_path = self.run_directory / CONSOLE_LOG_FILE_NAME
        self.tracker = tracker or NullTracker()
        self.console = console or configure_console_logging()
        self._start_time: float | None = None

        # Everything printed during the run is also kept with the run, so a
        # warning at epoch 63 survives the terminal closing.
        self._file_handler: logging.FileHandler | None = logging.FileHandler(
            self.console_log_path
        )
        self._file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        self.console.addHandler(self._file_handler)

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        if exception is not None:
            self.console.error(f"run failed: {exception_type.__name__}: {exception}")
        self.finish()

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def start(self, run_name: str, configuration: dict[str, Any]) -> None:
        self._start_time = time.monotonic()

        (self.run_directory / CONFIGURATION_FILE_NAME).write_text(
            json.dumps(configuration, indent=2, default=str)
        )
        self.tracker.start(run_name, configuration, self.run_directory)

        self.console.info(f"run        {run_name}")
        self.console.info(f"directory  {self.run_directory}")
        if self.tracker.run_url:
            self.console.info(f"tracker    {self.tracker.run_url}")

    def log_metrics(
        self, metrics: dict[str, Any], step: int, epoch: int | None = None
    ) -> None:
        values = {name: _to_float(value) for name, value in metrics.items()}

        record = {
            "step": step,
            "epoch": epoch,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            **values,
        }
        with open(self.metrics_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")

        self.tracker.log_metrics(values, step)

    def log_epoch_summary(
        self, epoch: int, total_epochs: int, metrics: dict[str, Any]
    ) -> None:
        formatted = "  ".join(
            f"{name} {_to_float(value):.4f}" for name, value in sorted(metrics.items())
        )
        self.console.info(f"epoch {epoch + 1}/{total_epochs}  {formatted}")

    def info(self, message: str) -> None:
        self.console.info(message)

    def warning(self, message: str) -> None:
        self.console.warning(message)

    def finish(self) -> None:
        self.tracker.finish()
        if self._start_time is not None:
            self.console.info(f"finished in {self.elapsed_seconds / 60:.1f} min")
            self._start_time = None
        self._close_file_handler()

    def _close_file_handler(self) -> None:
        """Detach so a later run in the same process does not write into this log."""
        if self._file_handler is not None:
            self.console.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None