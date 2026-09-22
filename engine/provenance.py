# engine/provenance.py
"""
Run provenance: what code produced a run, on which machine, and when.

Responsibilities are split with engine/configuration.py:

    configuration   what the run was told to do, and on which data:
                    settings, overrides, manifest hash, configuration hashes
    provenance      what code ran it, where, and when:
                    git commit and state, library versions, hardware

Both are written together to runs/<run_id>/configuration.json and embedded
in every checkpoint, so a model file found on its own can still be traced
back to its code, settings, and data.

Torch is imported lazily and only to read versions and device information,
so this module stays cheap to import and works where torch is absent.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.paths import REPO_ROOT


class ProvenanceError(RuntimeError):
    """Raised when provenance cannot be established or a run must not start."""


# --------------------------------------------------------------------------
# Git
# --------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise ProvenanceError(
            f"`git {' '.join(arguments)}` failed; is {REPO_ROOT} a git repository?"
        ) from error


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def git_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def git_dirty_files() -> list[str]:
    """Files that make the working tree dirty: modified, staged, or untracked."""
    output = _git("status", "--porcelain")
    return [line[3:] for line in output.splitlines()] if output else []


def git_is_dirty() -> bool:
    return bool(git_dirty_files())


def require_clean_tree(allow_dirty: bool = False) -> None:
    """Refuse to start an unreproducible run unless explicitly overridden."""
    if allow_dirty:
        return
    files = git_dirty_files()
    if not files:
        return
    preview = "\n  ".join(files[:10])
    more = f"\n  ... and {len(files) - 10} more" if len(files) > 10 else ""
    raise ProvenanceError(
        "Working tree is dirty — this run would not be reproducible.\n"
        f"  {preview}{more}\n"
        "Commit your changes, or allow a dirty tree explicitly for a throwaway run."
    )


# --------------------------------------------------------------------------
# Hashing — used by configuration, dataset builds, and checkpoints
# --------------------------------------------------------------------------


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_dict(dictionary: dict[str, Any]) -> str:
    """SHA-256 of a dictionary's content, independent of key order."""
    payload = json.dumps(dictionary, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvironmentRecord:
    """
    Software and hardware a run executed on.

    Worth recording because results can shift by fractions of a point across
    torch or CUDA versions through nondeterministic kernels, and "it got
    worse after the upgrade" is only diagnosable if the versions were kept.
    """

    python_version: str
    platform: str
    hostname: str
    torch_version: str | None
    cuda_version: str | None
    cudnn_version: int | None
    device_name: str
    device_count: int


def capture_environment() -> EnvironmentRecord:
    torch_version: str | None = None
    cuda_version: str | None = None
    cudnn_version: int | None = None
    device_name = "cpu"
    device_count = 0

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        if torch.backends.cudnn.is_available():
            cudnn_version = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)

    return EnvironmentRecord(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        hostname=socket.gethostname(),
        torch_version=torch_version,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        device_name=device_name,
        device_count=device_count,
    )


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    started_at: str
    git_commit: str
    git_branch: str
    git_dirty: bool
    git_dirty_files: tuple[str, ...]
    environment: EnvironmentRecord

    def to_dictionary(self) -> dict[str, Any]:
        return asdict(self)


def _sanitise_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")


def build_provenance(label: str | None = None) -> RunProvenance:
    """
    Collect provenance for a run. Call once, at startup.

    The run id is `<UTC timestamp>_<short commit>[_dirty][_<label>]`, so a
    run directory's name alone says when it ran, from which code, whether
    that code was committed, and what the run was called.
    """
    started_at = datetime.now(timezone.utc)
    commit = git_commit()
    dirty_files = tuple(git_dirty_files())
    dirty = bool(dirty_files)

    parts = [started_at.strftime("%Y%m%d_%H%M%S"), commit[:7]]
    if dirty:
        parts.append("dirty")
    if label:
        sanitised = _sanitise_label(label)
        if sanitised:
            parts.append(sanitised)

    return RunProvenance(
        run_id="_".join(parts),
        started_at=started_at.isoformat(),
        git_commit=commit,
        git_branch=git_branch(),
        git_dirty=dirty,
        git_dirty_files=dirty_files,
        environment=capture_environment(),
    )