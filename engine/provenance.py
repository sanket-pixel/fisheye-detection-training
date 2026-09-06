# engine/provenance.py
"""
Run provenance: everything needed to reconstruct a training run later.

A run is identified by (code, data, config). If any of those can't be
pinned, the resulting weights are scratch output, not a model — so this
module is what the trainer consults before spending GPU time.

No torch imports here on purpose: this is pure metadata and must stay
cheap to import and trivial to test.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------- git ----------

def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def git_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def git_is_dirty() -> bool:
    """True if tracked files are modified or untracked files exist."""
    return bool(_git("status", "--porcelain"))


def git_dirty_files() -> list[str]:
    """Which files make the tree dirty — so the error message is actionable."""
    out = _git("status", "--porcelain")
    return [line[3:] for line in out.splitlines()] if out else []


# ---------- hashing ----------

def hash_file(path: str | Path) -> str:
    """SHA256 of a file's bytes. Used to pin configs and manifests."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dict(d: dict) -> str:
    """Stable hash of a config dict, independent of key ordering."""
    payload = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


# ---------- record ----------

@dataclass
class RunProvenance:
    run_id: str
    timestamp: str
    git_commit: str
    git_branch: str
    git_dirty: bool
    config_path: str
    config_hash: str
    data_config_path: str | None
    data_config_hash: str | None
    manifest_path: str | None
    manifest_hash: str | None
    seed: int
    python_version: str
    platform: str

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def build_provenance(
    config_path: str | Path,
    config: dict,
    seed: int,
    data_config_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> RunProvenance:
    """Collect everything identifying this run. Call once, at startup."""
    commit = git_commit()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    return RunProvenance(
        run_id=f"{ts}_{commit[:7]}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=commit,
        git_branch=git_branch(),
        git_dirty=git_is_dirty(),
        config_path=str(config_path),
        config_hash=hash_dict(config),
        data_config_path=str(data_config_path) if data_config_path else None,
        data_config_hash=hash_file(data_config_path) if data_config_path else None,
        manifest_path=str(manifest_path) if manifest_path else None,
        manifest_hash=hash_file(manifest_path) if manifest_path else None,
        seed=seed,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )


def require_clean_tree(allow_dirty: bool = False) -> None:
    """Refuse to start an unreproducible run unless explicitly overridden."""
    if not git_is_dirty() or allow_dirty:
        return
    files = git_dirty_files()
    preview = "\n  ".join(files[:10])
    more = f"\n  ... and {len(files) - 10} more" if len(files) > 10 else ""
    raise RuntimeError(
        f"Working tree is dirty — this run would not be reproducible.\n"
        f"  {preview}{more}\n"
        f"Commit your changes, or pass allow_dirty=True for a throwaway run."
    )