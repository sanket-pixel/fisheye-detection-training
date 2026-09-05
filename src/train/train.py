# src/train/train.py
"""
Training entrypoint.

Enforces the reproducibility contract before any GPU time is spent:
every run is pinned to an exact git commit and logs its config to W&B.
A run whose code state cannot be reconstructed is not a run, it's scratch
output — so the script refuses to start on a dirty working tree.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from src.train.paths import REPO_ROOT, configure_ultralytics_paths


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def git_is_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    ).strip()
    return bool(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/yolo26n_baseline.yaml")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Run despite uncommitted changes. Produces an unreproducible run.",
    )
    args = parser.parse_args()

    if git_is_dirty() and not args.allow_dirty:
        sys.exit(
            "Working tree is dirty — this run would not be reproducible.\n"
            "Commit your changes, or pass --allow-dirty for a throwaway run."
        )

    configure_ultralytics_paths()

    cfg_path = REPO_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    commit = git_commit()
    print(f"commit:  {commit}")
    print(f"config:  {cfg_path.relative_to(REPO_ROOT)}")
    print(f"dirty:   {git_is_dirty()}")

    from ultralytics import YOLO

    model = YOLO(cfg.pop("model"))

    # Ultralytics reads W&B settings from its own integration; tag the run
    # with provenance so the registry entry can be reconstructed later.
    results = model.train(
        **cfg,
        # provenance, surfaced in the run metadata
        **{"name": f"{cfg.get('name', 'run')}_{commit[:7]}"},
    )
    print(results)


if __name__ == "__main__":
    main()