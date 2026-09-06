"""
Repo-root resolution and Ultralytics path setup.

Ultralytics resolves a dataset config's relative `path` against its global
`datasets_dir` setting (stored in ~/.config/Ultralytics/settings.json), NOT
against the config file or the working directory. That global is shared
across every project on the machine, so we override it per-process here
rather than relying on whatever it happens to be set to.

Import and call configure_ultralytics_paths() before any Ultralytics
dataset loading.
"""
from pathlib import Path

# paths.py lives at <repo>/src/train/paths.py -> parents[1] is <repo>
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"


def configure_ultralytics_paths() -> Path:
    """Point Ultralytics' dataset resolution at this repo. Returns repo root."""
    from ultralytics.utils import SETTINGS

    SETTINGS["datasets_dir"] = str(REPO_ROOT)
    return REPO_ROOT