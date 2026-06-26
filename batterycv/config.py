"""Load central path config (configs/paths.yaml).

Override the config file without editing the committed Windows paths by setting the
``BATTERYCV_PATHS`` env var (the GPU box points it at a generated Linux paths file).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS_YAML = REPO_ROOT / "configs" / "paths.yaml"


def load_paths(paths_yaml: Path | str | None = None) -> dict[str, Path]:
    """Return the paths config as a dict of Path objects.

    Resolution order: explicit ``paths_yaml`` arg > ``$BATTERYCV_PATHS`` > default.
    """
    if paths_yaml is None:
        paths_yaml = os.environ.get("BATTERYCV_PATHS", PATHS_YAML)
    with open(paths_yaml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {k: Path(v) for k, v in raw.items()}
