"""Load central path config (configs/paths.yaml)."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS_YAML = REPO_ROOT / "configs" / "paths.yaml"


def load_paths(paths_yaml: Path | str = PATHS_YAML) -> dict[str, Path]:
    """Return the paths config as a dict of Path objects."""
    with open(paths_yaml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {k: Path(v) for k, v in raw.items()}
