"""Frame discovery, filename timestamp parsing, manifest build, run segmentation.

Filename format (Basler Pylon): ``<model><serial>_<HH-MM-SS-mmm>_<MM-DD-YY>.bmp``
e.g. ``acA1300-200uc23075908_11-59-24-575_03-05-25.bmp``
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

# folder name -> (clean label, chemistry, form factor). Folder == weak session-level label.
CLASSES: dict[str, dict[str, str]] = {
    "Li_ion_mobile_battery_03_03_25": {"label": "li_ion_mobile", "chemistry": "Li-ion", "form": "mobile"},
    "Li_ion_laptop_battery_03_03_25": {"label": "li_ion_laptop", "chemistry": "Li-ion", "form": "laptop"},
    "Ni_Cd_bulk_battery_03_03_25":    {"label": "ni_cd_bulk",    "chemistry": "Ni-Cd",  "form": "bulk"},
    "LiSO2_battery_03_03_25":         {"label": "liso2",         "chemistry": "LiSO2",  "form": "cell"},
    "Ni_Cd_small_battery_03_03_25":   {"label": "ni_cd_small",   "chemistry": "Ni-Cd",  "form": "small"},
    "Ni_MH_all_battery_03_03_25":     {"label": "ni_mh_all",     "chemistry": "Ni-MH",  "form": "mixed"},
}

# capture trailing _HH-MM-SS-mmm_MM-DD-YY before the extension
_TS_RE = re.compile(r"_(\d{2})-(\d{2})-(\d{2})-(\d{3})_(\d{2})-(\d{2})-(\d{2})$")


def parse_timestamp(stem: str) -> datetime | None:
    """Parse the capture datetime from a filename stem; None if it doesn't match."""
    m = _TS_RE.search(stem)
    if not m:
        return None
    hh, mm, ss, ms, mo, dd, yy = (int(x) for x in m.groups())
    try:
        return datetime(2000 + yy, mo, dd, hh, mm, ss, ms * 1000)
    except ValueError:
        return None


def build_manifest(raw_dir: Path) -> pd.DataFrame:
    """Walk <raw_dir>/<class>/*.bmp -> tidy DataFrame (one row per frame)."""
    rows = []
    for folder, meta in CLASSES.items():
        cls_dir = raw_dir / folder
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.glob("*.bmp")):  # non-recursive: tree is clean post-quarantine
            ts = parse_timestamp(p.stem)
            rows.append(
                {
                    "path": str(p),
                    "filename": p.name,
                    "folder": folder,
                    "label": meta["label"],
                    "chemistry": meta["chemistry"],
                    "form": meta["form"],
                    "timestamp": ts,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["label", "timestamp"]).reset_index(drop=True)
    return df


def segment_runs(df: pd.DataFrame, gap_s: float = 2.0) -> pd.DataFrame:
    """Add a global ``run_id`` per class: a new run starts after a time gap > ``gap_s``.

    A "run" is a contiguous capture burst (belt moving) where tracking is meaningful.
    Returns a copy with integer ``run_id`` and ``dt_prev`` (seconds since previous frame).
    """
    out = []
    next_run = 0
    for _, g in df.sort_values(["label", "timestamp"]).groupby("label", sort=False):
        g = g.copy()
        dt = g["timestamp"].diff().dt.total_seconds()
        new_run = (dt.isna()) | (dt > gap_s)
        local = new_run.cumsum() - 1
        g["dt_prev"] = dt
        g["run_id"] = local + next_run
        next_run = int(g["run_id"].max()) + 1
        out.append(g)
    return pd.concat(out).sort_values(["run_id", "timestamp"]).reset_index(drop=True)
