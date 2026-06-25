"""Build the frame manifest CSV (one row per frame) and segment temporal runs.

    python scripts/build_manifest.py
    python scripts/build_manifest.py --gap 2.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.io import build_manifest, segment_runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap", type=float, default=2.0, help="seconds gap that starts a new run")
    args = ap.parse_args()

    paths = load_paths()
    raw_dir = paths["raw_dir"]
    df = build_manifest(raw_dir)
    if df.empty:
        sys.exit(f"no frames found under {raw_dir}")

    n_missing_ts = int(df["timestamp"].isna().sum())
    df = segment_runs(df, gap_s=args.gap)

    out = paths["manifest"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"manifest: {len(df)} frames -> {out}")
    print(f"timestamps parsed: {len(df) - n_missing_ts}/{len(df)} (missing: {n_missing_ts})")
    print("\n--- per class (frames / runs) ---")
    g = df.groupby("label").agg(frames=("path", "size"), runs=("run_id", "nunique"))
    print(g.to_string())
    print(f"\ntotal runs: {df['run_id'].nunique()}")
    print("\n--- run size distribution ---")
    rs = df.groupby("run_id").size()
    print(rs.describe().to_string())


if __name__ == "__main__":
    main()
