"""Exploratory data analysis: brightness stats + inter-frame gap distribution.

Reads a sample of frames per class (BMP I/O is heavy at 10 GB) to characterize darkness,
and analyzes filename timestamps to recommend a run-segmentation gap threshold.

    python scripts/eda.py --sample 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.io import build_manifest, segment_runs
from batterycv.preprocess import brightness_stats, read_bgr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=30, help="frames per class for brightness")
    args = ap.parse_args()

    paths = load_paths()
    out_dir = paths["data_root"] / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_manifest(paths["raw_dir"])
    print(f"frames: {len(df)} | classes: {df['label'].nunique()}")

    # ---- brightness on a per-class sample ----
    rows = []
    for label, g in df.groupby("label"):
        for p in g["path"].sample(min(args.sample, len(g)), random_state=0):
            s = brightness_stats(read_bgr(p))
            s["label"] = label
            rows.append(s)
    bdf = pd.DataFrame(rows)
    print("\n--- grayscale brightness (0-255), by class ---")
    print(bdf.groupby("label")[["mean", "p5", "p95"]].mean().round(1).to_string())
    print(f"\noverall mean brightness: {bdf['mean'].mean():.1f} "
          f"(dark; CLAHE/normalization recommended before detect/OCR)")

    plt.figure(figsize=(7, 4))
    for label, g in bdf.groupby("label"):
        plt.hist(g["mean"], bins=15, alpha=0.5, label=label)
    plt.xlabel("mean grayscale intensity"); plt.ylabel("frames"); plt.legend(fontsize=7)
    plt.title("Per-class brightness (sampled)"); plt.tight_layout()
    plt.savefig(out_dir / "brightness_hist.png", dpi=120)

    # ---- inter-frame gaps -> recommend run gap threshold ----
    dts = (
        df.dropna(subset=["timestamp"])
        .sort_values(["label", "timestamp"])
        .groupby("label")["timestamp"]
        .diff().dt.total_seconds().dropna()
    )
    print("\n--- inter-frame gap seconds (within class) ---")
    print(dts.describe(percentiles=[.5, .9, .95, .99]).to_string())
    rec = max(1.0, round(float(dts.quantile(0.95)) * 3, 1))
    print(f"\nrecommended run gap threshold ~ {rec}s (3x p95)")

    plt.figure(figsize=(7, 4))
    plt.hist(dts.clip(upper=dts.quantile(0.99)), bins=60)
    plt.xlabel("seconds between consecutive frames"); plt.ylabel("count")
    plt.title("Inter-frame gaps (clipped p99)"); plt.tight_layout()
    plt.savefig(out_dir / "interframe_gaps.png", dpi=120)

    seg = segment_runs(df, gap_s=rec)
    print(f"\nat {rec}s gap -> {seg['run_id'].nunique()} runs; "
          f"median run length {seg.groupby('run_id').size().median():.0f} frames")
    print(f"\nplots -> {out_dir}")


if __name__ == "__main__":
    main()
