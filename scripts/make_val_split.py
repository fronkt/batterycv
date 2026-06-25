"""Sample a stratified set of frames for the hand-verified detection eval set.

Writes normalized JPGs (so faint batteries are visible while labeling) and a frame list.
You then draw battery boxes in a labeler (Label Studio / labelImg) and export YOLO-format
txt files into <eval_dir>/labels/. This set is GROUND TRUTH and is excluded from training.

    python scripts/make_val_split.py --per-class 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.io import build_manifest, segment_runs
from batterycv.preprocess import normalize_illumination, read_bgr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=12, help="frames sampled per class")
    ap.add_argument("--gap", type=float, default=2.0)
    args = ap.parse_args()

    paths = load_paths()
    eval_dir = paths["eval_dir"]
    img_dir = eval_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "labels").mkdir(parents=True, exist_ok=True)

    df = segment_runs(build_manifest(paths["raw_dir"]), gap_s=args.gap)

    picks = []
    for label, g in df.groupby("label"):
        g = g.sample(frac=1, random_state=0)               # shuffle within class
        sample = g.groupby("run_id").head(1).head(args.per_class)  # spread across runs first
        if len(sample) < args.per_class:                   # top up from remaining frames
            rest = g.drop(sample.index)
            sample = pd.concat([sample, rest.head(args.per_class - len(sample))])
        picks.append(sample)

    sel = pd.concat(picks).reset_index(drop=True)

    listing = []
    for _, r in sel.iterrows():
        norm = normalize_illumination(read_bgr(r["path"]))
        name = f"{r['label']}__{Path(r['path']).stem}.jpg"
        cv2.imwrite(str(img_dir / name), norm)
        listing.append(name)
    (eval_dir / "val_list.txt").write_text("\n".join(sorted(listing)), encoding="utf-8")

    print(f"wrote {len(listing)} eval frames -> {img_dir}")
    print(f"frame list -> {eval_dir / 'val_list.txt'}")
    print("\nNEXT: draw battery boxes on these images and export YOLO txt to "
          f"{eval_dir / 'labels'}")
    print("Recommended tool: Label Studio (pip install label-studio) or labelImg. "
          "Single class id 0 = 'battery'.")


if __name__ == "__main__":
    main()
