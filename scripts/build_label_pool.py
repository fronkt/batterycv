"""Sample a pool of fresh frames to hand-label for fine-tuning (assisted by label_assisted.py).

Stratified by class, EXCLUDES the 72 hand-verified eval frames (they stay a held-out test), and
CLAHE-normalizes each frame so the pool matches training + what the detector sees. Spreads the
sample across each class's capture timeline (every k-th frame) rather than clumping one run.

    python scripts/build_label_pool.py --per-class 30
    # then:  python scripts/label_assisted.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.io import build_manifest
from batterycv.preprocess import normalize_illumination, read_bgr


def main() -> None:
    paths = load_paths()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--source", default=str(paths["raw_dir"]),
                    help="frame source dir (raw BMPs or JPEG stage)")
    ap.add_argument("--out", default=str(paths["data_root"] / "label_pool"))
    args = ap.parse_args()

    out_img = Path(args.out) / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    exclude = set()
    vlist = paths["eval_dir"] / "val_list.txt"
    if vlist.exists():
        exclude = {ln.split("__", 1)[1].rsplit(".", 1)[0]
                   for ln in vlist.read_text().splitlines() if "__" in ln}
    # also skip anything already in the pool (resumable / additive)
    existing = {p.stem for p in out_img.glob("*.jpg")}

    df = build_manifest(Path(args.source))
    df = df[~df["path"].map(lambda p: Path(p).stem in exclude)]

    total = 0
    print(f"sampling {args.per_class}/class from {args.source} (excluding {len(exclude)} eval frames)")
    for label, grp in df.groupby("label"):
        grp = grp.sort_values("filename").reset_index(drop=True)
        n = min(args.per_class, len(grp))
        step = max(1, len(grp) // n)
        picks = grp.iloc[::step].head(n)
        wrote = 0
        for _, r in picks.iterrows():
            stem = f"{label}__{Path(r['path']).stem}"
            if stem in existing:
                continue
            bgr = normalize_illumination(read_bgr(r["path"]))
            cv2.imwrite(str(out_img / f"{stem}.jpg"), bgr)
            wrote += 1
        total += wrote
        print(f"  {label:<16} {wrote:>3} frames")
    print(f"\npool ready: {total} new frames -> {out_img}")
    print(f"next:  python scripts/label_assisted.py --images {out_img}")


if __name__ == "__main__":
    main()
