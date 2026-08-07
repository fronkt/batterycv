"""Sanity-gate `batterycv.evalutil` against the published ft1 numbers before trusting it.

Phase-4 probes compare new methods to the numbers in `docs/recall_ceiling_findings.md`
(ft1: P 0.466 / R 0.446 / mAP50 0.252, from Ultralytics `model.val`). Those probes use this
repo's own matcher instead of Ultralytics, so the matcher has to be shown to land in the same
place first — otherwise a "gain" could just be a friendlier scorer.

Prints this harness's P/R/AP50 for ft1 at the same conf/imgsz Ultralytics used, plus the
recall ceiling (accept-all-boxes), which is the metric the 3-way labeler table reports.

    python scripts/validate_harness.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.evalutil import Accumulator, load_gt


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(repo / "runs/detect/battery_ft1/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001, help="low, so AP sees the full curve")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = load_paths()
    img_dir = paths["eval_dir"] / "images"
    gt = load_gt(paths["eval_dir"] / "labels")
    imgs = sorted(img_dir.glob("*.jpg"))
    print(f"harness validation: {len(imgs)} eval frames, {sum(len(v) for v in gt.values())} GT boxes")
    print(f"weights={Path(args.weights).parent.parent.name} imgsz={args.imgsz} conf={args.conf}\n")

    model = YOLO(args.weights)
    # accept-all (ceiling) and a deployment-like conf=0.25 slice, from ONE inference pass
    acc_all = Accumulator()
    acc_25 = Accumulator()
    for p in imgs:
        res = model.predict(str(p), conf=args.conf, imgsz=args.imgsz,
                            device=args.device, verbose=False)[0]
        b = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        s = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros(0)
        g = gt.get(p.stem, np.zeros((0, 4)))
        acc_all.add(p.stem, b, g, s)
        keep = s >= 0.25
        acc_25.add(p.stem, b[keep], g, s[keep])

    print(acc_all.table(f"accept-all (conf>={args.conf}) — recall ceiling + AP"))
    print(acc_25.table("conf>=0.25 — deployment slice"))
    print("\npublished ft1 (Ultralytics model.val): P 0.466  R 0.446  mAP50 0.252")
    print("This harness should land near those on the conf>=0.25 slice for P/R, and AP@0.5")
    print("within ~0.03 of mAP50. Larger gaps mean the matcher differs — investigate before")
    print("trusting any Phase-4 comparison.")


if __name__ == "__main__":
    main()
