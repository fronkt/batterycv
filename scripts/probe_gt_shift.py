"""Is the eval ground truth misregistered by a fixed translation?

The blind adjudication (adjudicate_boxes.py) left 12 boxes judged genuine detection misses. Crops
of all 12 show the battery sitting OUTSIDE the label box, up and to the right, on otherwise bare
belt -- and the displacement from each GT box to the nearest detector box is downward in 12 of 12
(median dy -104 px, sign test p = 2e-4). The same bias runs through the 84 near misses at half the
magnitude (median dy -34 px). That is the signature of a global offset between the labels and the
images they are paired with, not of a detector that cannot see batteries.

This tests it the only way a one-parameter fix may be tested: fit the shift on one half of the
frames, report it on the other half it never saw. A registration error transfers across the split
because it is a property of the label set. Overfitting to box noise does not -- that is exactly
how probe_boxscale.py's +0.006 in-sample gain turned into -0.011 held out.

    python scripts/probe_gt_shift.py
    python scripts/probe_gt_shift.py --step 2 --span 200      # finer/wider search
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.evalutil import CLASSES, class_of, iou_mat, load_gt


def recall_at(gt: dict, preds: dict, stems, dx: float, dy: float, iou: float = 0.5) -> tuple:
    """Greedy-matched recall over `stems` with every GT box translated by (dx, dy)."""
    hit = tot = 0
    for s in stems:
        g, p = gt[s].copy(), preds[s]
        if len(g) == 0:
            continue
        g[:, [0, 2]] += dx
        g[:, [1, 3]] += dy
        tot += len(g)
        if len(p) == 0:
            continue
        m = iou_mat(p, g)
        taken = np.zeros(len(g), bool)
        for i in range(len(p)):
            cand = np.where(~taken & (m[i] >= iou))[0]
            if len(cand):
                taken[cand[np.argmax(m[i, cand])]] = True
        hit += int(taken.sum())
    return hit, tot


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    paths = load_paths()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(paths["eval_dir"] / "labels"))
    ap.add_argument("--weights", default=str(repo / "runs/detect/battery_ft1/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--span", type=int, default=160, help="search +/- this many px in x and y")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--out", default=str(repo / "results/phase4/gt_shift.json"))
    args = ap.parse_args()

    gt = load_gt(args.labels)
    img_dir = paths["eval_dir"] / "images"

    from ultralytics import YOLO
    model = YOLO(args.weights)
    preds = {}
    for s in sorted(gt):
        r = model.predict(str(img_dir / f"{s}.jpg"), conf=args.conf, imgsz=args.imgsz,
                          verbose=False)[0]
        preds[s] = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))

    # Split within each class, alternating, so both halves carry every class. A random split
    # could put all of a 12-frame class on one side and make the held-out number meaningless.
    fit, rep = [], []
    for c in CLASSES:
        ss = [s for s in sorted(gt) if class_of(s) == c]
        fit += ss[0::2]
        rep += ss[1::2]
    print(f"fit on {len(fit)} frames, report on {len(rep)} frames it never saw")

    h, t = recall_at(gt, preds, rep, 0, 0)
    base_rep = h / t
    h0, t0 = recall_at(gt, preds, sorted(gt), 0, 0)
    print(f"baseline (no shift): all {h0}/{t0} = {h0/t0:.3f}   held-out {h}/{t} = {base_rep:.3f}")

    best = (-1.0, 0, 0)
    grid = range(-args.span, args.span + 1, args.step)
    for dx in grid:
        for dy in grid:
            h, t = recall_at(gt, preds, fit, dx, dy)
            r = h / t if t else 0.0
            if r > best[0]:
                best = (r, dx, dy)
    fit_r, dx, dy = best
    print(f"\nbest shift on the FIT half : dx {dx:+d}  dy {dy:+d}   recall {fit_r:.3f}")

    h, t = recall_at(gt, preds, rep, dx, dy)
    rep_r = h / t
    print(f"same shift on the HELD-OUT half        : recall {rep_r:.3f}  "
          f"({rep_r - base_rep:+.3f} vs its own baseline {base_rep:.3f})")

    print(f"\n{'class':<16}{'baseline':>10}{'shifted':>10}{'delta':>9}   (held-out frames only)")
    for c in CLASSES:
        ss = [s for s in rep if class_of(s) == c]
        h0c, t0c = recall_at(gt, preds, ss, 0, 0)
        hc, tc = recall_at(gt, preds, ss, dx, dy)
        if t0c:
            print(f"{c:<16}{h0c/t0c:>10.3f}{hc/tc:>10.3f}{hc/tc - h0c/t0c:>+9.3f}")

    verdict = ("The labels are misregistered. A single translation, fitted without ever seeing "
               "these frames, recovers most of the gap -- no detector change involved."
               if rep_r - base_rep > 0.15 else
               "A single translation does NOT transfer. The offset is not a global "
               "registration error; treat the per-box displacement as label noise instead.")
    print(f"\n{verdict}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "fit_frames": len(fit), "report_frames": len(rep), "dx": dx, "dy": dy,
        "recall_fit_at_best": fit_r, "recall_heldout_baseline": base_rep,
        "recall_heldout_shifted": rep_r, "recall_all_baseline": h0 / t0,
        "span": args.span, "step": args.step, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
