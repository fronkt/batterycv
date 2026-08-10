"""How much of the '0.45 recall ceiling' was the ruler? Compare eval labels v1 vs v2.

Run this after re-labeling any number of frames to `labels_v2/` (see docs/labeling_convention.md
— a 15-frame pilot is enough to see the direction). It answers three questions on exactly the
frames that exist in BOTH label sets, so a partial relabel is fine:

  1. GEOMETRY — how do the two label sets differ? Systematic size ratio and centre offset,
     matched box to box. The Phase-4 audit predicts v1 boxes are larger and offset; this is the
     first direct measurement of that (the audit only established which box a human prefers).
  2. RECALL — the frozen detector re-scored against each label set. The difference is the part
     of the ceiling that was measurement, not detection. Nothing about the detector changes
     between the two columns; only the ruler does.
  3. RESIDUAL — the objects still missed under v2 at IoU<0.1. Those are the genuine detection
     failures, the only population that could justify a hardware/lighting change, and this
     script writes crops of them for direct inspection.

    python scripts/compare_labelsets.py
    python scripts/compare_labelsets.py --v2 /path/to/labels_v2 --no-crops
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.evalutil import CLASSES, Accumulator, class_of, iou_mat, load_gt


def geometry_delta(v1: dict, v2: dict, stems: list[str]) -> dict:
    """Match v1 boxes to v2 boxes per frame (best IoU) and summarize the systematic difference."""
    wr, hr, dx, dy, ious = [], [], [], [], []
    unmatched_v1 = unmatched_v2 = 0
    for s in stems:
        a, b = v1.get(s, np.zeros((0, 4))), v2.get(s, np.zeros((0, 4)))
        if len(a) == 0 or len(b) == 0:
            unmatched_v1 += len(a); unmatched_v2 += len(b)
            continue
        m = iou_mat(a, b)
        used = np.zeros(len(b), bool)
        for i in range(len(a)):
            j = int(np.argmax(m[i]))
            if m[i, j] < 0.1 or used[j]:
                unmatched_v1 += 1
                continue
            used[j] = True
            ious.append(float(m[i, j]))
            aw, ah = a[i, 2] - a[i, 0], a[i, 3] - a[i, 1]
            bw, bh = b[j, 2] - b[j, 0], b[j, 3] - b[j, 1]
            wr.append(aw / max(bw, 1)); hr.append(ah / max(bh, 1))
            dx.append(((a[i, 0] + a[i, 2]) / 2 - (b[j, 0] + b[j, 2]) / 2) / max(bw, 1))
            dy.append(((a[i, 1] + a[i, 3]) / 2 - (b[j, 1] + b[j, 3]) / 2) / max(bh, 1))
        unmatched_v2 += int((~used).sum())
    med = lambda v: float(np.median(v)) if v else float("nan")  # noqa: E731
    return {"n_matched": len(ious), "median_iou_v1_v2": med(ious),
            "median_width_ratio_v1_over_v2": med(wr),
            "median_height_ratio_v1_over_v2": med(hr),
            "median_center_dx_frac": med(dx), "median_center_dy_frac": med(dy),
            "unmatched_v1": unmatched_v1, "unmatched_v2": unmatched_v2}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    paths = load_paths()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", default=str(paths["eval_dir"] / "labels"))
    ap.add_argument("--v2", default=str(paths["eval_dir"] / "labels_v2"))
    ap.add_argument("--weights", default=str(repo / "runs/detect/battery_ft1/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-crops", action="store_true")
    ap.add_argument("--outdir", default=str(repo / "results/phase4/labelset_compare"))
    args = ap.parse_args()

    v2dir = Path(args.v2)
    if not v2dir.exists() or not any(v2dir.glob("*.txt")):
        sys.exit(f"no v2 labels in {v2dir}\nRe-label first — see docs/labeling_convention.md")

    v1 = load_gt(args.v1)
    v2 = load_gt(v2dir)
    stems = sorted(set(v1) & set(v2))
    if not stems:
        sys.exit("v1 and v2 share no frame stems")
    print(f"comparing {len(stems)} frames labeled in BOTH sets "
          f"({sum(len(v1[s]) for s in stems)} v1 boxes vs {sum(len(v2[s]) for s in stems)} v2 boxes)")
    if len(stems) < len(v1):
        print(f"  (partial relabel: {len(stems)}/{len(v1)} frames done — numbers below cover only those)")

    # ---- 1. geometry -----------------------------------------------------------------------
    g = geometry_delta(v1, v2, stems)
    print(f"\n=== geometry: how v1 differs from v2 ===")
    print(f"  matched boxes            : {g['n_matched']}   (median IoU between them "
          f"{g['median_iou_v1_v2']:.3f})")
    print(f"  v1/v2 width ratio        : {g['median_width_ratio_v1_over_v2']:.3f}   "
          f"(>1 = old boxes were LARGER)")
    print(f"  v1/v2 height ratio       : {g['median_height_ratio_v1_over_v2']:.3f}")
    print(f"  centre offset dx (of box): {g['median_center_dx_frac']:+.3f}")
    print(f"  centre offset dy (of box): {g['median_center_dy_frac']:+.3f}")
    print(f"  boxes only in v1 / only in v2: {g['unmatched_v1']} / {g['unmatched_v2']}")

    # ---- 2. recall against each ruler ------------------------------------------------------
    from ultralytics import YOLO
    model = YOLO(args.weights)
    img_dir = paths["eval_dir"] / "images"
    preds = {}
    for s in stems:
        r = model.predict(str(img_dir / f"{s}.jpg"), conf=args.conf, imgsz=args.imgsz,
                          device=args.device, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        c = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        preds[s] = (b, c)

    print(f"\n=== the SAME detector, scored against each ruler (conf>={args.conf}) ===")
    print(f"{'class':<16}{'v1 recall':>11}{'v2 recall':>11}{'delta':>9}")
    accs = {}
    for tag, gtset in (("v1", v1), ("v2", v2)):
        acc = Accumulator()
        for s in stems:
            b, c = preds[s]
            acc.add(s, b, gtset.get(s, np.zeros((0, 4))), c)
        accs[tag] = acc
    for c in CLASSES:
        if c not in accs["v1"].stats and c not in accs["v2"].stats:
            continue
        r1, r2 = accs["v1"].recall(c), accs["v2"].recall(c)
        print(f"{c:<16}{r1:>11.3f}{r2:>11.3f}{r2-r1:>+9.3f}")
    r1, r2 = accs["v1"].recall(), accs["v2"].recall()
    p1, p2 = accs["v1"].precision(), accs["v2"].precision()
    print(f"{'TOTAL':<16}{r1:>11.3f}{r2:>11.3f}{r2-r1:>+9.3f}")
    print(f"{'precision':<16}{p1:>11.3f}{p2:>11.3f}{p2-p1:>+9.3f}")
    print(f"\n  The detector is byte-identical between these two columns. Any difference is the")
    print(f"  measurement, not the model — that is the number Phase 4 was arguing about.")

    # ---- 3. the residual genuine misses under v2 -------------------------------------------
    resid = []
    for s in stems:
        b, _ = preds[s]
        gtb = v2.get(s, np.zeros((0, 4)))
        if len(gtb) == 0:
            continue
        m = iou_mat(b, gtb) if len(b) else np.zeros((0, len(gtb)))
        best = m.max(axis=0) if len(b) else np.zeros(len(gtb))
        for j in range(len(gtb)):
            if best[j] < 0.1:
                resid.append({"stem": s, "cls": class_of(s), "box": gtb[j].tolist(),
                              "best_iou": float(best[j])})
    print(f"\n=== residual TOTAL misses under v2 (IoU<0.1): {len(resid)} of "
          f"{sum(len(v2[s]) for s in stems)} boxes ===")
    by_cls: dict[str, int] = {}
    for r in resid:
        by_cls[r["cls"]] = by_cls.get(r["cls"], 0) + 1
    for c, n in sorted(by_cls.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<16}{n:>4}")
    print("  These are the genuine detection failures — the ONLY population that could justify")
    print("  a lighting/hardware change. Look at every one before making that argument.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.no_crops and resid:
        for k, r in enumerate(resid):
            im = cv2.imread(str(img_dir / f"{r['stem']}.jpg"))
            x1, y1, x2, y2 = [int(v) for v in r["box"]]
            pad = 60
            crop = im[max(0, y1 - pad):min(im.shape[0], y2 + pad),
                      max(0, x1 - pad):min(im.shape[1], x2 + pad)].copy()
            if crop.size:
                cv2.rectangle(crop, (min(pad, x1), min(pad, y1)),
                              (min(pad, x1) + (x2 - x1), min(pad, y1) + (y2 - y1)), (0, 255, 0), 2)
                cv2.imwrite(str(outdir / f"residual_{k:02d}_{r['cls']}.jpg"), crop)
        print(f"  wrote {len(resid)} residual crops -> {outdir}")

    (outdir / "compare.json").write_text(json.dumps({
        "n_frames": len(stems), "geometry": g,
        "recall_v1": r1, "recall_v2": r2, "precision_v1": p1, "precision_v2": p2,
        "per_class": {c: {"v1": accs["v1"].recall(c), "v2": accs["v2"].recall(c)}
                      for c in CLASSES if c in accs["v2"].stats},
        "residual_total_misses": resid,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir / 'compare.json'}")


if __name__ == "__main__":
    main()
