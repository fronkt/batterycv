"""Are the near-misses a systematic box-GEOMETRY bias that a global correction can undo?

`analyze_misses.py` showed the miss population is dominated by near-misses, not by undetected
objects: of 186 GT boxes, 48.9% are covered, 32.8% sit at IoU 0.3-0.5, and only 5.9% have no
overlapping proposal at all. Recall would be 0.817 if every near-miss were nudged over the bar.

`docs/recall_ceiling_findings.md` diagnosed the mechanism: zero-shot labelers boxed the *bright
printed label* rather than the whole dark cell, and the detector inherited that habit, so its
boxes land undersized. If that bias is SYSTEMATIC — consistent scale factor, consistent center —
then simply expanding every predicted box by a fitted factor recovers recall for free, with no
retraining. If instead the near-misses are randomly offset, no global correction helps and the
problem is genuinely per-object.

This script measures the bias and then tests the correction honestly:
  1. geometry of near-miss pairs: width/height ratios, center offset (in GT-size units)
  2. a global scale sweep (isotropic, and separate x/y), scored with the shared matcher
  3. the SAME sweep fitted on one half of the frames and scored on the other half, because
     fitting a scale on the same 72 frames you report on is circular

Caveat that must accompany any positive result: expanding boxes trivially inflates IoU against
GT drawn generously. The honest checks are the held-out split and the precision column.

    python scripts/probe_boxscale.py
    python scripts/probe_boxscale.py --limit 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.evalutil import CLASSES, Accumulator, class_of, iou_mat, load_gt


def scale_boxes(boxes: np.ndarray, sx: float, sy: float,
                w: int = 1280, h: int = 1024) -> np.ndarray:
    """Expand each box about its own center by (sx, sy), clipped to the image."""
    if len(boxes) == 0:
        return boxes
    b = np.asarray(boxes, float).copy()
    cx = (b[:, 0] + b[:, 2]) / 2
    cy = (b[:, 1] + b[:, 3]) / 2
    bw = (b[:, 2] - b[:, 0]) * sx
    bh = (b[:, 3] - b[:, 1]) * sy
    out = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    out[:, 0] = np.clip(out[:, 0], 0, w)
    out[:, 2] = np.clip(out[:, 2], 0, w)
    out[:, 1] = np.clip(out[:, 1], 0, h)
    out[:, 3] = np.clip(out[:, 3], 0, h)
    return out


def score(preds: dict, gt: dict, stems: list[str], sx: float, sy: float,
          conf_thr: float = 0.0) -> tuple[float, float]:
    """(recall, precision) over `stems` after scaling every box by (sx, sy)."""
    acc = Accumulator()
    for s in stems:
        b, c = preds[s]
        keep = c >= conf_thr
        acc.add(s, scale_boxes(b[keep], sx, sy), gt.get(s, np.zeros((0, 4))), c[keep])
    return acc.recall(), acc.precision()


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(repo / "runs/detect/battery_ft1/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(repo / "results/phase4/boxscale.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = load_paths()
    gt = load_gt(paths["eval_dir"] / "labels")
    imgs = sorted((paths["eval_dir"] / "images").glob("*.jpg"))
    if args.limit:
        imgs = imgs[:args.limit]
    model = YOLO(args.weights)

    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for p in imgs:
        r = model.predict(str(p), conf=args.conf, imgsz=args.imgsz,
                          device=args.device, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        c = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        preds[p.stem] = (b, c)
    stems = [p.stem for p in imgs]

    # ---- 1. geometry of the near-miss pairs -------------------------------------------------
    ratios_w, ratios_h, off_x, off_y, ious = [], [], [], [], []
    per_cls: dict[str, list[float]] = {c: [] for c in CLASSES}
    for s in stems:
        g = gt.get(s, np.zeros((0, 4)))
        b, _ = preds[s]
        if len(g) == 0 or len(b) == 0:
            continue
        im = iou_mat(b, g)
        for j in range(len(g)):
            i = int(np.argmax(im[:, j]))
            v = im[i, j]
            if not (0.1 <= v < 0.5):
                continue
            gw, gh = g[j, 2] - g[j, 0], g[j, 3] - g[j, 1]
            dw, dh = b[i, 2] - b[i, 0], b[i, 3] - b[i, 1]
            ratios_w.append(dw / max(gw, 1)); ratios_h.append(dh / max(gh, 1))
            off_x.append(((b[i, 0] + b[i, 2]) / 2 - (g[j, 0] + g[j, 2]) / 2) / max(gw, 1))
            off_y.append(((b[i, 1] + b[i, 3]) / 2 - (g[j, 1] + g[j, 3]) / 2) / max(gh, 1))
            ious.append(v)
            per_cls[class_of(s)].append(dw / max(gw, 1))

    def st(v):
        v = np.array(v, float)
        return (f"n={len(v):<4} median={np.median(v):>6.2f} mean={v.mean():>6.2f} "
                f"p10={np.percentile(v,10):>6.2f} p90={np.percentile(v,90):>6.2f}")

    print(f"\n=== geometry of near/partial-miss pairs (0.1 <= IoU < 0.5) ===")
    print(f"  width  ratio det/GT : {st(ratios_w)}")
    print(f"  height ratio det/GT : {st(ratios_h)}")
    print(f"  center dx / GT width : {st(off_x)}")
    print(f"  center dy / GT height: {st(off_y)}")
    print("  (ratio < 1 = detector box UNDERSIZED; center offsets near 0 = concentric,")
    print("   which is the case a pure scale correction can fix)")
    print(f"\n  per-class median width ratio:")
    for c in CLASSES:
        if per_cls[c]:
            print(f"    {c:<16}{np.median(per_cls[c]):>6.2f}  (n={len(per_cls[c])})")

    # ---- 2. global scale sweep --------------------------------------------------------------
    grid = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]
    print(f"\n=== isotropic scale sweep (accept-all; recall is the priority metric) ===")
    print(f"{'scale':>7}{'recall':>9}{'prec':>8}{'recall@.25':>12}{'prec@.25':>10}")
    best = (0.0, 1.0)
    for s in grid:
        r, pr = score(preds, gt, stems, s, s, 0.0)
        r25, p25 = score(preds, gt, stems, s, s, 0.25)
        if r > best[0]:
            best = (r, s)
        print(f"{s:>7.2f}{r:>9.3f}{pr:>8.3f}{r25:>12.3f}{p25:>10.3f}")
    print(f"  best isotropic scale = {best[1]:.2f} -> accept-all recall {best[0]:.3f} "
          f"(baseline s=1.0)")

    print(f"\n=== anisotropic sweep (accept-all recall) ===")
    xs = [1.0, 1.2, 1.4, 1.6, 1.8]
    print("      sy:" + "".join(f"{v:>8.2f}" for v in xs))
    best_aniso = (0.0, 1.0, 1.0)
    for sx in xs:
        row = []
        for sy in xs:
            r, _ = score(preds, gt, stems, sx, sy, 0.0)
            row.append(r)
            if r > best_aniso[0]:
                best_aniso = (r, sx, sy)
        print(f"sx={sx:>4.2f}  " + "".join(f"{v:>8.3f}" for v in row))
    print(f"  best (sx,sy) = ({best_aniso[1]:.2f},{best_aniso[2]:.2f}) -> recall {best_aniso[0]:.3f}")

    # ---- 3. held-out check: fit the scale on half the frames, score on the other half --------
    even = [s for i, s in enumerate(sorted(stems)) if i % 2 == 0]
    odd = [s for i, s in enumerate(sorted(stems)) if i % 2 == 1]
    fold_out = {}
    print(f"\n=== held-out check (fit scale on one half, report on the other) ===")
    for name, fit, rep in (("fit=even report=odd", even, odd), ("fit=odd report=even", odd, even)):
        bs, br = 1.0, -1.0
        for s in grid:
            r, _ = score(preds, gt, fit, s, s, 0.0)
            if r > br:
                br, bs = r, s
        r_rep, p_rep = score(preds, gt, rep, bs, bs, 0.0)
        r_base, p_base = score(preds, gt, rep, 1.0, 1.0, 0.0)
        fold_out[name] = {"fitted_scale": bs, "recall": r_rep, "baseline_recall": r_base}
        print(f"  {name}: fitted scale {bs:.2f} -> held-out recall {r_rep:.3f} "
              f"(baseline {r_base:.3f}, delta {r_rep-r_base:+.3f}), prec {p_rep:.3f} vs {p_base:.3f}")

    # ---- 4. per-class effect of the best global scale ---------------------------------------
    print(f"\n=== per-class recall at the best global scale (accept-all) ===")
    a0, a1 = Accumulator(), Accumulator()
    for s in stems:
        b, c = preds[s]
        g = gt.get(s, np.zeros((0, 4)))
        a0.add(s, b, g, c)
        a1.add(s, scale_boxes(b, best[1], best[1]), g, c)
    print(f"{'class':<16}{'baseline':>10}{'scaled':>9}{'delta':>8}")
    for c in CLASSES:
        if c not in a0.stats:
            continue
        r0, r1 = a0.recall(c), a1.recall(c)
        print(f"{c:<16}{r0:>10.3f}{r1:>9.3f}{r1-r0:>+8.3f}")
    print(f"{'TOTAL':<16}{a0.recall():>10.3f}{a1.recall():>9.3f}{a1.recall()-a0.recall():>+8.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_frames": len(stems),
        "near_miss_geometry": {
            "n": len(ratios_w),
            "median_width_ratio": float(np.median(ratios_w)) if ratios_w else None,
            "median_height_ratio": float(np.median(ratios_h)) if ratios_h else None,
            "median_center_dx_frac": float(np.median(off_x)) if off_x else None,
            "median_center_dy_frac": float(np.median(off_y)) if off_y else None,
        },
        "best_isotropic": {"scale": best[1], "recall": best[0]},
        "best_anisotropic": {"sx": best_aniso[1], "sy": best_aniso[2], "recall": best_aniso[0]},
        "heldout": fold_out,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
