"""Characterize WHICH ground-truth batteries the detector misses, and why.

The recall ceiling (~0.45-0.49) is well established; what was never quantified is the *nature*
of the misses. That distinction decides where Phase-4 effort belongs:

  near-miss (best IoU 0.3-0.49)  the object IS proposed, the box is just undersized/offset ->
                                 a boundary/localization problem -> segmentation, box refinement
  total miss (best IoU < 0.1)    nothing is proposed there at all -> a *detectability* problem ->
                                 contrast/SNR, i.e. lighting or multi-frame denoising

`docs/recall_ceiling_findings.md` asserts the misses "land undersized at IoU 0.3-0.49, just under
the 0.5 bar". That claim underpins the whole "segmentation would fix it" idea, and it has never
been measured against the hand labels. This script measures it.

For every GT box it also records the photometry that would explain a total miss — object-vs-belt
contrast measured on the RAW (lossless) frame, plus size — so covered and missed populations can
be compared directly. Contrast is expressed in units of the measured per-pixel temporal noise
(sigma ~5.3/255), because "how many noise sigmas is this battery above the belt" is the physically
meaningful quantity when arguing for a lighting change.

    python scripts/analyze_misses.py                 # full 72-frame eval set
    python scripts/analyze_misses.py --limit 8
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
from batterycv.evalutil import CLASSES, class_of, iou_mat, load_gt
from batterycv.io import build_manifest
from batterycv.preprocess import read_bgr

TEMPORAL_SIGMA = 5.3   # grey levels, measured from consecutive-frame differences on static regions


def raw_lookup(raw_dir: Path) -> dict[str, str]:
    """{raw filename stem: full path} so an eval stem can be mapped back to its lossless BMP."""
    df = build_manifest(raw_dir)
    return {Path(p).stem: p for p in df["path"].tolist()}


def photometry(gray: np.ndarray, box: np.ndarray, ring_frac: float = 0.6) -> dict:
    """Object-vs-surround contrast for one box, measured on the raw grayscale frame.

    The 'surround' is a dilated-box annulus (the belt immediately around the battery), which is
    the background the detector actually has to separate the object from — a global belt mean
    would understate the difficulty where lighting is uneven.
    """
    h, w = gray.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return {}
    inner = gray[y1:y2, x1:x2]
    bw, bh = x2 - x1, y2 - y1
    ex, ey = int(bw * ring_frac), int(bh * ring_frac)
    ox1, oy1 = max(0, x1 - ex), max(0, y1 - ey)
    ox2, oy2 = min(w, x2 + ex), min(h, y2 + ey)
    outer = gray[oy1:oy2, ox1:ox2].astype(np.float32).copy()
    # blank the object itself out of the annulus
    outer[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = np.nan
    ring = outer[~np.isnan(outer)]
    if ring.size == 0:
        return {}
    inner_f = inner.astype(np.float32)
    contrast = float(inner_f.mean() - ring.mean())
    # per-pixel gradient energy inside the box: texture/edges the detector can key on
    gx = cv2.Sobel(inner_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(inner_f, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "area_px": float(bw * bh),
        "w": float(bw), "h": float(bh),
        "min_side": float(min(bw, bh)),
        "inner_mean": float(inner_f.mean()),
        "inner_std": float(inner_f.std()),
        "ring_mean": float(ring.mean()),
        "ring_std": float(ring.std()),
        "contrast": contrast,
        "abs_contrast": abs(contrast),
        "contrast_sigmas": abs(contrast) / TEMPORAL_SIGMA,
        "grad_energy": float(np.sqrt(gx ** 2 + gy ** 2).mean()),
    }


def summarize(rows: list[dict], key: str, sel) -> str:
    vals = [r[key] for r in rows if sel(r) and key in r]
    if not vals:
        return "  n/a"
    v = np.array(vals, float)
    return (f"n={len(v):<4} median={np.median(v):>8.1f}  mean={v.mean():>8.1f}  "
            f"p10={np.percentile(v,10):>7.1f}  p90={np.percentile(v,90):>7.1f}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(repo / "runs/detect/battery_ft1/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001, help="accept-all, so this is the ceiling")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(repo / "results/phase4/miss_analysis.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = load_paths()
    gt = load_gt(paths["eval_dir"] / "labels")
    imgs = sorted((paths["eval_dir"] / "images").glob("*.jpg"))
    if args.limit:
        imgs = imgs[:args.limit]
    raws = raw_lookup(Path(paths["raw_dir"]))
    model = YOLO(args.weights)

    rows: list[dict] = []
    for p in imgs:
        g = gt.get(p.stem, np.zeros((0, 4)))
        if len(g) == 0:
            continue
        raw_stem = p.stem.split("__", 1)[1]
        raw_path = raws.get(raw_stem)
        if raw_path is None:
            print(f"WARNING no raw frame for {p.stem} — skipped")
            continue
        gray = cv2.cvtColor(read_bgr(raw_path), cv2.COLOR_BGR2GRAY)

        res = model.predict(str(p), conf=args.conf, imgsz=args.imgsz,
                            device=args.device, verbose=False)[0]
        det = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        im = iou_mat(det, g) if len(det) else np.zeros((0, len(g)))
        best_iou = im.max(axis=0) if len(det) else np.zeros(len(g))

        for j in range(len(g)):
            r = {"stem": p.stem, "cls": class_of(p.stem), "gt_idx": j,
                 "best_iou": float(best_iou[j]), "covered": bool(best_iou[j] >= 0.5)}
            r.update(photometry(gray, g[j]))
            rows.append(r)

    if not rows:
        sys.exit("no GT rows analyzed")

    cov = [r for r in rows if r["covered"]]
    mis = [r for r in rows if not r["covered"]]
    near = [r for r in mis if 0.3 <= r["best_iou"] < 0.5]
    partial = [r for r in mis if 0.1 <= r["best_iou"] < 0.3]
    total = [r for r in mis if r["best_iou"] < 0.1]

    print(f"\n=== miss taxonomy ({len(rows)} GT boxes over {len(set(r['stem'] for r in rows))} frames) ===")
    print(f"  covered (IoU>=0.5)        : {len(cov):>4}  ({len(cov)/len(rows):.1%})")
    print(f"  NEAR-miss (0.3<=IoU<0.5)  : {len(near):>4}  ({len(near)/len(rows):.1%})  <- localization problem")
    print(f"  partial   (0.1<=IoU<0.3)  : {len(partial):>4}  ({len(partial)/len(rows):.1%})")
    print(f"  TOTAL miss (IoU<0.1)      : {len(total):>4}  ({len(total)/len(rows):.1%})  <- detectability problem")

    print(f"\n=== per-class miss taxonomy ===")
    print(f"{'class':<16}{'nGT':>5}{'cov':>6}{'near':>6}{'part':>6}{'total':>6}{'ceil_if_near_fixed':>20}")
    for c in CLASSES:
        cr = [r for r in rows if r["cls"] == c]
        if not cr:
            continue
        nc = sum(r["covered"] for r in cr)
        nn = sum(1 for r in cr if not r["covered"] and 0.3 <= r["best_iou"] < 0.5)
        np_ = sum(1 for r in cr if not r["covered"] and 0.1 <= r["best_iou"] < 0.3)
        nt = sum(1 for r in cr if r["best_iou"] < 0.1)
        print(f"{c:<16}{len(cr):>5}{nc:>6}{nn:>6}{np_:>6}{nt:>6}{(nc+nn)/len(cr):>20.3f}")
    nn_all = len(near)
    print(f"{'TOTAL':<16}{len(rows):>5}{len(cov):>6}{nn_all:>6}{len(partial):>6}{len(total):>6}"
          f"{(len(cov)+nn_all)/len(rows):>20.3f}")
    print("  (last column = recall if EVERY near-miss were nudged over the 0.5 bar — the")
    print("   absolute ceiling on any localization/segmentation fix)")

    print(f"\n=== photometry: covered vs missed (raw frame, grey levels) ===")
    for key in ["abs_contrast", "contrast_sigmas", "min_side", "area_px", "grad_energy", "inner_mean"]:
        print(f"  {key}")
        print(f"    covered   {summarize(rows, key, lambda r: r['covered'])}")
        print(f"    missed    {summarize(rows, key, lambda r: not r['covered'])}")
        print(f"    total-miss{summarize(rows, key, lambda r: r['best_iou'] < 0.1)}")

    print(f"\n=== per-class object-vs-belt contrast (raw) ===")
    print(f"{'class':<16}{'median|C|':>11}{'median sigmas':>15}{'median min_side':>17}")
    for c in CLASSES:
        cr = [r for r in rows if r["cls"] == c and "abs_contrast" in r]
        if not cr:
            continue
        print(f"{c:<16}{np.median([r['abs_contrast'] for r in cr]):>11.1f}"
              f"{np.median([r['contrast_sigmas'] for r in cr]):>15.1f}"
              f"{np.median([r['min_side'] for r in cr]):>17.0f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "temporal_sigma": TEMPORAL_SIGMA,
        "n_gt": len(rows),
        "taxonomy": {"covered": len(cov), "near": len(near),
                     "partial": len(partial), "total_miss": len(total)},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
