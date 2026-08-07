"""Is the 'recall ceiling' a detection failure, or a box-convention disagreement with the GT?

`analyze_misses.py` found the miss population is 32.8% near-miss (IoU 0.3-0.5) and only 5.9%
genuinely undetected. Spot-checking those near-misses visually showed something the aggregate
numbers cannot: batteries that are plainly visible, confidently detected (conf 0.89), and scored
as MISSES because the detector's box includes a wire connector that the hand-drawn GT box
excludes — or because the GT box is drawn looser than the object.

That matters enormously for what this project tells its sponsor. The documented root cause is
"dark battery bodies blend into the dark belt, ni_mh_all is essentially invisible", and the
recommended fix is a hardware lighting change. If instead the detector is finding these objects
and merely disagreeing about box extent, the lighting ask is aimed at the wrong problem.

The 72-frame eval set was hand-labeled through a vision model, not by a metrologist, so its box
convention (does the connector count? how tight is tight?) is exactly the kind of thing that can
be systematically off without anyone noticing — every downstream number inherits it.

Three measurements, none of which require re-labeling:
  1. recall vs IoU threshold, per class. If recall at 0.3 is far above recall at 0.5, the
     objects are being FOUND and the disagreement is about extent.
  2. a convention-free localization metric: is the detection's center inside the GT box (and
     vice versa)? This asks "did it point at the right object?" without judging box extent.
  3. contact sheets of every near-miss for direct visual audit — the only way to see a
     convention problem, since by construction it is invisible to IoU.

    python scripts/probe_gt_audit.py
    python scripts/probe_gt_audit.py --limit 8 --no-sheets
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
from batterycv.evalutil import CLASSES, Accumulator, class_of, greedy_match, iou_mat, load_gt


def center_hit(det: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """For each GT box, is there a detection whose center lies inside it? (convention-free)"""
    if len(det) == 0 or len(gt) == 0:
        return np.zeros(len(gt), bool)
    cx = (det[:, 0] + det[:, 2]) / 2
    cy = (det[:, 1] + det[:, 3]) / 2
    hit = np.zeros(len(gt), bool)
    used = np.zeros(len(det), bool)
    for j in range(len(gt)):
        x1, y1, x2, y2 = gt[j]
        cand = np.where((~used) & (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2))[0]
        if len(cand):
            used[cand[0]] = True
            hit[j] = True
    return hit


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(repo / "runs/detect/battery_ft1/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="use a real operating threshold — this is about whether the object is "
                         "found in practice, not about the accept-all ceiling")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--outdir", default=str(repo / "results/phase4/gt_audit"))
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = load_paths()
    gt = load_gt(paths["eval_dir"] / "labels")
    imgs = sorted((paths["eval_dir"] / "images").glob("*.jpg"))
    if args.limit:
        imgs = imgs[:args.limit]
    model = YOLO(args.weights)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    preds: dict[str, np.ndarray] = {}
    confs: dict[str, np.ndarray] = {}
    for p in imgs:
        r = model.predict(str(p), conf=args.conf, imgsz=args.imgsz,
                          device=args.device, verbose=False)[0]
        preds[p.stem] = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        confs[p.stem] = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)

    stems = [p.stem for p in imgs]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.75]

    # ---- 1. recall vs IoU threshold -----------------------------------------------------
    print(f"\n=== recall vs IoU threshold (conf>={args.conf}) ===")
    print(f"{'class':<16}" + "".join(f"{'IoU'+str(t):>9}" for t in thresholds) + f"{'center-in':>11}")
    table: dict[str, dict] = {}
    for c in CLASSES:
        cs = [s for s in stems if class_of(s) == c]
        if not cs:
            continue
        row = []
        for t in thresholds:
            acc = Accumulator()
            for s in cs:
                acc.add(s, preds[s], gt.get(s, np.zeros((0, 4))), confs[s], iou=t)
            row.append(acc.recall())
        ch_hit = ch_tot = 0
        for s in cs:
            h = center_hit(preds[s], gt.get(s, np.zeros((0, 4))))
            ch_hit += int(h.sum()); ch_tot += len(h)
        ch = ch_hit / max(ch_tot, 1)
        table[c] = {"recall_by_iou": dict(zip(map(str, thresholds), row)), "center_in": ch}
        print(f"{c:<16}" + "".join(f"{v:>9.3f}" for v in row) + f"{ch:>11.3f}")
    row = []
    for t in thresholds:
        acc = Accumulator()
        for s in stems:
            acc.add(s, preds[s], gt.get(s, np.zeros((0, 4))), confs[s], iou=t)
        row.append(acc.recall())
    ch_hit = ch_tot = 0
    for s in stems:
        h = center_hit(preds[s], gt.get(s, np.zeros((0, 4))))
        ch_hit += int(h.sum()); ch_tot += len(h)
    print(f"{'TOTAL':<16}" + "".join(f"{v:>9.3f}" for v in row) + f"{ch_hit/max(ch_tot,1):>11.3f}")
    print("\n  'center-in' = a detection's center falls inside the GT box: did the detector POINT")
    print("  at the object, ignoring box extent entirely. A large gap between center-in and")
    print("  IoU-0.5 recall means the objects ARE found and the boxes merely disagree.")

    # ---- 2. contact sheets of the near-misses -------------------------------------------
    if not args.no_sheets:
        cells: list[np.ndarray] = []
        meta: list[dict] = []
        for s in stems:
            g = gt.get(s, np.zeros((0, 4)))
            b = preds[s]
            if len(g) == 0:
                continue
            im = cv2.imread(str((paths["eval_dir"] / "images") / f"{s}.jpg"))
            M = iou_mat(b, g) if len(b) else np.zeros((0, len(g)))
            for j in range(len(g)):
                best = M[:, j].max() if len(b) else 0.0
                if not (0.1 <= best < 0.5):
                    continue
                i = int(np.argmax(M[:, j]))
                x1 = int(min(g[j, 0], b[i, 0])); y1 = int(min(g[j, 1], b[i, 1]))
                x2 = int(max(g[j, 2], b[i, 2])); y2 = int(max(g[j, 3], b[i, 3]))
                pad = 40
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(im.shape[1], x2 + pad), min(im.shape[0], y2 + pad)
                crop = im[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue
                cv2.rectangle(crop, (int(g[j, 0]) - x1, int(g[j, 1]) - y1),
                              (int(g[j, 2]) - x1, int(g[j, 3]) - y1), (0, 255, 0), 3)
                cv2.rectangle(crop, (int(b[i, 0]) - x1, int(b[i, 1]) - y1),
                              (int(b[i, 2]) - x1, int(b[i, 3]) - y1), (0, 0, 255), 2)
                h_, w_ = crop.shape[:2]
                sc = 320 / max(h_, w_, 1)
                crop = cv2.resize(crop, (max(1, int(w_ * sc)), max(1, int(h_ * sc))))
                canvas = np.zeros((360, 340, 3), np.uint8)
                canvas[:crop.shape[0], :crop.shape[1]] = crop[:360, :340]
                cv2.putText(canvas, f"{class_of(s)[:14]} IoU={best:.2f}", (4, 352),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cells.append(canvas)
                meta.append({"stem": s, "cls": class_of(s), "iou": float(best)})
        print(f"\n=== near-miss contact sheets: {len(cells)} cases (green=GT, red=detection) ===")
        per = 12
        for k in range(0, len(cells), per):
            chunk = cells[k:k + per]
            rows = []
            for r0 in range(0, len(chunk), 4):
                r = chunk[r0:r0 + 4]
                while len(r) < 4:
                    r.append(np.zeros_like(chunk[0]))
                rows.append(np.hstack(r))
            sheet = np.vstack(rows)
            fp = outdir / f"nearmiss_sheet_{k//per:02d}.png"
            cv2.imwrite(str(fp), sheet)
            print(f"  wrote {fp.name}  ({len(chunk)} cases)")
        (outdir / "nearmiss_index.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    (outdir / "iou_sweep.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
