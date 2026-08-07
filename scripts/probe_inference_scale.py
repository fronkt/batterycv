"""Does inference resolution / tiling convert 'found the object' into 'box good enough'?

Two facts from the Phase-4 diagnosis motivate this:
  - Localization, not detection, is the binding constraint: at conf 0.25 the detector puts a box
    centre inside 83.9% of GT boxes but clears IoU 0.5 on only 45.2%.
  - Size predicts failure: covered objects have median min-side 184 px, totally-missed ones 128 px.

And the frames are 1280x1024 while the detector runs at imgsz 1024 — every inference DOWNSCALES
by 0.8x, shrinking a 128 px cell to ~102 px before the network ever sees it. The Phase-1
resolution sweep varied the TRAINING resolution (1024 vs 1280) and concluded resolution was not
the lever; it never varied INFERENCE resolution on a fixed model, and never tried tiling. Tiled
inference (slice the frame, detect per tile at full network resolution, map back, merge) is the
standard small-object remedy and costs no training.

Sweeps, all on the frozen ft1 checkpoint:
  - inference imgsz in {1024 (baseline), 1280 (native), 1536, 1920}
  - 2x2 and 3x3 overlapping tiling at imgsz 1024

Reports recall at IoU 0.5 AND at IoU 0.3, plus centre-in-box, so it is visible whether a change
finds NEW objects or merely tightens boxes on objects already found — those are different wins and
the project has conflated them before.

    python scripts/probe_inference_scale.py
    python scripts/probe_inference_scale.py --limit 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.evalutil import CLASSES, Accumulator, class_of, load_gt
from batterycv.merge import merge_boxes


def tile_predict(model, img_path: str, grid: int, imgsz: int, conf: float,
                 device: str, overlap: float = 0.25):
    """Detect on an overlapping grid of tiles, map boxes back to full-frame coords, merge."""
    import cv2
    im = cv2.imread(img_path)
    H, W = im.shape[:2]
    th, tw = H / grid, W / grid
    oh, ow = th * overlap, tw * overlap
    boxes, scores = [], []
    for r in range(grid):
        for c in range(grid):
            y1 = max(0, int(r * th - oh)); y2 = min(H, int((r + 1) * th + oh))
            x1 = max(0, int(c * tw - ow)); x2 = min(W, int((c + 1) * tw + ow))
            tile = im[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            res = model.predict(tile, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            b = res.boxes.xyxy.cpu().numpy().copy()
            b[:, [0, 2]] += x1
            b[:, [1, 3]] += y1
            boxes.append(b)
            scores.append(res.boxes.conf.cpu().numpy())
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0)
    b = np.vstack(boxes); s = np.concatenate(scores)
    # merge duplicates from the overlap seams; score-ordered NMS only (no agglomerate, which
    # would fuse genuinely adjacent cells — the dense classes here sit shoulder to shoulder)
    mb, ms = merge_boxes(b, s, iou_thresh=0.55, contain_thresh=0.85, agglomerate=False)
    return mb, ms


def evaluate(preds: dict, gt: dict, stems: list[str], conf_thr: float) -> dict:
    out = {}
    for iou in (0.3, 0.5):
        acc = Accumulator()
        for s in stems:
            b, c = preds[s]
            keep = c >= conf_thr
            acc.add(s, b[keep], gt.get(s, np.zeros((0, 4))), c[keep], iou=iou)
        out[f"recall@{iou}"] = acc.recall()
        out[f"prec@{iou}"] = acc.precision()
        if iou == 0.5:
            out["per_class"] = {c: acc.recall(c) for c in CLASSES if c in acc.stats}
    # centre-in-box
    hit = tot = 0
    for s in stems:
        b, c = preds[s]
        b = b[c >= conf_thr]
        g = gt.get(s, np.zeros((0, 4)))
        if len(g) == 0:
            continue
        if len(b) == 0:
            tot += len(g); continue
        cx = (b[:, 0] + b[:, 2]) / 2; cy = (b[:, 1] + b[:, 3]) / 2
        used = np.zeros(len(b), bool)
        for j in range(len(g)):
            x1, y1, x2, y2 = g[j]
            cand = np.where((~used) & (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2))[0]
            if len(cand):
                used[cand[0]] = True; hit += 1
            tot += 1
    out["center_in"] = hit / max(tot, 1)
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(repo / "runs/detect/battery_ft1/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--eval-conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(repo / "results/phase4/inference_scale.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = load_paths()
    gt = load_gt(paths["eval_dir"] / "labels")
    imgs = sorted((paths["eval_dir"] / "images").glob("*.jpg"))
    if args.limit:
        imgs = imgs[:args.limit]
    stems = [p.stem for p in imgs]
    model = YOLO(args.weights)

    configs: list[tuple[str, dict]] = [
        ("imgsz1024 (baseline)", {"kind": "plain", "imgsz": 1024}),
        ("imgsz1280 (native)", {"kind": "plain", "imgsz": 1280}),
        ("imgsz1536", {"kind": "plain", "imgsz": 1536}),
        ("imgsz1920", {"kind": "plain", "imgsz": 1920}),
        ("tile2x2 @1024", {"kind": "tile", "grid": 2, "imgsz": 1024}),
        ("tile3x3 @1024", {"kind": "tile", "grid": 3, "imgsz": 1024}),
    ]

    results = {}
    for name, cfg in configs:
        preds = {}
        for p in imgs:
            if cfg["kind"] == "plain":
                r = model.predict(str(p), conf=args.conf, imgsz=cfg["imgsz"],
                                  device=args.device, verbose=False)[0]
                b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                c = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
            else:
                b, c = tile_predict(model, str(p), cfg["grid"], cfg["imgsz"],
                                    args.conf, args.device)
            preds[p.stem] = (b, c)
        results[name] = evaluate(preds, gt, stems, args.eval_conf)
        r = results[name]
        print(f"{name:<22} R@.5 {r['recall@0.5']:.3f}  P@.5 {r['prec@0.5']:.3f}  "
              f"R@.3 {r['recall@0.3']:.3f}  centre-in {r['center_in']:.3f}")

    print(f"\n=== per-class recall @ IoU 0.5 (conf>={args.eval_conf}) ===")
    print(f"{'config':<22}" + "".join(f"{c[:11]:>13}" for c in CLASSES))
    for name, _ in configs:
        pc = results[name]["per_class"]
        print(f"{name:<22}" + "".join(f"{pc.get(c, float('nan')):>13.3f}" for c in CLASSES))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
