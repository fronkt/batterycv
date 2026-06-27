"""Bootstrap detector training labels with SAM automatic mask generation.

Runs on the GPU box (see vast/setup.sh). For each frame: normalize illumination, generate
masks with SAM, filter to battery-like blobs (area / aspect / belt-edge rejection), convert
to YOLO single-class boxes, and assemble a YOLO dataset (images+labels under train/).
Frames listed in <eval_dir>/val_list.txt are EXCLUDED to avoid train/val leakage.

    python scripts/pseudo_label_sam.py --ckpt sam_vit_h.pth --model vit_h --limit 0
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.io import build_manifest
from batterycv.merge import merge_boxes
from batterycv.preprocess import normalize_illumination, read_bgr


def mask_to_box(seg: np.ndarray):
    ys, xs = np.where(seg)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def keep_mask(m: dict, h: int, w: int, area_min: float, area_max: float,
              aspect_min: float, aspect_max: float, stability: float) -> bool:
    a = m["area"] / (h * w)
    if not (area_min <= a <= area_max):         # too small (texture) / too big (belt)
        return False
    x, y, bw, bh = m["bbox"]
    if bw < 8 or bh < 8:
        return False
    aspect = bw / max(bh, 1)
    if not (aspect_min <= aspect <= aspect_max):
        return False
    if m.get("stability_score", 1.0) < stability:
        return False
    # reject masks hugging the whole frame border (belt segments)
    if x <= 1 and y <= 1 and (x + bw) >= w - 2 and (y + bh) >= h - 2:
        return False
    return True


def frame_to_boxes(masks, h, w, keep_kw, merge_kw, do_merge):
    """SAM masks -> list of pixel [x1,y1,x2,y2] battery boxes (filtered + de-duplicated).

    Shared by the full pipeline and the eval-frame probe so both score identical logic.
    `scores` for the merge use mask stability (prefer cleaner masks on ties); the
    containment pass is area-driven so whole-object masks win over their sub-parts.
    """
    boxes, scores = [], []
    for m in masks:
        if not keep_mask(m, h, w, **keep_kw):
            continue
        box = mask_to_box(m["segmentation"])
        if box is None:
            continue
        boxes.append(box)
        scores.append(m.get("stability_score", 1.0))
    if not boxes:
        return []
    if do_merge:
        out, _ = merge_boxes(boxes, scores, **merge_kw)
        return [tuple(map(float, b)) for b in out]
    return [tuple(map(float, b)) for b in boxes]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="SAM checkpoint path")
    ap.add_argument("--model", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of pseudo-labeled frames held out as YOLO val "
                         "(early-stopping signal; the hand-verified eval set is the real test)")
    ap.add_argument("--points-per-side", type=int, default=24,
                    help="SAM sampling grid density. 24 is a good speed/quality balance for these "
                         "frames (32 ~doubles cost and over-segments batteries into sub-parts)")
    ap.add_argument("--points-per-batch", type=int, default=256,
                    help="point prompts per forward pass — same masks, bigger = faster on a "
                         "big-VRAM GPU (SAM default 64; 256 is ~3x faster on a 32 GB card)")
    # --- keep_mask filter (tunable; small cells need a lower area floor) ---
    ap.add_argument("--area-min", type=float, default=5e-4)
    ap.add_argument("--area-max", type=float, default=0.15)
    ap.add_argument("--aspect-min", type=float, default=0.15)
    ap.add_argument("--aspect-max", type=float, default=6.0)
    ap.add_argument("--stability", type=float, default=0.85)
    # --- de-duplication / merge of fragmented masks (see batterycv.merge) ---
    ap.add_argument("--no-merge", dest="merge", action="store_false",
                    help="disable box de-duplication (debug; default merges)")
    ap.add_argument("--merge-iou", type=float, default=0.5)
    ap.add_argument("--contain-thresh", type=float, default=0.75)
    ap.add_argument("--no-agglomerate", dest="agglomerate", action="store_false")
    ap.add_argument("--agg-iou", type=float, default=0.2)
    ap.set_defaults(merge=True, agglomerate=True)
    args = ap.parse_args()

    keep_kw = dict(area_min=args.area_min, area_max=args.area_max,
                   aspect_min=args.aspect_min, aspect_max=args.aspect_max,
                   stability=args.stability)
    merge_kw = dict(iou_thresh=args.merge_iou, contain_thresh=args.contain_thresh,
                    agglomerate=args.agglomerate, agg_iou=args.agg_iou)

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    paths = load_paths()
    ds = paths["yolo_dataset"]
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / split).mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    exclude = set()
    vlist = paths["eval_dir"] / "val_list.txt"
    if vlist.exists():
        # val names look like "<label>__<stem>.jpg" -> recover stem
        exclude = {ln.split("__", 1)[1].rsplit(".", 1)[0]
                   for ln in vlist.read_text().splitlines() if "__" in ln}
    print(f"excluding {len(exclude)} held-out eval frames from training set")

    sam = sam_model_registry[args.model](checkpoint=args.ckpt).to(args.device)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=args.points_per_side,
                                    points_per_batch=args.points_per_batch,
                                    pred_iou_thresh=0.86, stability_score_thresh=0.9,
                                    min_mask_region_area=400)

    df = build_manifest(paths["raw_dir"])
    df = df[~df["path"].map(lambda p: Path(p).stem in exclude)]
    if args.limit:
        df = df.head(args.limit)

    n_boxes = 0
    n_split = {"train": 0, "val": 0}
    for i, (_, r) in enumerate(df.iterrows()):
        bgr = normalize_illumination(read_bgr(r["path"]))
        h, w = bgr.shape[:2]
        masks = gen.generate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        lines = []
        for x1, y1, x2, y2 in frame_to_boxes(masks, h, w, keep_kw, merge_kw, args.merge):
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        split = "val" if rng.random() < args.val_frac else "train"
        n_split[split] += 1
        stem = f"{r['label']}__{Path(r['path']).stem}"
        cv2.imwrite(str(ds / "images" / split / f"{stem}.jpg"), bgr)
        (ds / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_boxes += len(lines)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)} frames, {n_boxes} boxes "
                  f"(train={n_split['train']}, val={n_split['val']})")

    # YOLO val = held-out pseudo-labeled frames (early-stopping signal). The hand-verified
    # eval set stays a separate honest test, scored by eval_detection.py.
    have_val = n_split["val"] > 0
    data_yaml = ds / "data.yaml"
    data_yaml.write_text(
        f"path: {ds.as_posix()}\n"
        f"train: images/train\n"
        f"val: {'images/val' if have_val else 'images/train'}\n"
        f"names:\n  0: battery\n",
        encoding="utf-8",
    )
    print(f"\ndone: {len(df)} frames, {n_boxes} pseudo-boxes "
          f"(train={n_split['train']}, val={n_split['val']}) -> {ds}")
    print(f"data.yaml -> {data_yaml}")


if __name__ == "__main__":
    main()
