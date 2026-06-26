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
from batterycv.preprocess import normalize_illumination, read_bgr


def mask_to_box(seg: np.ndarray):
    ys, xs = np.where(seg)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def keep_mask(m: dict, h: int, w: int) -> bool:
    a = m["area"] / (h * w)
    if not (5e-4 <= a <= 0.15):                 # too small (texture) / too big (belt)
        return False
    x, y, bw, bh = m["bbox"]
    if bw < 8 or bh < 8:
        return False
    aspect = bw / max(bh, 1)
    if not (0.15 <= aspect <= 6.0):
        return False
    if m.get("stability_score", 1.0) < 0.85:
        return False
    # reject masks hugging the whole frame border (belt segments)
    if x <= 1 and y <= 1 and (x + bw) >= w - 2 and (y + bh) >= h - 2:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="SAM checkpoint path")
    ap.add_argument("--model", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of pseudo-labeled frames held out as YOLO val "
                         "(early-stopping signal; the hand-verified eval set is the real test)")
    args = ap.parse_args()

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
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.86,
                                    stability_score_thresh=0.9, min_mask_region_area=400)

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
        for m in masks:
            if not keep_mask(m, h, w):
                continue
            box = mask_to_box(m["segmentation"])
            if box is None:
                continue
            x1, y1, x2, y2 = box
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
