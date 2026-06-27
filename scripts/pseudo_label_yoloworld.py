"""Bootstrap detector labels with an open-vocab detector (YOLO-World, prompt='battery').

Replaces SAM auto-mask pseudo-labeling. SAM segments bright sub-parts (labels) of dark
batteries -> undersized boxes, recall ceiling ~0.45 and precision ~0.07 (see
docs/recall_ceiling_findings.md). YOLO-World proposes WHOLE objects at ~3x the precision and
~50 ms/frame (vs SAM's 3.5 s), so this run is minutes, not hours. Recall is still imagery-capped
~0.45; this is the best zero-shot baseline.

Same I/O contract as pseudo_label_sam.py: CLAHE-normalized images + YOLO single-class labels
under <yolo_dataset>/{images,labels}/{train,val}, eval frames excluded, data.yaml written.

    python scripts/pseudo_label_yoloworld.py --model yolov8x-worldv2.pt --conf 0.05
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.io import build_manifest
from batterycv.merge import merge_boxes
from batterycv.preprocess import normalize_illumination, read_bgr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolov8x-worldv2.pt")
    ap.add_argument("--prompts", nargs="+", default=["battery"])
    ap.add_argument("--conf", type=float, default=0.05,
                    help="label confidence; 0.05 balances precision (~.33) vs recall (~.36) on "
                         "the hand eval — lower poisons training with belt false-positives")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    ap.add_argument("--val-frac", type=float, default=0.1)
    # conservative dedup: YOLO-World already NMSes; only drop near-exact nests, never fuse cells
    ap.add_argument("--no-merge", dest="merge", action="store_false")
    ap.add_argument("--merge-iou", type=float, default=0.6)
    ap.add_argument("--contain-thresh", type=float, default=0.9)
    ap.set_defaults(merge=True)
    args = ap.parse_args()

    from ultralytics import YOLOWorld

    paths = load_paths()
    ds = paths["yolo_dataset"]
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / split).mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    exclude = set()
    vlist = paths["eval_dir"] / "val_list.txt"
    if vlist.exists():
        exclude = {ln.split("__", 1)[1].rsplit(".", 1)[0]
                   for ln in vlist.read_text().splitlines() if "__" in ln}
    print(f"excluding {len(exclude)} held-out eval frames from training set")

    model = YOLOWorld(args.model)
    model.set_classes(args.prompts)

    df = build_manifest(paths["raw_dir"])
    df = df[~df["path"].map(lambda p: Path(p).stem in exclude)]
    if args.limit:
        df = df.head(args.limit)

    import cv2
    n_boxes = 0
    n_split = {"train": 0, "val": 0}
    for i, (_, r) in enumerate(df.iterrows()):
        bgr = normalize_illumination(read_bgr(r["path"]))
        h, w = bgr.shape[:2]
        res = model.predict(bgr, conf=args.conf, imgsz=args.imgsz, device=args.device,
                            verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
        else:
            xyxy = np.zeros((0, 4)); scores = np.zeros(0)
        if args.merge and len(xyxy):
            xyxy, scores = merge_boxes(xyxy, scores, iou_thresh=args.merge_iou,
                                       contain_thresh=args.contain_thresh, agglomerate=False)
        lines = []
        for x1, y1, x2, y2 in xyxy:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        split = "val" if rng.random() < args.val_frac else "train"
        n_split[split] += 1
        stem = f"{r['label']}__{Path(r['path']).stem}"
        cv2.imwrite(str(ds / "images" / split / f"{stem}.jpg"), bgr)
        (ds / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_boxes += len(lines)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(df)} frames, {n_boxes} boxes "
                  f"(train={n_split['train']}, val={n_split['val']})")

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
