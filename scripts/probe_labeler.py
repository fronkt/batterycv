"""Probe the SAM pseudo-labeler on the 72 hand-labeled eval frames BEFORE the full re-label.

Answers the decisive question cheaply: with merge + retuned keep_mask (and a given
points-per-side), can the labeler produce >=0.5-IoU whole-object boxes for the small classes
(mobile / ni_cd_small / ni_mh) that currently cap recall at ~0.47?

Generates SAM masks ONCE per frame (the slow part) then sweeps several keep/merge configs
in-memory, scoring each per class against the hand labels. Run on the GPU box:

    python scripts/probe_labeler.py --ckpt /workspace/checkpoints/sam_vit_h_4b8939.pth --pps 24
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pseudo_label_sam import

from batterycv.config import load_paths
from batterycv.preprocess import normalize_illumination, read_bgr
from pseudo_label_sam import frame_to_boxes  # noqa: E402  (same dir on box)


def load_gt(labels_dir: Path, w: int, h: int):
    gt = {}
    for f in labels_dir.glob("*.txt"):
        b = []
        for ln in f.read_text().splitlines():
            if ln.strip():
                _, cx, cy, bw, bh = map(float, ln.split())
                b.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
        gt[f.stem] = np.array(b, float).reshape(-1, 4)
    return gt


def iou_mat(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match(boxes, gtb, iou=0.5):
    """greedy: returns (tp, fp, n_gt)."""
    if len(gtb) == 0:
        return 0, len(boxes), 0
    if len(boxes) == 0:
        return 0, 0, len(gtb)
    im = iou_mat(boxes, gtb)
    matched = np.zeros(len(gtb), bool)
    tp = fp = 0
    for k in range(len(boxes)):
        j = int(np.argmax(im[k]))
        if im[k, j] >= iou and not matched[j]:
            matched[j] = True; tp += 1
        else:
            fp += 1
    return tp, fp, len(gtb)


# configs: (name, keep_kw, merge_kw or None for no-merge)
def configs():
    base = dict(area_min=5e-4, area_max=0.15, aspect_min=0.15, aspect_max=6.0, stability=0.85)
    lowfloor = dict(base, area_min=2e-4)
    return [
        ("raw (no merge)",        base,     None),
        ("contain.75",            base,     dict(iou_thresh=0.5, contain_thresh=0.75, agglomerate=False)),
        ("contain.75+agg.2",      base,     dict(iou_thresh=0.5, contain_thresh=0.75, agglomerate=True, agg_iou=0.2)),
        ("contain.6+agg.1",       base,     dict(iou_thresh=0.5, contain_thresh=0.6, agglomerate=True, agg_iou=0.1)),
        ("lowfloor+cont.75+agg.2", lowfloor, dict(iou_thresh=0.5, contain_thresh=0.75, agglomerate=True, agg_iou=0.2)),
        ("lowfloor+cont.6+agg.15", lowfloor, dict(iou_thresh=0.5, contain_thresh=0.6, agglomerate=True, agg_iou=0.15)),
    ]


def cls(stem):
    return stem.split("__")[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="vit_h")
    ap.add_argument("--pps", type=int, default=24)
    ap.add_argument("--points-per-batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    paths = load_paths()
    img_dir = paths["eval_dir"] / "images"
    lbl_dir = paths["eval_dir"] / "labels"
    imgs = sorted(img_dir.glob("*.jpg"))
    if args.limit:
        imgs = imgs[:args.limit]
    gt = load_gt(lbl_dir, 1280, 1024)
    print(f"probe: {len(imgs)} eval frames, pps={args.pps}, model={args.model}")

    sam = sam_model_registry[args.model](checkpoint=args.ckpt).to(args.device)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=args.pps,
                                    points_per_batch=args.points_per_batch,
                                    pred_iou_thresh=0.86, stability_score_thresh=0.9,
                                    min_mask_region_area=400)

    cfgs = configs()
    # acc[cfg_name][class] = [tp, fp, ngt, nbox]
    acc = {c[0]: defaultdict(lambda: [0, 0, 0, 0]) for c in cfgs}

    for n, p in enumerate(imgs):
        bgr = normalize_illumination(read_bgr(str(p)))
        h, w = bgr.shape[:2]
        masks = gen.generate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        stem = p.stem
        gtb = gt.get(stem, np.zeros((0, 4)))
        c = cls(stem)
        for name, keep_kw, merge_kw in cfgs:
            boxes = frame_to_boxes(masks, h, w, keep_kw, merge_kw or {}, do_merge=merge_kw is not None)
            tp, fp, ng = match(boxes, gtb)
            a = acc[name][c]
            a[0] += tp; a[1] += fp; a[2] += ng; a[3] += len(boxes)
        if (n + 1) % 12 == 0:
            print(f"  {n+1}/{len(imgs)} frames")

    classes = sorted({cls(s) for s in gt})
    for name, _, _ in cfgs:
        print(f"\n=== {name} ===")
        print(f"{'class':<16}{'recall':>8}{'prec':>8}{'box/f':>8}")
        T = F = G = 0
        for c in classes:
            tp, fp, ng, nb = acc[name][c]
            r = tp / max(ng, 1); pr = tp / max(tp + fp, 1)
            T += tp; F += fp; G += ng
            print(f"{c:<16}{r:>8.2f}{pr:>8.2f}{nb/12:>8.1f}")
        print(f"{'TOTAL':<16}{T/max(G,1):>8.2f}{T/max(T+F,1):>8.2f}")


if __name__ == "__main__":
    main()
