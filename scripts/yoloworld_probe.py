"""Probe an open-vocab detector (YOLO-World, prompt='battery') as a WHOLE-OBJECT pseudo-labeler.

SAM auto-mask segments bright sub-parts (labels), not whole dark batteries -> recall ceiling
~0.45 and undersized boxes. A text-prompted detector proposes whole objects. Measure its
recall@0.5 ceiling per class on the 72 hand-labeled eval frames.

    python scripts/yoloworld_probe.py --model yolov8x-worldv2.pt --conf 0.01
"""
from __future__ import annotations
import argparse, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batterycv.config import load_paths
from batterycv.preprocess import normalize_illumination, read_bgr


def load_gt(lbl_dir, w, h):
    gt = {}
    for f in lbl_dir.glob("*.txt"):
        b = []
        for ln in f.read_text().splitlines():
            if ln.strip():
                _, cx, cy, bw, bh = map(float, ln.split())
                b.append([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h])
        gt[f.stem] = np.array(b, float).reshape(-1, 4)
    return gt


def iou_mat(a, b):
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2-x1, 0, None) * np.clip(y2-y1, 0, None)
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match(boxes, gtb, iou=0.5):
    if len(gtb) == 0:
        return 0, len(boxes), 0
    if len(boxes) == 0:
        return 0, 0, len(gtb)
    im = iou_mat(boxes, gtb); matched = np.zeros(len(gtb), bool); tp = fp = 0
    order = np.argsort([-(b[2]-b[0])*(b[3]-b[1]) for b in boxes])
    for k in order:
        j = int(np.argmax(im[k]))
        if im[k, j] >= iou and not matched[j]:
            matched[j] = True; tp += 1
        else:
            fp += 1
    return tp, fp, len(gtb)


def cls(stem):
    return stem.split("__")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8x-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--prompts", nargs="+", default=["battery"])
    ap.add_argument("--clahe", action="store_true", help="apply CLAHE before inference")
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    from ultralytics import YOLOWorld
    import cv2
    paths = load_paths()
    img_dir = paths["eval_dir"] / "images"; lbl_dir = paths["eval_dir"] / "labels"
    gt = load_gt(lbl_dir, 1280, 1024)
    imgs = sorted(img_dir.glob("*.jpg"))
    model = YOLOWorld(args.model); model.set_classes(args.prompts)
    print(f"YOLO-World {args.model} prompts={args.prompts} conf={args.conf} "
          f"clahe={args.clahe} imgsz={args.imgsz}  ({len(imgs)} frames)")

    acc = defaultdict(lambda: [0, 0, 0, 0])
    for p in imgs:
        im = read_bgr(str(p))
        if args.clahe:
            im = normalize_illumination(im)
        r = model.predict(im, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        gtb = gt.get(p.stem, np.zeros((0, 4)))
        tp, fp, ng = match(boxes, gtb)
        a = acc[cls(p.stem)]; a[0] += tp; a[1] += fp; a[2] += ng; a[3] += len(boxes)

    print(f"{'class':<16}{'recall':>8}{'prec':>8}{'box/f':>8}")
    T = F = G = 0
    for c in sorted(acc):
        tp, fp, ng, nb = acc[c]; T += tp; F += fp; G += ng
        print(f"{c:<16}{tp/max(ng,1):>8.2f}{tp/max(tp+fp,1):>8.2f}{nb/12:>8.1f}")
    print(f"{'TOTAL':<16}{T/max(G,1):>8.2f}{T/max(T+F,1):>8.2f}")


if __name__ == "__main__":
    main()
