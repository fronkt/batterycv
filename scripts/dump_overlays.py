"""Dump GT-vs-SAM-box overlays for named eval frames, to eyeball WHY small classes miss.
GT = green, raw kept SAM boxes = red. Run on the box; pull the PNGs and inspect.

    python scripts/dump_overlays.py --ckpt .../sam_vit_h.pth --pps 32 --prefix ni_mh_all li_ion_mobile ni_cd_small
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batterycv.config import load_paths
from batterycv.preprocess import normalize_illumination, read_bgr
from pseudo_label_sam import frame_to_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="vit_h")
    ap.add_argument("--pps", type=int, default=32)
    ap.add_argument("--prefix", nargs="+", default=["ni_mh_all", "li_ion_mobile", "ni_cd_small"])
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--out", default="/workspace/batterycv/probe_overlays")
    args = ap.parse_args()

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    paths = load_paths()
    img_dir = paths["eval_dir"] / "images"
    lbl_dir = paths["eval_dir"] / "labels"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    sam = sam_model_registry[args.model](checkpoint=args.ckpt).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=args.pps, points_per_batch=256,
                                    pred_iou_thresh=0.86, stability_score_thresh=0.9,
                                    min_mask_region_area=400)
    keep_kw = dict(area_min=5e-4, area_max=0.15, aspect_min=0.15, aspect_max=6.0, stability=0.85)

    chosen = []
    for pre in args.prefix:
        fs = sorted(p for p in img_dir.glob("*.jpg") if p.stem.startswith(pre))[:args.per_class]
        chosen += fs
    for p in chosen:
        bgr = normalize_illumination(read_bgr(str(p)))
        h, w = bgr.shape[:2]
        masks = gen.generate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        boxes = frame_to_boxes(masks, h, w, keep_kw, {}, do_merge=False)
        vis = bgr.copy()
        # GT green
        lf = lbl_dir / (p.stem + ".txt")
        ng = 0
        if lf.exists():
            for ln in lf.read_text().splitlines():
                if ln.strip():
                    _, cx, cy, bw, bh = map(float, ln.split())
                    x1, y1 = int((cx-bw/2)*w), int((cy-bh/2)*h)
                    x2, y2 = int((cx+bw/2)*w), int((cy+bh/2)*h)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3); ng += 1
        # SAM raw red
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        cv2.putText(vis, f"GT(green)={ng}  SAM(red)={len(boxes)}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(out / (p.stem[:40] + ".jpg")), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {len(chosen)} overlays -> {out}")


if __name__ == "__main__":
    main()
