"""End-to-end demo: frames -> detect+track -> classify type -> metadata -> annotated video.

The one-command version of the whole project. For a capture run it (1) tracks batteries with
the fine-tuned detector + ByteTrack (same recipe as track.py), (2) classifies each tracked
battery's TYPE from its best crop with the Phase-3 classifier, (3) joins the trusted OCR
metadata (brand / part # / marks — the fields the Phase-2b audit validated; chemistry/V/cap
are excluded as prior-filled) from results/phase2_ocr by (run_id, track_id), and (4) renders
the sorting-line view: boxes colored by predicted chemistry, with a running bin count.

Outputs under <work_dir>/pipeline/<label>_run<rid>/:
  pipeline.mp4    annotated video — box + "#id TYPE p" colored by chemistry, header bin counts
  batteries.csv   one row per battery: predicted type/chemistry, prob, brand/part#/marks, frames
  crops/*.jpg     best crop per battery (same naming as track.py)

Classification happens after tracking, so every frame shows each battery's FINAL predicted
type (two-pass render). OCR is joined from the archived Phase-2b results, not re-run — live
VLM OCR is ~80 s/crop on CPU; for new footage run ocr_crops.py separately and pass --ocr-json.

    python scripts/run_pipeline.py                        # longest li_ion_laptop run
    python scripts/run_pipeline.py --run-id 60            # a LiSO2 run
    python scripts/run_pipeline.py --label ni_cd_bulk
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.io import build_manifest, segment_runs
from batterycv.preprocess import normalize_illumination, read_bgr

CHEM_GROUP = {
    "li_ion_mobile": "li_ion", "li_ion_laptop": "li_ion",
    "liso2": "liso2",
    "ni_cd_bulk": "ni_cd", "ni_cd_small": "ni_cd",
    "ni_mh_all": "ni_mh",
}
CHEM_COLOR = {  # BGR — the "sorting bin" color of each chemistry
    "li_ion": (0, 165, 255),   # orange
    "liso2": (0, 0, 230),      # red (primary lithium = the hazardous bin)
    "ni_cd": (255, 140, 0),    # blue
    "ni_mh": (0, 200, 0),      # green
    "?": (160, 160, 160),
}


def pick_run(runs, label, run_id):
    if run_id is not None:
        g = runs[runs["run_id"] == run_id]
        if g.empty:
            sys.exit(f"run-id {run_id} not found (use track.py --list-runs)")
        return run_id, g
    sub = runs[runs["label"] == label] if label else runs
    if sub.empty:
        sys.exit(f"no runs for label {label!r}")
    rid = int(sub.groupby("run_id").size().idxmax())
    return rid, sub[sub["run_id"] == rid]


def main() -> None:
    paths = load_paths()
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(repo / "runs/detect/battery_ft1/weights/best.pt"))
    ap.add_argument("--cls-weights",
                    default=str(repo / "runs/classify/type_v1/weights/best.pt"))
    ap.add_argument("--ocr-json", default=str(repo / "results/phase2_ocr/ocr.json"),
                    help="archived OCR records to join metadata from ('' to skip)")
    ap.add_argument("--label", default="li_ion_laptop")
    ap.add_argument("--run-id", type=int, default=None)
    ap.add_argument("--source", default=str(paths["raw_dir"]))
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--cls-imgsz", type=int, default=224)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    for w, what in ((args.weights, "detector"), (args.cls_weights, "classifier")):
        if not Path(w).exists():
            sys.exit(f"{what} weights not found: {w}")

    df = build_manifest(Path(args.source))
    if df.empty:
        sys.exit(f"no frames under {args.source}")
    rid, grp = pick_run(segment_runs(df, gap_s=args.gap), args.label, args.run_id)
    grp = grp.sort_values("timestamp").reset_index(drop=True)
    if args.max_frames:
        grp = grp.head(args.max_frames)
    label = grp["label"].iloc[0]
    out = Path(args.out) if args.out else paths["work_dir"] / "pipeline" / f"{label}_run{rid}"
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    print(f"pipeline: run {rid} ({label}), {len(grp)} frames -> {out}")

    from ultralytics import YOLO
    det = YOLO(args.weights)

    # ---- pass 1: track, keeping every frame's boxes + the best crop per battery
    frames_boxes: list[list[tuple[int, int, int, int, int, float]]] = []  # per frame: (tid,x1,y1,x2,y2,cf)
    best: dict[int, tuple[float, np.ndarray]] = {}
    for _, r in grp.iterrows():
        bgr = normalize_illumination(read_bgr(r["path"]))
        res = det.track(bgr, persist=True, tracker=args.tracker, conf=args.conf,
                        imgsz=args.imgsz, device=args.device, verbose=False)[0]
        boxes = []
        if res.boxes is not None and res.boxes.id is not None:
            for (x1, y1, x2, y2), tid, cf in zip(res.boxes.xyxy.cpu().numpy(),
                                                 res.boxes.id.cpu().numpy().astype(int),
                                                 res.boxes.conf.cpu().numpy()):
                tid, cf = int(tid), float(cf)
                x1, y1, x2, y2 = (int(v) for v in (x1, y1, x2, y2))
                boxes.append((tid, x1, y1, x2, y2, cf))
                if cf > best.get(tid, (0.0, None))[0]:
                    crop = bgr[max(0, y1):y2, max(0, x1):x2].copy()
                    if crop.size:
                        best[tid] = (cf, crop)
        frames_boxes.append(boxes)
    print(f"  tracked {len(best)} batteries")

    # ---- classify each battery's best crop
    cls = YOLO(args.cls_weights)
    pred: dict[int, tuple[str, float]] = {}
    for tid, (cf, crop) in sorted(best.items()):
        p = cls.predict(crop, imgsz=args.cls_imgsz, device=args.device, verbose=False)[0].probs
        pred[tid] = (cls.names[int(p.top1)], float(p.top1conf))
        cv2.imwrite(str(crops_dir / f"{label}__run{rid}__id{tid:03d}_c{cf:.2f}.jpg"), crop)

    # ---- join trusted OCR metadata by (run_id, track_id)
    meta: dict[int, dict] = {}
    if args.ocr_json and Path(args.ocr_json).exists():
        for rec in json.loads(Path(args.ocr_json).read_text(encoding="utf-8")):
            if rec.get("run_id") == rid and rec.get("track_id") is not None:
                meta[int(rec["track_id"])] = rec

    # ---- pass 2: render the sorting-line view
    h, w = read_bgr(grp["path"].iloc[0]).shape[:2]
    vw = cv2.VideoWriter(str(out / "pipeline.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    bins = Counter(CHEM_GROUP.get(pred[t][0], "?") for t in pred)
    header2 = "bins: " + "  ".join(f"{c}={n}" for c, n in sorted(bins.items()))
    for fi, (_, r) in enumerate(grp.iterrows()):
        disp = normalize_illumination(read_bgr(r["path"]))
        for tid, x1, y1, x2, y2, cf in frames_boxes[fi]:
            typ, p = pred.get(tid, ("?", 0.0))
            chem = CHEM_GROUP.get(typ, "?")
            col = CHEM_COLOR[chem]
            cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2)
            tag = f"#{tid} {typ} {p:.2f}" if typ != "?" else f"#{tid} ?"
            cv2.putText(disp, tag, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
            brand = str(meta.get(tid, {}).get("manufacturer") or "").strip()
            part = str(meta.get(tid, {}).get("model") or "").strip()
            if brand or part:
                cv2.putText(disp, " ".join(x for x in (brand, part) if x),
                            (x1, min(h - 4, y2 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp, f"{label}  run {rid}  frame {fi+1}/{len(grp)}  batteries: {len(best)}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, header2, (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        vw.write(disp)
    vw.release()

    # ---- per-battery manifest + summary
    import csv
    with (out / "batteries.csv").open("w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["track_id", "pred_type", "pred_chem", "prob", "det_conf",
                     "brand", "part_no", "marks", "session_label"])
        for tid in sorted(pred):
            typ, p = pred[tid]
            m = meta.get(tid, {})
            cw.writerow([tid, typ, CHEM_GROUP.get(typ, "?"), f"{p:.3f}",
                         f"{best[tid][0]:.3f}", m.get("manufacturer") or "",
                         m.get("model") or "", " ".join(m.get("marks") or []), label])

    print(f"\n--- pipeline summary (run {rid}, session label = {label}) ---")
    for tid in sorted(pred):
        typ, p = pred[tid]
        m = meta.get(tid, {})
        extra = " ".join(x for x in (str(m.get('manufacturer') or ''),
                                     str(m.get('model') or '')) if x.strip())
        print(f"  #{tid:<3} {typ:<15} p={p:.2f}  chem={CHEM_GROUP.get(typ, '?'):<7} {extra}")
    agree = sum(1 for t in pred if pred[t][0] == label)
    print(f"  type agreement with session label: {agree}/{len(pred)}")
    print(f"  video -> {out / 'pipeline.mp4'}\n  manifest -> {out / 'batteries.csv'}")


if __name__ == "__main__":
    main()
