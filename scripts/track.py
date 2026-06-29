"""Track batteries across a capture run with ByteTrack — persistent per-battery IDs.

Phase-1 step 2. Runs the fine-tuned detector frame-by-frame over one contiguous capture *run*
(a belt burst, segmented by timestamp gap in io.segment_runs) and links detections into tracks
with ByteTrack, so each physical battery gets one stable ID as it moves down the belt. Outputs:

  - <out>/track.mp4         annotated video (box + battery #ID), for the demo
  - <out>/crops/*.jpg       best (highest-conf) crop per track ID — the input to the OCR phase
  - <out>/tracks.csv        one row per (frame, track ID): box + conf
  - prints a summary: # unique batteries counted over the run

Frames are CLAHE-normalized first (the detector was trained that way, and CLAHE also makes the
printed text legible for the later OCR step, so crops are saved normalized). Single-class detector,
so every track is "battery"; the run's folder label is the weak TYPE tag carried onto each crop.

    python scripts/track.py                              # longest li_ion_laptop run, ft1 detector
    python scripts/track.py --label ni_cd_bulk
    python scripts/track.py --run-id 12 --fps 8
    python scripts/track.py --list-runs                  # show runs (id, label, #frames) and exit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.io import build_manifest, segment_runs
from batterycv.preprocess import normalize_illumination, read_bgr


def _color(i: int) -> tuple[int, int, int]:
    """Stable, well-separated BGR color per track ID."""
    rng = np.random.default_rng(int(i) * 9973 + 1)
    return tuple(int(c) for c in rng.integers(60, 256, size=3))


def pick_run(runs, label, run_id):
    """Return (run_id, frame-rows-DataFrame) for the chosen run."""
    if run_id is not None:
        g = runs[runs["run_id"] == run_id]
        if g.empty:
            sys.exit(f"run-id {run_id} not found (use --list-runs)")
        return run_id, g
    sub = runs[runs["label"] == label] if label else runs
    if sub.empty:
        sys.exit(f"no runs for label {label!r} (use --list-runs)")
    # longest run of the chosen label = the best tracking demo
    rid = int(sub.groupby("run_id").size().idxmax())
    return rid, sub[sub["run_id"] == rid]


def main() -> None:
    paths = load_paths()
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights",
                    default=str(repo / "runs/detect/battery_ft1/weights/best.pt"))
    ap.add_argument("--label", default="li_ion_laptop",
                    help="class whose longest run to track (ignored if --run-id given)")
    ap.add_argument("--run-id", type=int, default=None, help="track this exact global run_id")
    ap.add_argument("--source", default=str(paths["raw_dir"]),
                    help="frame source dir (defaults to the raw frames)")
    ap.add_argument("--gap", type=float, default=2.0, help="run-segmentation time gap (s)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024, help="match the detector's train imgsz")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--fps", type=int, default=6, help="output video fps (capture is ~4)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole run")
    ap.add_argument("--out", default=None, help="output dir (default <work_dir>/track/<name>)")
    ap.add_argument("--list-runs", action="store_true", help="print runs and exit")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        sys.exit(f"weights not found: {args.weights}")

    df = build_manifest(Path(args.source))
    if df.empty:
        sys.exit(f"no frames under {args.source}")
    runs = segment_runs(df, gap_s=args.gap)

    if args.list_runs:
        summary = (runs.groupby(["run_id", "label"]).size()
                   .reset_index(name="frames").sort_values("frames", ascending=False))
        print(f"{'run_id':>6}  {'label':<16} {'frames':>6}")
        for _, r in summary.iterrows():
            print(f"{r['run_id']:>6}  {r['label']:<16} {r['frames']:>6}")
        return

    rid, grp = pick_run(runs, args.label, args.run_id)
    grp = grp.sort_values("timestamp").reset_index(drop=True)
    if args.max_frames:
        grp = grp.head(args.max_frames)
    label = grp["label"].iloc[0]
    name = f"{label}_run{rid}"
    out = Path(args.out) if args.out else paths["work_dir"] / "track" / name
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    print(f"tracking run {rid} ({label}): {len(grp)} frames -> {out}")

    from ultralytics import YOLO
    model = YOLO(args.weights)

    h, w = read_bgr(grp["path"].iloc[0]).shape[:2]
    vw = cv2.VideoWriter(str(out / "track.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    best: dict[int, tuple[float, np.ndarray]] = {}   # track id -> (conf, crop)
    rows: list[str] = []
    seen_ids: set[int] = set()

    for fi, r in grp.iterrows():
        bgr = normalize_illumination(read_bgr(r["path"]))
        res = model.track(bgr, persist=True, tracker=args.tracker, conf=args.conf,
                          imgsz=args.imgsz, device=args.device, verbose=False)[0]
        disp = bgr.copy()
        if res.boxes is not None and res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), tid, cf in zip(xyxy, ids, confs):
                tid = int(tid)
                seen_ids.add(tid)
                x1, y1, x2, y2 = (int(v) for v in (x1, y1, x2, y2))
                col = _color(tid)
                cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2)
                cv2.putText(disp, f"#{tid} {cf:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
                rows.append(f"{fi},{r['filename']},{tid},{x1},{y1},{x2},{y2},{cf:.4f}")
                # keep the highest-conf crop of each battery for the OCR phase
                if cf > best.get(tid, (0.0, None))[0]:
                    crop = bgr[max(0, y1):y2, max(0, x1):x2].copy()
                    if crop.size:
                        best[tid] = (float(cf), crop)
        cv2.putText(disp, f"{label}  run {rid}  frame {fi+1}/{len(grp)}  "
                          f"batteries so far: {len(seen_ids)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        vw.write(disp)
    vw.release()

    for tid, (cf, crop) in best.items():
        cv2.imwrite(str(crops_dir / f"{label}__run{rid}__id{tid:03d}_c{cf:.2f}.jpg"), crop)

    (out / "tracks.csv").write_text(
        "frame,filename,track_id,x1,y1,x2,y2,conf\n" + "\n".join(rows), encoding="utf-8")

    lengths: dict[int, int] = {}
    for ln in rows:
        t = int(ln.split(",")[2]); lengths[t] = lengths.get(t, 0) + 1
    mean_len = (sum(lengths.values()) / len(lengths)) if lengths else 0
    print(f"\n--- track summary (run {rid}, {label}) ---")
    print(f"  frames processed : {len(grp)}")
    print(f"  unique batteries : {len(seen_ids)}   (mean track length {mean_len:.1f} frames)")
    print(f"  crops for OCR    : {len(best)} -> {crops_dir}")
    print(f"  video            : {out / 'track.mp4'}")
    print(f"  tracks csv       : {out / 'tracks.csv'}")


if __name__ == "__main__":
    main()
