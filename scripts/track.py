"""Detect + track batteries across one temporal run, assigning stable per-battery IDs.

Feeds timestamp-ordered frames of a run through YOLO + ByteTrack (persist=True), saves
annotated frames, and exports one crop per (track_id, frame) for later OCR/classification.

    python scripts/track.py --weights runs/detect/battery_yolo11/weights/best.pt --run 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.io import build_manifest, segment_runs
from batterycv.preprocess import normalize_illumination, read_bgr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--run", type=int, default=0, help="run_id to track")
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = load_paths()
    df = segment_runs(build_manifest(paths["raw_dir"]), gap_s=args.gap)
    run = df[df["run_id"] == args.run].sort_values("timestamp")
    if run.empty:
        sys.exit(f"no frames for run {args.run}")
    label = run["label"].iloc[0]
    print(f"run {args.run}: {len(run)} frames, class={label}")

    out = paths["work_dir"] / "tracks" / f"run{args.run:04d}"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    track_ids: set[int] = set()
    for i, (_, r) in enumerate(run.iterrows()):
        frame = normalize_illumination(read_bgr(r["path"]))
        res = model.track(frame, persist=True, conf=args.conf, device=args.device,
                          tracker="bytetrack.yaml", verbose=False)[0]
        cv2.imwrite(str(out / "frames" / f"{i:04d}.jpg"), res.plot())
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.int().tolist()
            xyxy = res.boxes.xyxy.int().tolist()
            for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                track_ids.add(tid)
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size:
                    d = out / "crops" / f"id{tid:03d}"
                    d.mkdir(exist_ok=True)
                    cv2.imwrite(str(d / f"{i:04d}.jpg"), crop)

    print(f"unique tracks (battery count estimate): {len(track_ids)}")
    print(f"annotated frames -> {out / 'frames'}")
    print(f"per-battery crops -> {out / 'crops'}  (feeds OCR/classification later)")


if __name__ == "__main__":
    main()
