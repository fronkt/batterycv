# Hand-verified evaluation set (Phase-1 detector)

72 frames (12 per class) hand-labeled for the single `battery` class — the honest test for the
detector. `batterycv-data/` is gitignored, so these labels are archived here as the durable copy.

- `labels/*.txt` — YOLO format (`0 cx cy w h`, normalized). One file per eval frame stem.
  **182 boxes** across the 72 frames; 4 frames are genuine empties (bare belt) and carry empty
  files — those measure belt false-positives.
- `labels.json` — master store, `{ "<frame>.jpg": [[x1,y1,x2,y2], ...] }` (normalized corners).

## Provenance
Labeled by eye against CLAHE-enhanced + grid-overlaid frames, then verified by overlaying every
box back on its frame (per-class contact sheets). Box coordinates are tight enough for mAP@50 and
exact on empty/occupied classification; mAP@50-95 treats them as a lower bound.

## Re-run the eval
The working copy lives at `batterycv-data/eval/labels/` (where `eval_detection.py` reads it). To
restore it from this archive: `cp eval_set/labels/*.txt <eval_dir>/labels/`, then:

    python scripts/eval_detection.py --weights runs/detect/battery_yolo11/weights/best.pt --device cpu

## Result (2026-06-26, best.pt = YOLO11s on 46k SAM pseudo-boxes)
P 0.23 · R 0.42 · mAP@50 0.19 · mAP@50-95 0.045 — the detector over-segments (SAM-inherited: one
battery → many sub-part boxes) and fires on bare belt. See `tasks/todo.md` Review for the full
diagnosis and fix path.
