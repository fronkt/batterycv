# Hand-verified evaluation set (Phase-1 detector)

72 frames (12 per class) hand-labeled for the single `battery` class — the honest test for the
detector. `batterycv-data/` is gitignored, so these labels are archived here as the durable copy.

> **Use `labels_v3/`. `labels/` (v1) is superseded and is kept only as the provenance of the
> published 0.452 recall.** v1 carries a systematic placement error — boxes sit low and ~6% oversized
> — which alone accounts for most of the "recall ceiling" that Phase 4 spent three rounds chasing.
> Scored against v3 the *same frozen detector* reads recall 0.941 / precision 0.927 instead of
> 0.452 / 0.410. See `docs/recall_ceiling_round3.md`. There is no `labels_v2` here on purpose: that
> pass was seeded from the detector and so cannot measure it.

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

## `labels_v3/` — the authoritative set (2026-08-11)

**202 boxes across the same 72 frames**, 1 genuine empty. Same YOLO format as `labels/`.

### Provenance
Drawn by a human in `label_assisted.py --no-prefill`, which never loads the detector, so the labels
cannot echo it. This mattered: an earlier `labels_v2` pass used the tool's default pre-fill, which
seeds each frame from the detector's own output, and it scored that detector at recall 1.000 on all
six classes with 66 of 72 files byte-identical to raw model output. A label set seeded from a model
cannot measure that model. v3 was drawn in two sittings (36 frames each), and the second half moved
the headline by less than the CI width, so the first-half result was not a lucky draw.

Independently corroborated by a blind A/B adjudication of the v1-vs-detector disputes, which put
corrected recall at 0.928 by a completely different method (`scripts/adjudicate_boxes.py`).

### Restore and re-measure

    cp eval_set/labels_v3/*.txt <eval_dir>/labels_v3/
    python scripts/compare_labelsets.py --v2 <eval_dir>/labels_v3

That prints both rulers side by side against one frozen checkpoint. The v1 column reproducing
**0.452** is the control — it confirms the harness, not the model, is constant between the columns.

### A caveat on the v1 archive
`labels/` and `labels.json` here hold **182** boxes, but the working copy in `batterycv-data` holds
**186** — four boxes were added to the working copy after this archive was written. The published
0.452 and every Phase-4 number were computed against the 186-box working copy. Restoring `labels/`
from this archive therefore does *not* exactly reproduce the published baseline. `labels_v3/` has no
such drift: it was copied here the same day it was finished, and matches the working copy at 202.
