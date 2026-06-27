# Why the detector caps at ~0.45 recall — and what merge does / doesn't fix

Investigation (2026-06-27) into the honest-eval result (P 0.23 · R 0.42 · mAP50 0.19). Goal was
to implement SAM mask de-duplication/merging, re-pseudo-label, and retrain. The investigation
re-rooted the diagnosis before spending GPU on a full re-label.

## What I built
`batterycv/merge.py` — `merge_boxes()`: containment-drop (remove a box mostly inside a larger
one) + NMS + optional agglomerate (union still-overlapping survivors). Wired into
`pseudo_label_sam.py` (now exposes keep_mask + merge + pps as flags via a shared
`frame_to_boxes`). Probes: `probe_labeler.py` (SAM recall vs hand-GT, per class),
`yoloworld_probe.py` (open-vocab labeler), `dump_overlays.py` (visual GT-vs-box).

## Finding 1 — merge is a precision win, not a recall fix
On the trained detector's own predictions (offline, vs 72 hand labels), pre-filtering to
conf≥0.5 then containment-merging: boxes/frame 14→5.6, **precision 0.085→0.197** (>2×), recall
essentially held (0.473→0.434). So fragmentation is real and merge cleans it. But:

## Finding 2 — recall is capped at ~0.45 by the imagery, upstream of any merge
Recall@0.5 *ceiling* (best case, accept all boxes) measured three independent ways on the 72
hand-labeled frames:

| class          | trained YOLO11s | SAM auto-mask (pps24) | YOLO-World "battery" |
|----------------|----------------:|----------------------:|---------------------:|
| li_ion_laptop  | 0.90 | 0.83 | 0.76 |
| ni_cd_bulk     | 0.81 | 0.69 | 0.75 |
| liso2          | 0.78 | 0.65 | 0.48 |
| ni_cd_small    | 0.43 | 0.43 | 0.48 |
| li_ion_mobile  | 0.36 | 0.33 | 0.30 |
| ni_mh_all      | 0.08 | 0.00 | 0.08 |
| **TOTAL**      | **0.51** | **0.45** | **0.43** |

Three methods, one wall, same classes missed. CLAHE, denser SAM sampling (pps 24→48), richer
prompts, and lower conf did **not** move it. YOLO-World gives ~3× the precision of SAM
(whole-object boxes, 2–9/frame vs 17–37) at the same recall.

## Root cause (verified visually, `docs`/overlays)
The dark battery **body** blends into the dark, textured belt (frames ~70/255). Zero-shot
models segment only the **bright printed label** inside each cell — so boxes land undersized at
IoU 0.3–0.49, just under the 0.5 bar — or miss the cell entirely amid belt-dirt false
positives. Hand GT was checked against the overlays and is correct; the small/dark classes
(ni_mh AA-style cells, mobile, ni_cd_small) are at/below the zero-shot detection limit on this
imagery. ni_mh is essentially invisible (≤0.08 everywhere).

## Implication
Re-pseudo-labeling with SAM+merge and retraining will improve **precision** but cannot raise
recall above ~0.45 — the detector can't learn batteries the labeler never boxed. Breaking the
ceiling needs whole-object supervision on *this* domain, not a better zero-shot labeler:
- **Best zero-shot baseline:** switch labeler SAM→YOLO-World (+merge) — cleaner, whole-object,
  3× precision, and far cheaper than the 2.5 GPU-hr SAM pass. Recall still ~0.45.
- **Path to deployable recall:** a few hundred hand-labeled frames to fine-tune (semi-supervised
  from the YOLO-World boxes), and/or flag belt **lighting** to the hardware side — better
  contrast on the dark bodies is the cheapest real lever.

## Outcome — labeler swap SAM → YOLO-World (2026-06-27, executed)
New labeler `scripts/pseudo_label_yoloworld.py` (YOLO-World `yolov8x-worldv2`, prompt "battery",
conf 0.05, conservative dedup). Full re-label of 2,421 frames in **1m46s** (vs SAM's ~2.5 hr) →
8,049 whole-object boxes (~3.3/frame vs SAM's ~19). Retrained YOLO11s with identical hyperparams
(80 ep, imgsz 1024, batch 16). Honest eval vs the 72 hand labels:

| metric      | SAM-trained | YOLO-World-trained |
|-------------|------------:|-------------------:|
| precision   | 0.23 | **0.42** |
| recall      | 0.42 | 0.35 |
| mAP@0.5     | 0.19 | 0.19 |
| mAP@0.5:.95 | 0.045 | 0.047 |

Precision **nearly doubled** with far fewer belt false-positives (overlays: clean whole-object
boxes at 0.9–1.0 conf), but **mAP@0.5 is pinned at 0.19** — confirming the recall ceiling is
structural, not a labeler artifact. The swap moves along the same PR frontier (recall→precision)
and can't break it. Net engineering wins regardless: 80× faster labeling, interpretable
whole-object boxes, deployment-grade precision. Weights (gitignored): SAM =
`runs/detect/battery_yolo11/`, YOLO-World = `runs/detect/battery_yolo11_yw/`.
