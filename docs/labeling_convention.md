# Eval-set box convention (v2) — read this before drawing a single box

The Phase-4 audit (`recall_ceiling_round2.md`) showed the 72-frame eval set's box geometry, not
the imagery, is what caps measured recall at ~0.45. Blind auditors preferred the *detector's* box
to the ground-truth box in 34 of 36 disputed cases. Re-labeling is the one action everything else
is blocked behind.

The failure that produced the current labels was not carelessness — it was that **no convention
was ever written down**, so each ambiguous case got resolved differently. This document exists to
be decided ONCE, in advance, and then applied mechanically.

## Decide these seven before starting

Defaults are proposed for each. Change any of them if you disagree — what matters is that the
answer is fixed before labeling, not which answer is chosen. Edit this file first, then label.

**1. Attached wires / plastic connectors — INCLUDE them in the box.**
This is the single commonest disagreement in the current set (e.g. a Ni-MH pack whose box scored
IoU 0.45 purely because the detector included the connector and the label did not). Default is
include, for two reasons: a bounding box conventionally bounds the whole physical object, and
"where does the cell end and the lead begin" is a judgement call that will drift between frames,
whereas "everything physically attached" is mechanical. Downstream this is harmless — the type
classifier reads the cell body, which dominates the crop either way.

**2. One box per physically separate battery, even when they touch.**
Cells lying shoulder-to-shoulder each get their own box. Do NOT draw one box around a cluster.
The detector's current failure of merging two adjacent laptop packs into one box is a real error
and must stay visible as one.

**3. A shrink-wrapped / bagged multi-cell PACK is ONE object.**
If the cells are bound together as a single unit that would be sorted as one item, it is one box.
The distinction from (2) is physical attachment, not proximity.

**4. Frame-clipped objects: label if roughly a quarter or more is visible; box only the visible part.**
Do not extrapolate the hidden extent — the metric compares against pixels, so an imagined
continuation is guaranteed disagreement. Below ~25% visible, skip it entirely (and be consistent:
a skipped object must not be labeled in one frame and ignored in the next).

**5. Exclude shadows.** Box the object silhouette, not the shadow it casts on the belt.

**6. Boxes are axis-aligned and TIGHT to the silhouette.**
For a rotated battery this means the axis-aligned box of the rotated object — it will contain
belt in the corners, which is correct and expected. "Tight" means no deliberate margin: the
current set's boxes are systematically larger than the object, which is precisely the defect.

**7. Only batteries.** Belt debris, dirt, scratches, and stray plastic are not labeled.

## How to relabel

Write to a **new** directory. Do not overwrite `labels/` — the comparison between v1 and v2 is
itself a result, and quantifies how much the old ruler was off.

```bash
# from the repo root, with the venv active
python scripts/label_assisted.py \
  --images  <eval_dir>/images \
  --labels  <eval_dir>/labels_v2 \
  --weights runs/detect/battery_ft1/weights/best.pt \
  --imgsz 1024 --conf 0.25
```

`<eval_dir>` is the `eval_dir` entry in `configs/paths.yaml`. Because `labels_v2/` starts empty,
every frame pre-fills from the detector rather than loading the old boxes — which is what you
want, since the detector's boxes were judged the better ones. The tool is resumable: a frame you
have already saved to `labels_v2/` reloads your work, so you can stop and restart freely.

Controls: `left-drag` add a box, `right-click` delete the box under the cursor, `u` undo,
`r` re-run the detector on this frame (discard edits), `n`/`SPACE` save + next, `p` back,
`q` save + quit. A frame saved with zero boxes writes a valid empty label — 4 of the 72 frames
are genuinely bare belt and must stay that way.

Expect roughly 30-60 s per frame (186 boxes over 72 frames, most already pre-filled correctly),
so about an hour total.

**Do a 15-frame pilot first.** Label 15 frames, run the comparison below, and look at the delta
before committing to the remaining 57. If the corrected recall barely moves, that itself is a
finding worth knowing before spending the hour.

## Then quantify what changed

```bash
python scripts/compare_labelsets.py                      # v1 vs v2: geometry delta + re-measured recall
```

This reports, for the frames labeled so far: how many boxes changed, the systematic offset and
size ratio between v1 and v2, and the detector's recall re-measured against each — i.e. how much
of the "0.45 ceiling" was the ruler.

## A note on what re-labeling can and cannot fix

It fixes the near-miss population (32.8% of GT), which is where the audit showed the labels are
the worse box. It does **not** speak to the 11 genuine total misses (IoU < 0.1, 5.9%) — those
have no detector box at all, and they are the only remaining evidence that could support a
lighting change. After relabeling, look at that residual population specifically and directly
(crop and view each one) before any hardware conversation.
