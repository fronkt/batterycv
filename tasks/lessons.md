# Lessons / gotchas — batterycv

## Data quality
- **Nested duplicate folder in the delivered zip.** `OneDrive_1_3-7-2025.zip` nests an exact
  copy of `Li_ion_laptop_battery_03_03_25/` *inside* `Li_ion_mobile_battery_03_03_25/`
  (366 files, identical MD5). Naive top-level counting gives mobile=1465 / total=2859, but the
  **true distinct dataset is mobile=1099 / total=2493**. `extract_data.py` auto-quarantines it.
  Pattern to apply: after any extraction, **recursive-count and check for unexpected subfolders**
  before trusting per-class counts; never label by naive top-level grouping.

## Imagery
- Frames are **very dark / low-contrast**; ~2.6x brightness (or CLAHE) is needed before the
  batteries and their text become legible. Always normalize illumination before detection/OCR.
- **Many batteries per frame** (10+), rotated, often clipped at edges. Belt is textured/striped
  (can fool naive edge/threshold detectors → prefer SAM or a trained detector).
- Labels are **session-level (folder)**, not per-object. Every battery in a folder shares that
  folder's type — usable as a weak label, but detection/OCR have **no ground truth** (hence the
  small hand-verified eval set).

## VLM OCR (Phase 2/2b)
- **Coverage % is not accuracy.** Qwen2.5-VL-3B returned chemistry/voltage/capacity on ~98 % of
  698 crops — and the value distributions showed it was prior-filling (chemistry literally
  constant "Li-ion", incl. every LiSO2/NiCd/NiMH cell). Always check the **distribution of
  values against known class truth** before treating a field as signal; "do not guess" prompts
  are not obeyed at 2B–3B scale.
- Trust in order: verbatim-looking strings (part #, brand) > symbols/marks > any field with a
  plausible-default answer (V, mAh, chemistry).

## Tracking (Phase 1b / demos)
- **IoU trackers silently drop objects smaller than the per-frame motion.** The belt moves
  ~156 px/frame; any battery whose box is ≲160 px (ni_cd_small, ni_mh_all) moves its own
  length between frames → consecutive-frame IoU ≈ 0 → ByteTrack tracks never confirm → no
  ID, no crop, no box in the video (run 92: 41 raw detections → 1 unique id). The detector
  was fine — the association was. Fix: **BoT-SORT with global motion compensation** (the
  textured belt gives GMC a strong translation estimate) → run 92: 1 → 7 batteries.
- The failure was invisible in aggregate metrics (96.6% agreement!) because the dropped
  batteries never became rows. **Sanity-check COUNTS per class against the footage, not
  just accuracy of what survived** — absence doesn't show up in accuracy.
- Corollary: dense classes (li_ion_mobile, same ~167 px boxes) didn't collapse — neighbors
  provided accidental IoU matches → inflated/ID-switched tracks instead. Counts from IoU
  tracking on small fast objects are soft in both directions.

## Evaluation / metrics (Phase 4, 2026-08-06)
- **A metric can only be as good as its ground truth, and nobody audits the ground truth.**
  The eval set's 186 boxes were drawn through a vision model, not by a metrologist. Scoring the
  detector against them at IoU 0.5 produced a "0.45 recall ceiling" that drove ~4 weeks of work
  and a hardware recommendation. Loosening only the IoU threshold (0.5→0.3) raises recall
  0.452→0.790; asking the convention-free question "is a detection's centre inside the GT box"
  gives 0.839. The detector was finding the batteries all along.
- **Look at the failures as images before theorising about them.** One contact sheet of
  near-misses showed the pattern instantly (GT boxes loose/oversized/offset, detector boxes
  tight and correct) — something no aggregate table over 4 weeks had surfaced. Same lesson as
  the ByteTrack drop-out, one level up: aggregates hide the mechanism.
- **Check whether the proposed cause correlates with the effect before spending on the cure.**
  The documented root cause was "dark bodies blend into the dark belt". Measured object-vs-belt
  contrast inside tight detector boxes is *anti*-correlated with recall: ni_mh_all 6.6σ and
  ni_cd_small 10.7σ (the two highest-contrast classes) have the worst recall, while ni_cd_bulk
  2.1σ and li_ion_laptop 3.2σ (the lowest) have the best. One cheap plot would have falsified
  the darkness story before the lighting ask was drafted.
- **Measure photometry inside TIGHT boxes.** Contrast measured inside the loose GT boxes read
  ~15-18 grey levels because the boxes included belt; measured inside tight detector boxes the
  same objects read 35-56 for the small-cell classes. A loose box silently dilutes any
  appearance statistic computed from it.
- **Fit-then-test, even for a one-parameter "fix".** A global box-scale correction looked like a
  free win in-sample (best scale 1.10, +0.006 recall). Fitting the scale on half the frames and
  scoring the other half gave −0.011 and −0.053 — it was fitting noise. One-parameter fixes feel
  too small to need a held-out split; they aren't.
- **Validate a shift estimate against object displacement, not whole-frame residual.** Brute-force
  minimising whole-frame mean absdiff said the belt was static (dx*≈0) and nearly produced a
  confident "the 156 px/frame figure is an artifact" claim. The belt genuinely moves ~100-150
  px/frame — detected box centroids march 1212→1070→919→764. The belt is near-featureless and
  objects cover a tiny area fraction, so whole-frame residual is insensitive to the true shift.

## Remote-box transfers / downloads
- **Never trust a downloader's exit code — verify artifacts.** On a box with flaky HF egress,
  `snapshot_download` exited 0 with zero weight shards on disk; a completion marker keyed on
  exit code fired spuriously. Gate on file count + byte sizes + sha256 vs the source.
- Long single-stream transfers die on cheap-box links. Resumable pattern without rsync on
  Windows: `dd iflag=skip_bytes,count_bytes skip=$(remote byte size) count=256M | ssh 'cat >> f'`
  in a loop until sizes match, then sha256 both ends.
- Prefer shipping small derived artifacts (698 crops = 19 MB) over staging raw data onto
  disk-tight shared boxes; run the heavy model where the disk is.

## Evaluation / metrics (Phase 4 round 3, 2026-08-07)

- **A label set seeded from the model cannot measure that model.** Pre-filling labels from the
  detector and accepting/rejecting boxes yields recall ~1.0 by construction — every box in the set
  came from the detector, so "recall" degenerates into "how many of your own boxes did we keep."
  Two full 72-frame passes produced exactly that (1.000 on all six classes, 66/72 files
  byte-identical to raw detector output, zero boxes added). If the answer must be independent of
  the model, the labels must be drawn without the model on screen.
- **When a tool can only show what was FOUND, nothing that was MISSED will ever be added.**
  The labeler displayed detector boxes and nothing else, so misplaced/absent objects had no
  on-screen cue and silently vanished. Any review UI needs an explicit channel for "something
  should be here that isn't" — a reference overlay, a count, a flag.
- **Prefer adjudication to re-annotation when the question is "which of these is right".** 102
  blind A/B keypresses answered in 15 minutes what a full re-label could not answer at all. Design
  it so unjudged items count AGAINST the hypothesis, so a partial pass can only under-state.
- **Blinding the colour does not blind the geometry.** The rater favoured the detector 84/85 vs an
  independent panel's 34/36 (binomial p=0.047). GT boxes were systematically larger and lower, so a
  rater who knows the signature can self-unblind. State the residual bias rather than claiming
  "blind."
- **A regression slope below 1 is usually noise, not scale.** Check `slope(y|x) × slope(x|y)`:
  a real scale error gives 1, pure measurement noise gives r². Here both axes gave r² to four
  decimals, killing a "vertical compression / letterbox bug" story before it reached the doc.
- **A displaced label is charged twice** — once as a false negative and once as a false positive
  from the orphan detection beside it. 11/12 cases here. Don't diagnose precision and recall as
  separate problems before checking whether one misplacement explains both.
- **Test the tool you're about to blame.** `WINDOW_NORMAL` + raw callback coords looked like a
  perfect explanation for scaled labels; driving the real cursor to known client coordinates showed
  OpenCV maps into image space correctly. Two minutes of experiment beat a plausible story.
- **Synthetic input events are not a substitute for real ones.** Posted WM_MOUSEMOVE messages
  returned an identical coordinate for three different probes — OpenCV reads `GetCursorPos`, not
  lParam. Identical outputs across distinct inputs means the harness is broken, not that the
  hypothesis is confirmed.
- **Sorted file order is class-ordered here.** Any partial pass over the eval frames without
  stratification samples one class — and it's `li_ion_laptop`, the best-performing one.
