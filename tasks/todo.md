# batterycv — Phase 1: Detection + Tracking

Plan: detection + tracking first → Vast.ai GPU from day one → small hand-verified eval set.
Full plan: `../.claude/plans/buzzing-tinkering-panda.md` (or repo `docs/` once copied).

## Checklist
- [x] Scaffold repo + configs + package modules
- [x] Create `.venv`
- [x] Install dependencies (`requirements.txt`) — torch 2.12 CPU, ultralytics 8.4.78
- [x] `extract_data.py` — extract zip, auto-quarantine nested duplicate
- [x] Data verified: **2,493 distinct frames**, 6 classes (mobile corrected 1465→1099)
- [x] `build_manifest.py` — manifest CSV, 2493 rows, 100% timestamps, 103 runs @ gap 2s
- [x] `eda.py` — brightness mean ~70/255 (dark, uniform across classes); ~4 fps capture
- [x] `make_val_split.py` — 72 eval frames sampled + CLAHE-normalized (`eval/images`)
- [x] hand-label the 72 eval frames → `eval/labels` (182 boxes, 4 genuine-empty frames; CLAHE+grid assisted, overlay-verified)
- [x] Turnkey Vast path: `bmp_to_jpg.py` (9 GB→0.8 GB JPEG stage), `vast/{setup,run_pipeline,stage_data,pull_results}.sh`, `vast/README.md`
- [x] `BATTERYCV_PATHS` env override + manifest globs jpg/bmp (box trains on compact JPEGs)
- [x] **GPU run DONE (RTX 5090, 2026-06-26):** SAM pseudo-label → YOLO11s train
  - [x] `pseudo_label_sam.py` — 2421 frames, **46,125 boxes** (train=2163, val=258, ~19/frame, pps=24)
  - [x] `train_detector.py` — YOLO11s 80ep imgsz1024, 26 min; best.pt pulled to `runs/detect/battery_yolo11/`
  - [x] `eval_detection.py` on hand labels — **honest: P .23 R .42 mAP50 .19 mAP50-95 .045** (vs pseudo-val .79/.79/.86/.77). Diagnosis below.
- [x] `track.py` — ByteTrack per-battery IDs over a run; export crops (DONE 2026-06-29)
- [x] Push to GitHub (`fronkt/batterycv`, main) — initial scaffold live

## Phase 2 — OCR on tracked crops (DONE 2026-06-30, see docs/ocr_findings.md)
- [x] `ocr_crops.py` built (pluggable `--engine qwen|easyocr`; outputs ocr.json/ocr.csv/vis → Phase 3)
- [x] EasyOCR baseline — **fails on this imagery** (4 garbage fragments across 6 crops); same
  dark/low-contrast wall as detection. Kept as `--engine easyocr` for comparison.
- [x] Engine decision (user): **local Qwen2-VL** (open VLM) — free/offline/reproducible
- [x] Qwen2-VL-2B run on the 6 demo crops (laptop CPU, ~80s/crop) → structured fields. **Reads the
  signal classical OCR can't:** brand (DELL, confirmed vs sticker), chemistry (Li-ion), all 6 crops'
  cert marks (CE/WEEE/UL/RoHS), and a real part # (HP 727897-001). Fine specs (V/cap) lower-conf
  (some real e.g. 11.55V/41Wh, some hallucinated) — small model invents plausible numbers; dense
  micro-print still unread.
- [x] Documented (docs/ocr_findings.md) + requirements updated (transformers/qwen-vl-utils/easyocr)
- Note: HF unauth download throttling was the real friction; fixed via plain-HTTPS resume loop
  (HF_HUB_DISABLE_XET=1). Bigger model (Qwen2.5-VL-3B/7B + --device cuda) lifts fine-field fidelity.

## Phase 2b — OCR at SCALE (DONE 2026-07-02, see docs/ocr_findings.md "Scaled run")
- [x] `track.py` looped over ALL 103 runs locally (CPU, 0 failures) → **698 crops** pooled
- [x] Qwen2.5-VL-3B (bf16, rented RTX 3090) over all 698 → **698/698 records, 50 min (~4.3 s/crop)**
- [x] Results archived in-repo: `results/phase2_ocr/ocr.{json,csv}` + 2 evidence panels
- [x] **Honest verdict: 3B scales the hallucination, doesn't fix it.** chemistry = "Li-ion" on
  100% of non-empty rows incl. all LiSO2/NiCd/NiMH (constant → zero class signal); voltage 94%
  ∈ {3.7V, 11.55V}; capacity 54% = 2600mAh. TRUST: model/part# (15%) + manufacturer (14%,
  spot-checked vs print) > marks >> chemistry/V/cap (do NOT use in Phase 3).
- [x] Infra: box HF egress flaky (one download attempt exited 0 with NO weight shards — gate on
  artifacts+sha256, never exit codes); model shipped from laptop HF cache via resumable
  dd-offset loop over ssh (256 MB rounds), loaded from local path (no network).

## Phase 3 — type classification (DONE 2026-07-03, see docs/type_classifier_findings.md)
- [x] `train_type_classifier.py` — one script: run-grouped split → yolo11n-cls fine-tune →
  one honest eval on held-out runs → OCR/visual/fusion ablation. 14 min on laptop CPU.
- [x] **Group split by run id** (25 held-out runs / 199 crops vs 70 / 499), all 6 classes both
  sides; minority train classes oversampled capped ×10 (1,424 train files).
- [x] **RESULT: visual solves it — 6-way acc 0.874 / macro-F1 0.819, 4-way chemistry acc
  0.899 / macro-F1 0.870** (majority baseline 0.603); best.pt == last.pt (converged, not a
  lucky epoch). Weights `runs/classify/type_v1/` (gitignored).
- [x] **Trust-tier vindicated downstream:** trusted-OCR-features-only LR = 0.206 (below
  majority — brand/part# accurate but sparse/type-agnostic); fusion adds nothing (−0.5 pt).
  OCR's role = per-item metadata, NOT the sorting decision.
- [x] Weak spot: **LiSO2 recall 0.68** at P 1.00 (thinnest class by runs: 6). Levers: more
  LiSO2 sessions > threshold/class-weight trade > lighting. li_ion clean at P/R 0.95/0.95.
- [x] Artifacts archived: `results/phase3_type/metrics.{json},predictions.csv`; findings doc.
- Pipeline loop CLOSED: detect → track → OCR → classify all functional on this imagery.
- Optional fidelity pass still parked: 7B (needs ≥25 GB-disk box) or transcribe-then-parse.

## Wrap-up — package the finished pipeline (DONE 2026-07-13)
- [x] `scripts/run_pipeline.py` — ONE command, frames → detect+track → classify type →
  join trusted OCR metadata (brand/part#/marks from results/phase2_ocr by run+track id) →
  annotated video (boxes colored by predicted chemistry, bin counts in header) +
  per-battery `batteries.csv`. Two-pass render so every frame shows final types.
- [x] Verified on 3 runs across chemistries: li_ion_laptop run4 **6/6** (part #s GT-10U /
  LX6092 / B403XT overlaid), liso2 run60 **15/21** (0.71 ≈ the holdout LiSO2 recall 0.68;
  misclassifications visibly lower-conf 0.54–0.88 vs ~1.00 correct — threshold story on
  display), ni_cd_bulk run79 **9/9**. Demo agreement is illustrative — the honest numbers
  remain the Phase-3 run-grouped holdout (runs here may overlap classifier training).
- [x] Demo GIF (560px, 16 frames, 4.1 MB) → `results/demo/pipeline_demo.gif`.
- [x] README rewritten: one-command demo up top + GIF, per-phase results table, the two
  project-level lessons (lighting is the lever; VLM coverage ≠ accuracy), repro section.

## Review (fill in as steps complete)
- _Data:_ delivered zip had a nested duplicate of the laptop folder inside mobile (366 exact
  dupes); quarantined. True dataset = 2,493 frames. See `lessons.md`.
- _EDA:_ 100% of filenames parsed to timestamps. Capture ≈ **4 fps** (median 0.25s gap), slow
  belt → tracking very feasible. Mean grayscale brightness **~70/255 and uniform across all 6
  classes** → CLAHE required before detect/OCR, and brightness can't act as a class shortcut.
  Run count: 103 @ 2s gap / 60 @ 3.8s gap (median ~20 frames per run).
- _Preprocessing check:_ `normalize_illumination` (CLAHE) makes batteries + text legible
  (SAMSUNG/LG/Panasonic, mAh, recycling symbols readable) — good omen for the OCR phase.
- _Classical detector:_ runs end-to-end but is a weak fallback — misses dark batteries that
  blend into the belt and fires on belt texture. Confirms SAM pseudo-labels are the right
  primary path for the trained detector.
- _Transfer:_ raw is 9.13 GB uncompressed BMP; re-encoding to JPEG q95 (`bmp_to_jpg.py`) gives a
  0.81 GB stage (11.3× smaller, visually lossless) that tar-pipes to the box in minutes — and is
  consistent with the already-JPEG eval set.
- _Detection (held-out pseudo-val, 258 frames / 5138 boxes):_ P 0.787, R 0.794, mAP50 0.859,
  mAP50-95 0.771. **Caveat:** measured vs SAM pseudo-labels, which share the detector's belt-FP
  bias → optimistic. Honest precision needs the hand-labeled eval set.
- _Qualitative (best.pt on eval frames):_ reliably boxes real laptop/mobile batteries at high conf
  (0.9+); **over-fires on empty belt texture and frame edges**, worst on sparse frames (low-conf
  0.3–0.5 ghosts). Next levers: tighten SAM `keep_mask` (belt rejection), raise inference conf,
  and use the hand-labeled eval to tune conf/NMS. SAM pass is CPU-bound (GPU ~7% util) — a future
  speedup is parallel SAM workers or vit_b.
- _Honest eval (72 hand-labeled frames, 182 boxes, IoU 0.5):_ **P 0.23, R 0.42, mAP50 0.19,
  mAP50-95 0.045** — far below the 0.86 pseudo-val (which shared SAM's bias). Threshold sweep:
  precision never exceeds ~0.21 even at conf≥0.9 (303 FP vs 78 TP); recall caps at ~0.47 at any
  conf. Root causes, diagnosed from GT-vs-pred overlays:
  1. **Over-segmentation (dominant):** the detector learned SAM's habit of splitting one battery
     into many sub-part boxes — a dense laptop frame has 6 real batteries but 71 predictions ≥0.5
     (boxes on every label/cell/logo). Kills precision and depresses IoU-0.5 recall (fragments
     don't match whole-battery GT).
  2. **Belt false positives:** fires 6 confident boxes on a totally empty belt frame (seam/edges).
  3. Isolated bright cells (ni_mh) are handled well — confirms the labels are fair and the problem
     is the pseudo-label strategy, not the eval. Fix path: dedup/merge SAM masks (NMS + drop masks
     contained in larger ones) so one battery = one box; tighten belt rejection; consider
     `min_mask_region_area`↑ and whole-object prompting. Re-pseudo-label → retrain is the real fix.
- _Merge + recall-ceiling investigation (2026-06-27, see `docs/recall_ceiling_findings.md`):_
  Implemented `batterycv/merge.py` (containment-drop + NMS + agglomerate) and wired it into
  `pseudo_label_sam.py` (keep_mask/merge/pps now flags). **Merge is a precision win** (detector
  output, conf≥0.5: boxes/frame 14→5.6, P 0.085→0.197, recall held) **but not a recall fix.**
  Probed the recall ceiling 3 ways on the 72 hand frames: trained YOLO11s 0.51, SAM auto-mask
  0.45, YOLO-World "battery" 0.43 — **same wall, same classes missed** (ni_mh ~0.08, mobile
  ~0.30). CLAHE / pps 24→48 / multi-prompt / lower conf did not move it. Root cause (verified on
  GT-vs-box overlays): dark battery bodies blend into the dark belt; zero-shot models box only
  the bright label → undersized IoU 0.3–0.49. GT confirmed correct. **Re-label+retrain on
  SAM+merge will lift precision, not recall.**
- _Labeler swap SAM→YOLO-World DONE (2026-06-27, user chose option B):_ new
  `scripts/pseudo_label_yoloworld.py` (yolov8x-worldv2, "battery", conf 0.05) re-labeled 2,421
  frames in **1m46s** (vs SAM 2.5 hr) → 8,049 whole-object boxes; retrained YOLO11s (same
  hyperparams). Honest eval vs hand GT: **P 0.42 · R 0.35 · mAP50 0.19 · mAP50-95 0.047** — vs
  SAM-trained P 0.23 · R 0.42 · mAP50 0.19. **Precision ~2× (belt-FP storm gone, clean
  whole-object boxes), mAP50 pinned at 0.19 = the structural recall ceiling.** Weights:
  `runs/detect/battery_yolo11_yw/weights/best.pt` (gitignored; SAM baseline kept at
  `battery_yolo11/`).
- _Resolution/capacity sweep DONE (2026-06-28):_ retrained s@1280 and m@1280 (same YOLO-World
  labels). s@1280 **P 0.40 R 0.38 mAP50 0.194**, m@1280 P 0.42 R 0.36 mAP50 0.187 — vs s@1024
  P 0.42 R 0.35 mAP50 0.19. Native 1280 nudged recall +0.03 but **mAP50 pinned ~0.19 across all
  res/model sizes** → ceiling is the imagery, not resolution/capacity/epochs. Best operating point
  = `runs/detect/battery_yw_s1280/weights/best.pt` (kept). Only remaining recall levers:
  hand-labeled fine-tune (~150–300 frames) and/or belt **lighting** (hardware).
- _Fine-tune tooling BUILT (2026-06-29):_ `scripts/build_label_pool.py` (stratified, eval-excluded,
  CLAHE'd sampling → `batterycv-data/label_pool/`) + `scripts/label_assisted.py` (detector
  pre-fills boxes; right-click=delete FP, drag=add miss, resumable). This is the path past the
  0.19/0.45 ceiling.
- _Fine-tune EXPERIMENT 1 (2026-06-29):_ user hand-labeled the 36-frame demo pool (6/class, 81
  boxes, 3 genuine-empty) via the assisted GUI. New `scripts/finetune_detector.py` continues from
  `battery_yw_s1280` best.pt on the pool — explicit AdamW lr0=0.001 + cos_lr (NB: `optimizer=auto`
  silently overrides lr0 to ~0.002, too hot for a tiny set from a good init → must pass optimizer
  explicitly), 80 ep, imgsz 1280, all 6 classes balanced so easy classes aren't forgotten. Baseline
  to beat (same eval harness, 72 frames): **P 0.446 · R 0.409 · mAP50 0.224**. **RESULT — it works:**
  ft1 best.pt **P 0.466 · R 0.446 · mAP50 0.252** (last.pt P 0.476 · R 0.435 · mAP50 0.250). Every
  metric up; **mAP50 +0.028 is the first thing to move it off the ~0.19–0.22 wall** that held across
  all zero-shot labelers / resolutions / model sizes. Gain shows in both best AND last → real, not a
  lucky epoch. Mechanism confirmed: human whole-object boxes teach the dark/small cells. Trained
  imgsz 1024 / 40 ep / AdamW lr0 0.001 cos_lr, ~63 min on laptop CPU. best.pt kept at
  `runs/detect/battery_ft1/weights/` (gitignored). **Next: scale the pool** `build_label_pool.py
  --per-class 30` (~180 frames) → label assisted → re-run finetune; 36 frames gave +0.03–0.04, 180
  should compound. (Gotcha logged: `optimizer=auto` overrides lr0; two stray trainers raced the same
  save_dir once — always confirm a single PID + use a fresh run name.)
- _Pool scaled 36→201, gain PLATEAUS (2026-06-29, GPU):_ user labeled full 201-frame pool (560 boxes).
  Re-ran on a Vast RTX 5090 (~6 s/epoch vs ~90 s CPU). Same recipe as ft1 (1024/40ep): ft3@201 best
  **P 0.442 R 0.441 mAP50 0.234** (last 0.455/0.446/0.236) — *below* ft1@36's 0.466/0.446/0.252, a gap
  inside 72-frame eval noise. ft2@201 (1280/60ep) 0.444/0.435/0.224 — overfit, no better. **5.6× more
  labels added nothing.** Fine-tune gives a one-shot bump over baseline (recall 0.41→~0.45, mAP50
  0.224→~0.24) then plateaus; 36 careful frames captured it all. **Imagery ceiling reasserts — the
  step-change lever is belt LIGHTING (hardware), not more labels.** Keeper = `battery_ft1/best.pt`
  (best mAP50, local; ft3 tied within noise). Box used was a shared 1×5090 (also runs STS2027) — do
  NOT destroy it.
- _Tracking DONE (2026-06-29):_ `scripts/track.py` — ByteTrack (`bytetrack.yaml`, needs `lapx`)
  over a timestamp-segmented run via `io.segment_runs`; CLAHE per frame, `model.track(persist=True)`.
  Auto-picks the longest run of `--label` (or `--run-id`/`--list-runs`). Outputs annotated
  `track.mp4`, best-conf crop per track ID (→ OCR phase), `tracks.csv`, summary. Demo on longest
  laptop run (run 4, 32 frames): **6 unique batteries**, mean track len 5 frames, track #1 glides
  855→305 px at ~0.95 conf — clean persistent IDs. Crops land in `<work_dir>/track/<run>/crops`.
  Completes Phase-1 detect→track. Default weights = `battery_ft1/best.pt`. ID counter does fragment
  a bit on the dark classes (re-id when a cell is briefly lost) — fine on laptop/ni_cd_bulk; same
  imagery limit as detection elsewhere.

## Phase 4 — recall ceiling, round 2: the compute-only levers (2026-08-06)

Context: every lever tried in Phase 1 was a variation on ONE family — single-frame, static,
absolute-brightness detection (SAM / YOLO-World / YOLO11 at 3 resolutions, 2 backbones,
36 and 201 hand-labeled frames). All hit the same ~0.45 recall wall, and the doc concluded the
only remaining lever is belt **lighting** (hardware). That conclusion is correct *for that
family*. It was never tested against methods that use information the family throws away:
**time** (the belt moves), **model disagreement** (5 trained checkpoints sit unused on disk),
and **the preprocessing itself** (CLAHE's clip/grid were never swept — every sweep held them fixed).

Goal: determine whether any compute-only lever moves recall on the two classes that carry the
whole deficit (`ni_mh_all` 0.12, `li_ion_mobile` 0.45) BEFORE asking Chen for a lighting rig.
Deliverable either way is a defensible answer: a working lever, or evidence that closes the
question so the hardware ask is backed by more than one family of experiments.

### Ground rules
- [x] One shared matcher for every probe — `batterycv/evalutil.py` (greedy IoU-0.5, the same
  semantics as `probe_labeler.py` that produced the published tables). No probe rolls its own.
- [x] Harness gated against published ft1 before use — `scripts/validate_harness.py`:
  R 0.452 vs published 0.446, AP@0.5 0.233 vs mAP50 0.252, per-class pattern reproduced
  (ni_mh 0.12, mobile 0.45). Precision reads lower (0.410 vs 0.466) only because Ultralytics
  reports P at best-F1, not at a fixed conf. Comparisons are therefore trustworthy.
- Report **per-class** recall always. Aggregate accuracy hid the ByteTrack drop-out for weeks
  (see lessons.md); it will hide this too.
- Complementarity (does method X cover objects the detector misses?) is the decisive question,
  not X's standalone recall. A method with 0.30 recall that covers a *disjoint* 0.30 is worth
  more than one with 0.45 that covers the same objects.

### A — temporal / multi-frame (NOT motion detection — see the physics correction below)
Belt motion re-measured from scratch 2026-08-06, because the whole track depends on it:
- Objects **ride the belt and translate with it** — pure horizontal, dy≈0, and per-run speed
  varies ~30-220 px/frame (not one constant). Measured by tracking detected box centroids:
  run 92 x = 1212→1070→919→764 (~145 px/f), run 4 x = 1067→967→816→664 (~100-150 px/f).
  The ~156 px/frame in `lessons.md` is **confirmed correct**.
- **Consequence: a naive moving-object detector cannot work here.** Objects and belt move
  together, so motion-compensated differencing cancels both. The original framing of this
  track ("find what moves differently from the belt") was wrong and is retracted.
- Method gotcha worth keeping: brute-force searching the shift that minimises WHOLE-FRAME mean
  absdiff returns ~0 and is flatly wrong — the belt is near-featureless and objects cover a tiny
  area fraction, so that average is insensitive to the true shift. It briefly produced a
  confident "the scene is static, the 156 px/frame figure is an artifact" conclusion that the
  object-centroid check demolished. Estimate shift on high-gradient pixels or by phase
  correlation, and always validate against object displacement.

What actually has headroom is using time against **sensor noise**, not against motion:
- [ ] `scripts/probe_temporal.py`, idea A1 — **motion-aligned temporal stacking.** In
  belt-aligned coordinates the scene is static, so warping k in-run neighbors onto the eval
  frame and averaging is pure denoising: measured per-pixel temporal σ ≈ 5.3/255, cut by √k.
  Against a battery-vs-belt contrast of only a few grey levels that is a plausible part of the
  real detection limit. Costs image margin (~150 px per stacked frame) — track a valid-count
  mask and report the degraded area.
- [ ] idea A2 — **temporal-median flat-field.** In camera coordinates (no alignment) the belt
  slides past, so a per-pixel median over a run approximates the *static* components: vignetting
  ("bright center / dark corners", per `preprocess.py`), fixed-pattern noise, lens dirt, mean
  belt level. Divide it out, renormalize, then CLAHE. Standard flat-fielding, independent of A1.
- [ ] Report standalone per-class recall ceiling AND union-with-ft1 coverage (the payoff metric):
  a method that re-finds the same objects is worthless even at equal recall.

### B — ensemble of the checkpoints already on disk
- [ ] `scripts/probe_ensemble.py`. 5 checkpoints exist (`battery_yolo11` SAM-trained,
  `battery_yolo11_yw` YOLO-World-trained, `battery_yw_s1280`, `battery_ft1`, `battery_ft3`).
  They were only ever compared and the winner kept — never fused. Different pseudo-labelers →
  different failure modes → plausibly different misses. Fuse with WBF/NMS, sweep the fusion
  params. Zero training cost.
- [ ] Report the oracle union recall too: the ceiling any fusion rule could reach.

### C — preprocessing sweep (the untuned knob)
- [ ] `scripts/probe_preprocess.py`. `normalize_illumination` is CLAHE clip=2.5 grid=8, fixed
  since day one and never swept. Test clip/grid variants, gamma, multi-scale Retinex, unsharp.
- [ ] Caveat to respect: the detector was TRAINED on clip=2.5/grid=8, so changing inference
  preprocessing risks train/test mismatch and may cost recall. Interpret a drop as mismatch,
  not as evidence the variant is bad — and check the zero-shot labeler (no mismatch) separately.

### D — deferred, needs GPU (spec only, not run this round)
- [ ] Synthetic hard-example augmentation: composite bright crops from the strong classes onto
  real belt at reduced contrast to manufacture the diagnosed failure mode, instead of buying
  more real labels (which plateaued at 36 frames).
- [ ] Segmentation head (yolo11n-seg, bootstrapped from the Phase-1 SAM masks): misses cluster at
  IoU 0.3-0.49, so tighter boundary fitting could push near-misses over the 0.5 bar.
- [ ] Learned low-light enhancement (Zero-DCE class, self-supervised, no paired GT) if C shows
  preprocessing has real headroom.

### Review (2026-08-06) — the question changed underneath the plan

The plan above asks "which compute lever raises recall". The diagnostic that was supposed to
*aim* that search answered a different and more important question first: **the ~0.45 ceiling is
substantially the ruler, not the imagery.** Full writeup in `docs/recall_ceiling_round2.md`;
`docs/recall_ceiling_findings.md` now carries a superseded banner.

- [x] Miss taxonomy (`analyze_misses.py`): only **5.9%** of the 186 GT boxes are genuine total
  misses (IoU<0.1); 32.8% are near-misses at IoU 0.3-0.5. Recall would be 0.817 if near-misses
  alone were fixed.
- [x] Threshold sensitivity (`probe_gt_audit.py`): recall 0.452 @IoU0.5 → **0.790 @IoU0.3**;
  convention-free centre-in-GT = **0.839**. Every class shows a 15-48 pt gap.
- [x] Blind audit, 3 independent judges, not told which box source was which: **34/36 panels
  favour the DETECTOR's box** over the ground truth (91.7% unanimous, sign-test p≈2e-9).
  Visibility 96 "obvious" / 12 "subtle" / **0 "invisible"** out of 108 ratings; all 30 ni_mh_all
  ratings "obvious" — the class documented as "essentially invisible".
- [x] Contrast (`contrast_detboxes.json`): measured in TIGHT detector boxes, contrast is
  **anti-correlated** with recall (ni_mh 6.6σ → recall .08; ni_cd_bulk 2.1σ → recall .875).
- [x] Track A temporal — **clean negative.** Stacking is a low-pass filter, not a denoiser (object
  edge SNR ×0.97→×0.66) because of 13 px residual misregistration, not interpolation. Flat-field
  corrects a real 1.89× vignette but lowers recall. All 10 variants: **zero** new GT covered.
- [x] Box-scale correction — **clean negative**, and a good reminder to hold out even for a
  one-parameter fix (+0.006 in-sample, −0.011/−0.053 held out).
- [x] Matcher adversarially reviewed; AP@0.5 verified exact; 6 bugs fixed; divergence from the
  original `probe_labeler.match` bounded at +0.0019 recall, so Phase-4 and published numbers
  are comparable.
- [ ] **BLOCKED, and deliberately so — Tracks B (ensemble), C (preprocessing) and the inference
  resolution/tiling sweep are built and smoke-tested but NOT run to conclusions.** Running them
  to three decimals against labels now known to be the binding error would repeat exactly the
  round-1 mistake. They are cheap to run the moment the eval set is re-labeled.
- [ ] **The one action everything else waits on: re-label the 72 eval frames to a WRITTEN box
  convention, by a human.** Minimum spec: does an attached wire/connector belong inside the box
  (the most common single disagreement); how are touching cells separated; how are frame-clipped
  objects handled.
- [ ] Only after re-measuring: revisit whether the residual 5.9% total-miss population justifies
  a hardware/lighting ask. That population is the only evidence that could still support one, and
  the blind audit is structurally blind to it (no detector box ⇒ no panel to judge).
