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

## Deferred (scaffolded only)
- Step 2 OCR on tracked crops · Step 3 type classification (text vs image vs multimodal)

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
