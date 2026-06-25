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
- [ ] `make_val_split.py` — sample ~12/class eval frames → **hand-label (user)**
- [ ] `pseudo_label_sam.py` — SAM auto-boxes → YOLO train set (on GPU box)
- [ ] `vast/setup.sh` — provision RTX 5090 (cu128), stage data
- [ ] `train_detector.py` — YOLO11 single-class battery
- [ ] `eval_detection.py` — mAP / recall / precision vs hand-verified set
- [ ] `track.py` — ByteTrack per-battery IDs over a run; export crops
- [ ] Push to GitHub (`batterycv`) after `gh auth login`

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
- _Detection metrics:_ TBD (after GPU training)
- _Tracking sanity:_ TBD
