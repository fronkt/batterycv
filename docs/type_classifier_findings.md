# Phase 3 — battery type classifier (visual + trusted-OCR fusion)

Goal: close the pipeline loop — detect → track → OCR → **classify each tracked battery's type**
so a sorting line can divert it. `scripts/train_type_classifier.py` trains and evaluates in one
pass on the 698 Phase-2b crops; session (folder) labels are the ground truth. Run 2026-07-03,
laptop CPU, **14 min** end to end.

## Setup — the honest cut

- **Split grouped by run id, never by crop.** Crops from one run share a conveyor session
  (same batteries, same lighting), so a crop-level split would leak. Per class ~25 % of runs →
  test: 70 train runs / 499 crops vs **25 held-out runs / 199 crops**, all 6 classes on both
  sides. Evaluated once, for both best.pt and last.pt (identical — converged, not a lucky pick).
- **Visual model:** yolo11n-cls fine-tune @ 224, 40 epochs, minority classes oversampled
  (capped ×10 → 1,424 train files). Weights: `runs/classify/type_v1/weights/best.pt` (gitignored).
- **OCR features per the Phase-2b trust tier only:** `n_chars`, brand presence, part-# presence,
  mark count, det_conf. Chemistry/voltage/capacity **excluded** — prior-dominated
  (docs/ocr_findings.md); OCR "chemistry" is constant Li-ion and would only teach the prior.
- Ablation on the same test runs: OCR-features-only logistic regression vs visual-only vs late
  fusion (LR on visual probs + OCR features).

## Results (held-out runs; majority-class baseline = 0.603)

| model | 6-way acc | 6-way macro-F1 | 4-way chem acc | 4-way macro-F1 |
|---|---|---|---|---|
| **visual (yolo11n-cls)** | **0.874** | **0.819** | **0.899** | **0.870** |
| OCR-features-only LR | 0.206 | 0.178 | 0.271 | 0.229 |
| fusion (visual + OCR) | 0.869 | 0.816 | 0.894 | 0.866 |

Per-class, visual, 6-way: li_ion_mobile F1 0.92 (n=120), li_ion_laptop 0.87 (28), liso2 0.81
(25; P 1.00 / R 0.68), ni_cd_bulk 0.75 (23), ni_cd_small 0.57 (n=2), ni_mh_all 1.00 (n=1) —
the last two supports are too small to read.

4-way chemistry confusion (rows = true):

|  | li_ion | liso2 | ni_cd | ni_mh |
|---|---|---|---|---|
| li_ion (148) | **140** | 0 | 8 | 0 |
| liso2 (25) | 4 | **17** | 4 | 0 |
| ni_cd (25) | 4 | 0 | **21** | 0 |
| ni_mh (1) | 0 | 0 | 0 | **1** |

## Findings

1. **Type classification is a visual problem — and it works.** 87 % / 90 % (6-way / chemistry)
   on runs the model never saw, vs a 60 % majority baseline, trained in 14 min on CPU. Form
   factor, size, label layout and color carry the signal the OCR text couldn't.
2. **The trusted OCR features carry ~zero type signal on their own** (20.6 %, *below* majority
   guessing) **and add nothing in fusion** (−0.5 pt vs visual-only). This is the Phase-2b audit
   confirmed downstream: brand/part-# reads are *accurate* but *sparse* (15 %) and
   type-agnostic — a Samsung sticker sits on both mobile and laptop packs. Had we naively fed
   the ~98 %-coverage chemistry field instead, the classifier would have inherited the VLM's
   constant-"Li-ion" prior. OCR's value in this pipeline is per-item *metadata* (who made it,
   which part), not the *sorting decision*.
3. **The weak spot is LiSO2 recall (0.68)** — 8 of 25 held-out LiSO2 crops land in li_ion or
   ni_cd bins. Precision is 1.00 (nothing is falsely flagged LiSO2). For a DOE sorting line
   primary-lithium leakage is the safety-relevant error; see levers below.
4. li_ion is clean at P/R 0.95/0.95; residual errors are li_ion↔ni_cd confusions on dark
   cylindrical cells — the same imagery limit detection hit.

## Caveats

- Labels are session-level: every battery in a run inherits the folder type. The run-grouped
  split removes session memorization, but all footage shares one belt — a deployment on a
  different belt/lighting needs a small re-calibration set.
- liso2 has only 6 runs (2 in test), ni_cd_small/ni_mh_all test supports are 2 and 1 —
  6-way numbers for the small classes are indicative only; the 4-way chemistry view is the
  robust readout.

## Levers if LiSO2 recall must rise

1. More LiSO2 sessions (6 runs is the thinnest class by runs — data, not architecture).
2. Threshold tuning: LiSO2 P=1.00 means its decision boundary is conservative; lowering the
   accept threshold (or class-weighted loss) trades false alarms for recall — the right trade
   when the error is asymmetric (missed primary lithium >> a li_ion mis-binned to LiSO2).
3. Better imagery (lighting) — the ceiling every phase of this project keeps hitting.

Artifacts: `results/phase3_type/metrics.json` (all models, both granularities, confusions) +
`predictions.csv` (per-crop true/pred/prob for the 199 test crops). Reproduce:
`python scripts/train_type_classifier.py` (deterministic at seed 0).
