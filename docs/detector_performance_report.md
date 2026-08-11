# Detector performance: a correction

**To:** Prof. Bin Chen
**From:** Frank Cai
**Date:** 2026-08-11
**Re:** Phase-1 detector — corrected evaluation, and withdrawal of the belt-lighting recommendation

---

## Cover note

> Prof. Chen,
>
> A correction on the detection results, and it is good news. The recall figure I reported for the
> Phase-1 detector — around 0.45 — was wrong. It was measuring errors in our hand-drawn evaluation
> labels, not errors in the model. Re-labeled properly, the same unmodified detector reads **recall
> 0.941 and precision 0.927**.
>
> The practical consequence is that the belt-lighting upgrade I had been building a case for is
> **not needed**, and I would recommend against spending on it. Detection is not the bottleneck in
> this pipeline and, as far as we can now measure, never was.
>
> The memo below documents how I established this, including the checks I ran against my own bias,
> and what I think the remaining work is. Happy to walk through any of it.
>
> — Frank

---

## 1. Bottom line

| | previously reported | corrected |
|---|---|---|
| recall | 0.452 | **0.941**  95% CI [0.899, 0.966] |
| precision | 0.410 | **0.927**  95% CI [0.883, 0.955] |

Both columns are the **same frozen checkpoint** (`battery_ft1/best.pt`, trained 2026-06-29 and
untouched since), the same 72 evaluation frames, the same matcher, the same confidence threshold.
Nothing about the model changed. Only the ground truth it was scored against changed.

The worst class moved the most: `ni_mh_all`, which I had described in an earlier writeup as
"essentially invisible" to the detector, goes from **0.080 to 0.929**.

| class | old recall | corrected recall |
|---|---|---|
| li_ion_laptop | 0.733 | 0.968 |
| li_ion_mobile | 0.406 | 0.911 |
| liso2 | 0.435 | 1.000 |
| ni_cd_bulk | 0.750 | 1.000 |
| ni_cd_small | 0.435 | 0.917 |
| ni_mh_all | 0.080 | **0.929** |

## 2. What went wrong

The 72-frame evaluation set was hand-labeled early in the project, before I had a written
convention for where a box goes. Measured against the complete set now, those original labels carry
a systematic geometric error: boxes sit about **21 px low** and are drawn about **8% larger** than
the object. Individually each looked fine on screen. Collectively they are enough to push a correct
detection below the IoU 0.5 threshold that recall is scored at.

The failure is worse than a simple undercount, because a misplaced label damages the score twice.
The detector's correct box fails to match any label and is counted a **false positive**; the
displaced label fails to match any detection and is counted a **false negative**. That is why
precision (0.410) looked nearly as bad as recall (0.452) — one object, two penalties. It is also
why the deficit looked like a property of the imagery: it was concentrated in the small, dark
classes, which are exactly the classes where a fixed placement error is large relative to the
object.

I compounded this by spending three rounds of experiments optimizing against that broken ruler.
Resolution, model capacity, ensembling, preprocessing, temporal stacking, more training labels — all
returned "no improvement," which I read as evidence of a hard imagery ceiling. In hindsight they
returned no improvement because the metric could not register one.

## 3. How the correction was established

Three methods, deliberately chosen to fail in different ways. They agree.

**(a) Blind A/B adjudication** — all 102 disputed boxes, ground truth and detector box drawn in
randomly assigned colors so the judge could not tell which was which. Result: 84 favored the
detector's box, 1 favored the label. Corrected recall **0.928**.

**(b) Independent 3-judge panel** — judges not told which box source was which. **34 of 36 panels**
favored the detector's box over the ground truth (p ≈ 2e-9). Of 108 visibility ratings, 96 "obvious",
12 "subtle", **0 "invisible"** — including all 30 ratings for `ni_mh_all`, the class I had called
invisible.

**(c) Full re-label from blank frames** — the definitive test. All 72 frames re-labeled in a mode
where the detector is never loaded, so the labels cannot be seeded from it. 202 boxes. Recall
**0.941**, precision **0.927**.

An earlier attempt at (c) was **invalid and discarded**: it used the labeling tool's default, which
pre-fills each frame with the detector's own output. It scored the detector at recall 1.000 on all
six classes, with 66 of 72 files byte-identical to raw model output. A label set seeded from a model
cannot measure that model. Flagging this because it is the trap that would most easily have produced
a flattering and meaningless number.

### The obvious objection: I had seen the detector's boxes

I drew the new labels, and I had spent weeks looking at this detector's output. Even in blank mode,
memory is a real contamination risk. So I measured it. Matching every label to its nearest detector
box:

| label set | n | median IoU vs detector | fraction > 0.95 | vertical offset | size ratio |
|---|---|---|---|---|---|
| v1 (original) | 173 | 0.494 | 0.0% | +20.6 px | 1.078 |
| v3 (blank re-label) | 199 | **0.886** | **9.5%** | **+1.7 px** | **1.018** |

Labels reproduced from memory of the detector would cluster near IoU 1.0. The new set sits at 0.886
with only 9.5% above 0.95 — agreeing on *which object is there* while disagreeing on the exact
pixels, which is what two careful independent annotators produce. It also shows neither defect of
the original set: no vertical bias (+1.7 px vs +20.6 px) and boxes essentially tight (1.018 vs
1.078). The old set's errors are gone rather than transferred.

The set was also drawn in two sittings of 36 frames. The first half was measured and reported
before the second was drawn; the full 72 moved the result by less than the width of the confidence
interval. So the first half was not a lucky draw.

## 4. The lighting recommendation is withdrawn

I had been assembling a case for a belt-lighting upgrade, on the theory that dark battery bodies
against a dark conveyor were unresolvable. **I no longer believe this, and I would not spend on it.**
To my knowledge this never reached you as a formal request — please tell me if it did and I will
correct it directly.

After the re-label, **3 of 202 objects (1.5%)** are genuine total misses. That is the entire
population that could justify a hardware change, so I inspected all three:

| class | box | what it actually is |
|---|---|---|
| ni_cd_small | x1 = 0, width 20 px | a cell entering the frame — a 20 px sliver clipped at the border |
| ni_mh_all | x1 = 0, width 10 px | the same, a 10 px sliver |
| li_ion_mobile | x1 = 19, 54 × 82 px | a battery overlapping its neighbor — a separation question |

All three sit at the **left edge of the frame**. Two are objects the belt has not finished carrying
into view; the tracker sees both fully resolved one or two frames later, so in the deployed pipeline
they are not missed at all. The third is a question about where one object ends and the next begins.

**None is a dark object lost against a dark belt.** There is no residual population at any
measurable level that better lighting would address.

One related note: the same "the imagery is the wall" reasoning appears in my Phase-2 OCR writeup,
where classical OCR failed and I attributed it to contrast. A vision-language model subsequently
read brands, part numbers and certification marks off those same crops. That is the same shape of
error — a method limitation I read as a sensor limitation — and I would treat the OCR conclusion as
provisional until it gets the same scrutiny.

## 5. What I am not claiming

- **The evaluation set is 72 frames.** Confidence intervals are given above and are not narrow.
  0.941 should be read as "low-to-mid 90s," not as three significant figures.
- **All footage comes from one belt, one camera, one lighting condition.** Nothing here predicts
  performance on a different line. Deployment will need re-calibration and a fresh evaluation set on
  the target hardware — drawn to a written convention from the start, which is the cheapest lesson
  in this memo.
- **This is detection only** — finding batteries, single class. Type classification is a separate
  measurement and is unchanged: 6-way type accuracy 0.874, 4-way chemistry accuracy 0.899 on a
  run-grouped holdout, against a 0.603 majority baseline.
- **The person who built the system also drew the labels and ran the audits.** The blind protocols
  and the echo test above are my attempt to control for that, but they are not a substitute for an
  outside annotator. In the blind adjudication, color is blinded while box *geometry* is not, so a
  rater can partially self-unblind; that is why adjudication reads slightly high (98.8% vs the
  independent panel's 94.4%, p = 0.047). If you want a number that does not depend on my judgment
  at all, the cheapest path is a few hundred frames labeled by someone else against
  `docs/labeling_convention.md`.
- **The previously published figures came from a slightly different evaluation script** (P 0.466,
  R 0.446). The 0.452 / 0.410 pair above is that same checkpoint under the Phase-4 harness, which
  reproduces the recall figure and differs modestly on precision. The comparison that matters is
  within one harness — which is exactly what the table in §1 is.

## 6. Where the project stands

The full pipeline is closed and runs end to end with one command: detect → track → OCR → classify,
producing an annotated video and a per-battery manifest.

| phase | headline |
|---|---|
| 1a detect | **recall 0.941 · precision 0.927** (this memo) |
| 1b track | stable per-battery IDs across all 103 capture runs; 756 batteries |
| 2 OCR | brand (14%) and part number (15%) trustworthy; chemistry/voltage/capacity are prior-filled by the model and must not be used |
| 3 classify | 6-way type 0.874 · 4-way chemistry 0.899, run-grouped holdout (majority 0.603) |

## 7. What I would do next

1. **Nothing on lighting.** The question is closed.
2. **Precision, not recall.** At 0.927 with 15 unmatched detections, precision is now the weaker
   number and the only one with measurable headroom. It is also the one that matters for a sorting
   line, where a false detection puts a phantom item in a bin.
3. **An outside annotator on a few hundred frames**, if you want performance numbers that are
   independent of me. This is the highest-value small ask I can make.
4. **Re-examine the Phase-2 OCR ceiling** for the same class of error, per §4.
5. **Not more training data.** Adding labels plateaued long ago — 36 hand-labeled frames captured
   essentially all the available gain, and 201 added nothing.

Full technical record: `docs/recall_ceiling_round3.md`. Evaluation labels are archived in the
repository at `eval_set/labels_v3/`, and the comparison in §1 reproduces with a single command:

```bash
python scripts/compare_labelsets.py --v2 <eval_dir>/labels_v3
```
