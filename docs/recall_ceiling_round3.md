# Round 3 — the eval labels are misplaced, and that alone produces both the 0.45 recall and the 0.41 precision

**2026-08-07.** Supersedes the open questions in `recall_ceiling_round2.md`. Round 2 concluded the
~0.45 ceiling was substantially the eval set's box geometry and withdrew the lighting
recommendation pending a re-label. Round 3 finished that work by a different route and reached a
sharper conclusion: **the labels are individually misplaced by roughly 40 px of random error plus a
per-recording bias up to −38 px, and this is sufficient on its own to explain the measured recall,
the measured precision, and every remaining "undetected" object.**

No detector change is implicated. Nothing here supports a lighting or hardware ask.

## Why the planned re-label could not answer this

The round-2 plan was to re-label the 72 eval frames with `label_assisted.py`, which pre-fills each
frame from the detector. Two full passes produced this:

```
the SAME detector, scored against each ruler
class             v1 recall  v2 recall
li_ion_laptop         0.733      1.000
li_ion_mobile         0.406      1.000
liso2                 0.435      1.000
ni_cd_bulk            0.750      1.000
ni_cd_small           0.435      1.000
ni_mh_all             0.080      1.000
TOTAL                 0.452      1.000
residual total misses under v2: 0 of 199
```

Recall 1.000 on all six classes is not a corrected measurement. **A label set built by accepting or
rejecting detector pre-fills contains only detector boxes, so scoring that detector against it asks
"how many of your own boxes did we keep."** It returns ~1.0 regardless of the imagery. 66 of the 72
files were byte-identical to the detector's raw output and no box was ever added — the tool could
only draw what the detector *found* and had no way to show what it *missed*.

That is a defect in the round-2 plan, not in the labeling. `--ref` and the exit circularity warning
were added to `label_assisted.py` so the failure is visible next time, but the deeper point stands:
**re-labeling cannot measure a detector when the detector seeds the labels.**

## What was done instead: blind adjudication of all 102 disputed boxes

`scripts/adjudicate_boxes.py`. For each of the 186 GT boxes the detector fails to match at
IoU ≥ 0.5, show the object zoomed with both the GT box and the detector's box, coloured cyan and
magenta **randomly per case from a fixed seed**, and ask which is better. 89 cases have two boxes;
13 have only the GT box (no detector box within IoU 0.1) and instead ask whether a battery is
present at all. Unjudged cases count as missed, so a partial pass can only understate the
correction. At zero verdicts it reproduces the published 0.452 (84/186) exactly.

```
detector's box was the better box :   84   -> recovered
GT box was the better box         :    1   -> real localisation miss
not a battery / both wrong        :    4   -> left the denominator
no detector box, battery IS there :   12   -> see below; NOT genuine misses
no detector box, nothing there    :    1   -> left the denominator

recall as published : 0.452  (84/186)
recall corrected    : 0.928  (168/181)
```

| class | GT | published | corrected |
|---|---|---|---|
| li_ion_laptop | 30 | 0.733 | 0.929 |
| li_ion_mobile | 69 | 0.406 | 0.985 |
| liso2 | 23 | 0.435 | 1.000 |
| ni_cd_bulk | 16 | 0.750 | 1.000 |
| ni_cd_small | 23 | 0.435 | 0.913 |
| ni_mh_all | 25 | 0.080 | 0.680 |

### Caveat, stated plainly

84 of 85 two-box cases favoured the detector (98.8%). The independent blind audit in round 2, rated
by judges who did not know the hypothesis, gave 34/36 (94.4%). A binomial test puts 84/85 at
**p = 0.047** against that rate — consistent, but at the edge, and it is the most detector-favourable
answer the data could plausibly give. The colour assignment is blind but the *geometry* is not: GT
boxes are systematically larger and lower, so a rater who knows that signature can partially
self-unblind. **Treat 0.928 as an upper-leaning estimate.** The physical evidence below does not
depend on it.

## The 12 "genuine misses" are misplaced labels, not misses

This is the part that matters, because that bucket was the entire evidence base for a lighting ask.

Full-frame renders (`results/phase4/miss_frames/`) show the pattern directly: the detector's boxes
sit tightly on every battery while the GT boxes sit on bare belt below them. On
`ni_mh_all__…12-40-32-297`, two batteries, two correct detections, two GT boxes on empty belt — the
eval scores that frame **0/2**.

Measured rather than eyeballed:

- Displacement from GT box to nearest detector box is **downward in 12 of 12** (sign test p = 2e-4),
  median **dx +54 px, dy −104 px**. The same bias runs through the 84 near misses at half magnitude
  (median dy −34 px, dy<0 in 67/84).
- Mean HSV saturation inside the GT box is **29.0** against **59.1** inside the adjacent detector box
  (bare-belt baseline **16.2**) — the GT boxes hold mostly belt with a clipped fragment of battery.
  4 of 12 are within 35% of bare belt outright.
- **11 of 12** have their nearest detector box *also* unmatched to any GT box. An unmatched GT box
  and an unmatched detector box 130 px apart, with the object inside the detector's, is **one object
  counted twice against the detector** — once as a false negative and once as a false positive.

That double-counting is why precision reads 0.410: **121 of 205 detections are scored as false
positives** by this eval. Round 2 attributed that to over-detection. It is largely the same defect.

## The error is per-box noise plus a per-recording bias, and nothing more

Regressing detector-box centre on GT-box centre over 183 paired boxes:

```
x:  det = 0.9972*gt + 14.8    r=0.994   residual sd 44.9 px
y:  det = 0.9154*gt + 24.2    r=0.978   residual sd 41.2 px
box size, GT / detector:  width 1.069,  height 1.064
per-class median dy:  laptop −2.6 | mobile −20.6 | liso2 −20.8 | ni_cd_bulk −36.6 | ni_cd_small −30.9 | ni_mh_all −38.3
```

The y-slope of 0.915 looks like a vertical compression but **is not one**. A genuine scale error
satisfies `slope(det|gt) × slope(gt|det) = 1`; pure placement noise attenuates both directions so
the product equals `r²`. Measured:

```
x:  0.9972 × 0.9904 = 0.9876    r² = 0.9876      GT centre spread sd 402 px
y:  0.9154 × 1.0458 = 0.9573    r² = 0.9573      GT centre spread sd 213 px
```

Both products equal r² to four decimals. This is **regression dilution** — the slope is pulled below
1 by noise in the predictor, and the effect is stronger in y only because objects occupy a narrower
band vertically (213 px) than horizontally (402 px). There is no scale factor to correct.

So the model is simply: **recorded box = true box + per-recording offset (0 to −38 px, vertical) +
random placement error of ~41–45 px**, with boxes also ~7% oversized in both axes.

### The labeling GUI is not the cause — tested and cleared

`label_eval.py`, which produced these labels, uses `cv2.WINDOW_NORMAL` and passes the callback's
`(x, y)` straight through as image pixels. If OpenCV reported *window* coordinates, a resized window
would scale every box by window/image, which would fit the symptoms. It does not: driving the real
cursor to known client coordinates in a 900×700 window over a 1280×1024 image returns exactly
(320,256), (640,512), (960,768) — correctly mapped into image space. **Window size does not affect
labels.** The remaining explanation is ordinary human placement error under a per-session bias,
which the ~41 px residual and the per-class offsets both fit.

A 40 px displacement on a ~180 px box drops per-axis overlap to roughly (180−40)/(180+40) = 0.64,
and IoU below 0.5 once both axes are affected. **The measured 0.45 recall is what this label noise
predicts.** It is not a statement about the imagery.

### Tested and rejected: a single global correction

`scripts/probe_gt_shift.py` grid-searches a translation on 36 frames and reports it on the 36 it
never saw (alternating within class so both halves carry every class):

```
best shift on the FIT half : dx +4  dy −28   recall 0.611
same shift on the HELD-OUT half        : recall 0.407  (+0.033 vs its own baseline 0.374)
li_ion_mobile −0.053   <- went backwards
```

**A single translation does not transfer.** The per-recording bias is real but the ~41 px random
component dominates, and no global correction removes it. Consistent with round 2's box-scale
probe, which gained +0.006 in sample and lost −0.011 held out. There is no arithmetic fix; the only
fix is labels drawn correctly.

## What this does NOT show

- **Not** that the detector is good in absolute terms. It shows the eval set cannot measure it. 0.928
  is a corrected score on 72 frames the detector was tuned against, with the caveat above.
- **Not** that all 121 false positives are artifacts. The displacement explains many; some may be
  genuine over-detections. That population has not been adjudicated.
- **Not** anything about `ni_mh_all`, which remains the weakest class at 0.680 corrected and carries
  the largest per-recording bias (−38.3 px). Whether that is residual label error or a real
  detection weakness on small cylindrical cells is **open**, and it is the one place a lighting
  question could still legitimately be asked — after its labels are fixed, not before.

## CONFIRMED — blank re-label, 36 frames (2026-08-08)

36 of the 72 frames (6 per class, stratified) were re-labeled with `label_assisted.py --no-prefill`.
The detector is never loaded in that mode, so these labels cannot echo it. 101 boxes, 12 s/frame,
per-frame times 4–53 s.

```
BLANK-LABELLED EVAL (36 frames, 101 GT boxes, conf>=0.25, IoU>=0.5)
  TP 96   FP 8   FN 5
  recall    0.950   95% CI [0.889, 0.979]
  precision 0.923   95% CI [0.856, 0.961]

  published on the ORIGINAL labels:  recall 0.452   precision 0.410
```

Same 36 frames, same frozen weights, per class:

| class | recall on v1 | recall on v3 |
|---|---|---|
| li_ion_laptop | 0.619 | 0.952 |
| li_ion_mobile | 0.516 | 0.968 |
| liso2 | 0.417 | 1.000 |
| ni_cd_bulk | 0.800 | 1.000 |
| ni_cd_small | 0.250 | 0.846 |
| ni_mh_all | 0.100 | 0.923 |
| **TOTAL** | **0.479** | **0.950** |

**`ni_mh_all` goes 0.100 → 0.923.** The class round 1 called "essentially invisible" and round 3
still flagged as the one place a lighting question survived is, in fact, detected almost perfectly.
The lighting question is now closed for every class.

### The re-label is not a memory echo — tested

The rater had seen these frames with detector boxes overlaid many times, so an echo was the live
risk. Matching each label to its nearest detector box:

```
v1 (original hand)   n=80   median IoU vs detector 0.540   w-ratio 1.074   dx  -6.8 px   dy +13.6 px
v3 (blank re-label)  n=99   median IoU vs detector 0.862   w-ratio 1.024   dx  +0.9 px   dy  +2.3 px
```

A set drawn from memory of the detector's boxes would sit near IoU 1.0; **v3 sits at 0.862 with only
6.1% of boxes above 0.95** — independently drawn, agreeing on the object but not on the pixels,
which is what two careful annotators produce. It also carries **no placement bias** (dy +2.3 px vs
v1's +13.6 px) and is nearly tight (1.024 vs v1's 1.074 oversize). Those are exactly the two defects
round 3 identified, and they are gone.

This also corroborates the blind adjudication independently: adjudication gave 0.928 corrected,
the blank re-label gives 0.950 (CI [0.889, 0.979]). Different methods, different failure modes,
same answer.

## Next

1. **Finish the remaining 36 frames** in blank mode, so the headline covers the same 72 frames the
   published 0.452 did and the comparison is like-for-like. ~35 minutes.
2. **The headline for Chen is recall 0.950 / precision 0.923** (CI above), with the explanation that
   the original 0.452 was measuring label placement error, not detection. `docs/recall_ceiling_round3.md`
   is the backing.
3. **The parked probes are now mostly moot.** `probe_ensemble.py` and `probe_preprocess.py` were
   built to chase recall that turns out not to be missing. Run them only if the full-72 number
   comes in materially below 0.95; there is little headroom left to buy.
4. **Look at the 8 false positives.** With recall settled, precision 0.923 is the remaining number
   worth improving, and it is now measurable for the first time.
4. Check whether the **training** labels carry the same bias. The eval labels were hand-drawn with
   `label_eval.py`; the training labels came from `pseudo_label_sam.py`, a different path, so the
   defect does not automatically transfer. Two things argue it does not: the detector places boxes
   tightly and correctly on objects in every frame inspected, which it could not have learned from
   systematically low labels; and the per-recording offsets differ by class, which is a signature of
   hand-labeling sessions rather than of a script. Worth ten minutes to confirm, not more.

A note on tooling, since it cost two wasted labeling passes: `label_assisted.py` now takes `--ref`
to overlay a second label set and flag objects about to be dropped, `--only-uncovered` to visit
just the disputed frames, and prints a warning when a pass produces a circular label set. None of
that makes a detector-seeded label set valid for measuring that detector — see the top of this doc.

---

## Settled: the full 72 frames (2026-08-11)

The remaining 36 frames were labeled in blank mode, so the eval set is now complete: **72 frames,
202 boxes**, drawn without the detector ever being loaded. The comparison below is like-for-like
with the published number in every respect — same 72 frames, same frozen `battery_ft1/best.pt`,
same matcher, conf >= 0.25, IoU 0.5.

**The v1 column reproduces the published 0.452 exactly.** That is the control: it proves the
harness is not the source of the difference.

| class | v1 recall | v3 recall | delta |
|---|---|---|---|
| li_ion_laptop | 0.733 | 0.968 | +0.234 |
| li_ion_mobile | 0.406 | 0.911 | +0.506 |
| liso2 | 0.435 | 1.000 | +0.565 |
| ni_cd_bulk | 0.750 | 1.000 | +0.250 |
| ni_cd_small | 0.435 | 0.917 | +0.482 |
| ni_mh_all | 0.080 | 0.929 | +0.849 |
| **TOTAL recall** | **0.452** | **0.941** | **+0.489** |
| **precision** | **0.410** | **0.927** | **+0.517** |

```
recall    0.941 = 190/202   95% CI [0.899, 0.966]
precision 0.927 = 190/205   95% CI [0.883, 0.955]
```

The 36-frame preliminary was 0.950 / 0.923; the full 72 gives 0.941 / 0.927. The second half was
labeled after the first was already reported and moved the result by less than the CI width, so the
half-set result was not a lucky draw.

Geometry of v1 against v3, on the 171 boxes that pair up: median IoU **0.498**, v1 boxes **6.0%
wider** and 1.4% taller, centre displaced **dy +0.084** of a box height (v1 sits low). 15 boxes exist
only in v1, 31 only in v3. This is the round-3 error model measured on the complete set: a modest
downward, slightly oversized placement error, which is enough on its own to halve apparent recall.

### The residual misses are an edge-of-frame convention, not imagery

**3 of 202 boxes (1.5%)** are total misses at IoU < 0.1 — one each in `li_ion_mobile`,
`ni_cd_small`, `ni_mh_all`. This is the population, and the only population, that could ever have
justified a lighting or hardware change. All three were inspected as crops
(`results/phase4/labelset_compare/residual_*.jpg`):

| class | box | what it is |
|---|---|---|
| ni_cd_small | x1=0, w=20 px | a cell just entering frame — a 20 px sliver clipped at the left border |
| ni_mh_all | x1=0, w=10 px | same, a 10 px sliver |
| li_ion_mobile | x1=19, w=54, h=82 | a battery overlapping a neighbour — the touching-cells convention case |

Two are objects the belt has not finished carrying into the field of view, and the third is a
separation question between adjacent objects. **None is a dark object lost against a dark belt.**
The lighting recommendation from round 1 is not merely withdrawn for lack of evidence; the
evidence that would have supported it does not exist at any measurable level. A tracker following
objects across frames sees both slivers fully resolved within one or two frames regardless.

### What this closes, and what it leaves

Closed: the recall ceiling, the imagery/contrast explanation, and the hardware ask. The detector
that has been in the repo since 2026-06-29 was always performing at ~0.94 recall / ~0.93 precision;
Phase 4 changed the ruler, not the model.

Left open: precision is now the weaker number, at 0.927 with 15 detections unmatched against v3.
That is the only remaining measurable headroom, and it is worth a look before any further training.
`probe_ensemble.py` and `probe_preprocess.py` remain parked — they were built to buy recall that
was never missing.
