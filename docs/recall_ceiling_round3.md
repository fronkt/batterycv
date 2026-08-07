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

## Next

1. **Do not send the current numbers to Chen.** Neither 0.452 nor 0.928 is a defensible headline.
2. **Re-label the eval set from scratch, not from detector pre-fills** — blank frames, the written
   convention in `docs/labeling_convention.md`, and no detector seeding. That is the only way to get
   a number that measures the detector rather than echoing it.
3. **Then** re-run `analyze_misses.py`, `probe_ensemble.py`, and `probe_preprocess.py`, which were
   deliberately left unrun. Tuning against the current ruler would repeat the round-1 mistake.
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
