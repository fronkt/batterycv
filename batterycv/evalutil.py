"""Shared, exact yardstick for detection probes — one matcher, one set of numbers.

Every recall/precision figure in `docs/recall_ceiling_findings.md` came from the greedy
IoU-0.5 matcher in `scripts/probe_labeler.py`. Phase-4 probes (motion, ensemble,
preprocessing) must be measured the *same* way or their numbers can't be compared to the
published table. This module is that matcher, lifted out and made scoring-aware.

Two distinct metrics, deliberately kept separate — conflating them is how the earlier
"recall ceiling" tables get misread:

  recall_ceiling  accept EVERY candidate box, no score threshold. "Could this method
                  produce a >=0.5-IoU box for this object at all?" This is the number in
                  the 3-way labeler table (YOLO11s 0.51 / SAM 0.45 / YOLO-World 0.43).
  pr_at_conf      score-thresholded precision/recall — deployment behaviour.

`ap50` is an all-point-interpolated AP over the score-sorted PR curve. It is NOT
Ultralytics' mAP50 (different matching + interpolation), so compare ap50 only against
other numbers from THIS module. `validate_harness.py` quantifies the offset.

GT convention: eval labels are YOLO-normalized `cls cx cy w h` on 1280x1024, and the file
stem is `<class>__<frame stem>` — the class prefix is the run's weak folder label.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

IMG_W, IMG_H = 1280, 1024
CLASSES = ["li_ion_laptop", "li_ion_mobile", "liso2", "ni_cd_bulk", "ni_cd_small", "ni_mh_all"]


def class_of(stem: str) -> str:
    """`li_ion_laptop__acA1300_..._03-05-25` -> `li_ion_laptop`."""
    return stem.split("__")[0]


def load_gt(labels_dir, w: int = IMG_W, h: int = IMG_H) -> dict[str, np.ndarray]:
    """Read YOLO-normalized labels -> {stem: (N,4) xyxy pixels}. Empty files stay as (0,4)."""
    gt: dict[str, np.ndarray] = {}
    for f in sorted(Path(labels_dir).glob("*.txt")):
        boxes = []
        for ln in f.read_text().splitlines():
            if ln.strip():
                _, cx, cy, bw, bh = map(float, ln.split())
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h])
        gt[f.stem] = np.array(boxes, float).reshape(-1, 4)
    return gt


def iou_mat(a, b) -> np.ndarray:
    """(Na,4) x (Nb,4) -> (Na,Nb) IoU."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def greedy_match(boxes, gt_boxes, scores=None, iou: float = 0.5):
    """Score-ordered greedy matching. Returns (tp_flags, matched_gt_mask).

    Boxes are consumed high-score-first (arbitrary order if `scores` is None); each GT can
    be claimed once. tp_flags[i] tells whether box i matched — the caller aggregates.
    """
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    n, g = len(boxes), len(gt_boxes)
    tp_flags = np.zeros(n, bool)
    matched = np.zeros(g, bool)
    if n == 0 or g == 0:
        return tp_flags, matched
    order = np.arange(n) if scores is None else np.argsort(-np.asarray(scores, float).reshape(-1))
    im = iou_mat(boxes, gt_boxes)
    for i in order:
        cand = np.where(~matched & (im[i] >= iou))[0]
        if len(cand):
            j = cand[np.argmax(im[i, cand])]
            matched[j] = True
            tp_flags[i] = True
    return tp_flags, matched


class Accumulator:
    """Per-class TP/FP/GT tallies plus score records for AP."""

    def __init__(self) -> None:
        self.stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.records: list[tuple[float, bool]] = []   # (score, is_tp) across all frames
        self.n_gt_total = 0
        self.n_frames = 0

    def add(self, stem: str, boxes, gt_boxes, scores=None, iou: float = 0.5) -> None:
        cls = class_of(stem)
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
        tp_flags, _ = greedy_match(boxes, gt_boxes, scores, iou)
        s = self.stats[cls]
        s[0] += int(tp_flags.sum())
        s[1] += int((~tp_flags).sum())
        s[2] += len(gt_boxes)
        s[3] += len(boxes)
        self.n_gt_total += len(gt_boxes)
        self.n_frames += 1
        if scores is not None and len(boxes):
            sc = np.asarray(scores, float).reshape(-1)
            self.records.extend(zip(sc.tolist(), tp_flags.tolist()))

    def recall(self, cls: str | None = None) -> float:
        tp, _, ngt = self._agg(cls)[:3]
        return tp / max(ngt, 1)

    def precision(self, cls: str | None = None) -> float:
        tp, fp = self._agg(cls)[:2]
        return tp / max(tp + fp, 1)

    def _agg(self, cls: str | None):
        if cls is not None:
            s = self.stats[cls]
            return s[0], s[1], s[2], s[3]
        tp = sum(v[0] for v in self.stats.values())
        fp = sum(v[1] for v in self.stats.values())
        ng = sum(v[2] for v in self.stats.values())
        nb = sum(v[3] for v in self.stats.values())
        return tp, fp, ng, nb

    def ap50(self) -> float:
        """All-point-interpolated AP from the score-sorted records. Needs scores."""
        if not self.records or self.n_gt_total == 0:
            return float("nan")
        recs = sorted(self.records, key=lambda r: -r[0])
        tps = np.array([r[1] for r in recs], float)
        ctp = np.cumsum(tps)
        cfp = np.cumsum(1 - tps)
        rec = ctp / self.n_gt_total
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        # monotonically decreasing precision envelope, then integrate over recall
        mrec = np.concatenate(([0.0], rec, [rec[-1]]))
        mpre = np.concatenate(([1.0], prec, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    def table(self, title: str, classes: list[str] | None = None) -> str:
        classes = classes or [c for c in CLASSES if c in self.stats]
        out = [f"\n=== {title} ===",
               f"{'class':<16}{'recall':>8}{'prec':>8}{'nGT':>6}{'box/f':>8}"]
        for c in classes:
            tp, fp, ng, nb = self._agg(c)
            out.append(f"{c:<16}{tp/max(ng,1):>8.3f}{tp/max(tp+fp,1):>8.3f}"
                       f"{ng:>6}{nb/max(self.n_frames,1):>8.1f}")
        tp, fp, ng, nb = self._agg(None)
        out.append(f"{'TOTAL':<16}{tp/max(ng,1):>8.3f}{tp/max(tp+fp,1):>8.3f}"
                   f"{ng:>6}{nb/max(self.n_frames,1):>8.1f}")
        ap = self.ap50()
        if ap == ap:  # not NaN
            out.append(f"{'AP@0.5':<16}{ap:>8.3f}   (this harness, not Ultralytics mAP50)")
        return "\n".join(out)


def coverage(boxes_by_stem: dict[str, np.ndarray], gt: dict[str, np.ndarray],
             iou: float = 0.5) -> dict[str, np.ndarray]:
    """Which GT objects does a candidate set cover? -> {stem: bool mask over that frame's GT}.

    The unit of complementarity analysis: two methods are complementary when their masks
    differ, not when their aggregate recalls differ.
    """
    cov: dict[str, np.ndarray] = {}
    for stem, gt_boxes in gt.items():
        b = boxes_by_stem.get(stem, np.zeros((0, 4)))
        _, matched = greedy_match(b, gt_boxes, None, iou)
        cov[stem] = matched
    return cov


def union_recall(*cov_maps: dict[str, np.ndarray]) -> float:
    """Recall of the union of several methods' coverage masks (the oracle-fusion ceiling)."""
    tot = hit = 0
    keys = set().union(*[set(c) for c in cov_maps]) if cov_maps else set()
    for k in keys:
        masks = [c[k] for c in cov_maps if k in c and len(c[k])]
        if not masks:
            continue
        u = np.zeros_like(masks[0])
        for m in masks:
            u |= m
        tot += len(u)
        hit += int(u.sum())
    return hit / max(tot, 1)


def per_class_coverage(cov: dict[str, np.ndarray]) -> dict[str, tuple[int, int]]:
    """{class: (covered, total)} from a coverage map."""
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for stem, mask in cov.items():
        a = agg[class_of(stem)]
        a[0] += int(mask.sum())
        a[1] += len(mask)
    return {k: (v[0], v[1]) for k, v in agg.items()}
