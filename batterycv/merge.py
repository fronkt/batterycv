"""De-duplicate / merge overlapping detection boxes.

The SAM auto-mask generator (and a detector trained on its output) over-segments a single
battery into many sub-part boxes — one per cell, label, logo, terminal — plus, sometimes, a
box around the whole battery. That fragmentation tanks precision and depresses IoU-0.5 recall
(fragments don't match whole-battery ground truth).

`merge_boxes` collapses a fragmented set back toward one-box-per-object in three passes:

  1. containment drop  — remove a box that sits (mostly) inside a larger box. Kills the
     label/cell/logo fragments that live within a whole-battery box.
  2. NMS               — greedy score-ordered non-max suppression for plain duplicates.
  3. agglomerate       — union any survivors that still overlap appreciably into one box.
     Catches batteries that were tiled by fragments with no single parent box.

Used at pseudo-label time (clean the SAM boxes before training) and, optionally, at inference
(post-process the detector output). Pure-numpy, no torch dependency.
"""
from __future__ import annotations

import numpy as np


def _pairwise_inter(boxes: np.ndarray) -> np.ndarray:
    """N×N intersection-area matrix for boxes given as [x1,y1,x2,y2]."""
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def merge_boxes(
    boxes,
    scores=None,
    iou_thresh: float = 0.5,
    contain_thresh: float = 0.75,
    agglomerate: bool = True,
    agg_iou: float = 0.2,
):
    """Collapse fragmented boxes toward one-box-per-object.

    Parameters
    ----------
    boxes : (N,4) array-like of [x1,y1,x2,y2] (any consistent units; pixels expected).
    scores : (N,) array-like, optional. Defaults to box area (prefer bigger / whole-object).
    iou_thresh : NMS IoU above which the lower-scored duplicate is dropped.
    contain_thresh : drop a box if this fraction of its area lies inside a *larger* box.
    agglomerate : union survivors that still overlap (IoU >= agg_iou) into one box.
    agg_iou : overlap gate for the agglomerate pass (low, since fragments tile rather than
              stack — but > 0 so distinct, merely-adjacent objects are not fused).

    Returns
    -------
    (boxes_out, scores_out) as float arrays. Empty input -> two empty arrays.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    n = len(boxes)
    if n == 0:
        return boxes, np.zeros(0)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    scores = areas.copy() if scores is None else np.asarray(scores, dtype=float).reshape(-1)

    inter = _pairwise_inter(boxes)

    # --- pass 1: containment drop (area-descending so bigger boxes win) ---
    order = np.argsort(-areas)
    kept_mask = np.ones(n, dtype=bool)
    kept: list[int] = []
    for i in order:
        a_i = areas[i]
        drop = False
        for j in kept:                      # kept boxes are all >= a_i in area
            if areas[j] <= 0:
                continue
            if inter[i, j] / max(a_i, 1e-9) >= contain_thresh:
                drop = True
                break
        if drop:
            kept_mask[i] = False
        else:
            kept.append(i)
    idx = np.array(kept, dtype=int)

    # --- pass 2: NMS (score-descending) ---
    idx = idx[np.argsort(-scores[idx])]
    keep2: list[int] = []
    for i in idx:
        ok = True
        for j in keep2:
            union = areas[i] + areas[j] - inter[i, j]
            if union > 0 and inter[i, j] / union > iou_thresh:
                ok = False
                break
        if ok:
            keep2.append(i)
    idx = np.array(keep2, dtype=int)

    out_boxes = boxes[idx]
    out_scores = scores[idx]
    if not agglomerate or len(idx) < 2:
        return out_boxes, out_scores

    # --- pass 3: agglomerate (union-find over IoU >= agg_iou) ---
    m = len(idx)
    sub = _pairwise_inter(out_boxes)
    a = (out_boxes[:, 2] - out_boxes[:, 0]) * (out_boxes[:, 3] - out_boxes[:, 1])
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in range(m):
        for q in range(p + 1, m):
            union = a[p] + a[q] - sub[p, q]
            if union > 0 and sub[p, q] / union >= agg_iou:
                parent[find(p)] = find(q)

    groups: dict[int, list[int]] = {}
    for p in range(m):
        groups.setdefault(find(p), []).append(p)

    merged_boxes, merged_scores = [], []
    for members in groups.values():
        mb = out_boxes[members]
        merged_boxes.append([mb[:, 0].min(), mb[:, 1].min(), mb[:, 2].max(), mb[:, 3].max()])
        merged_scores.append(out_scores[members].max())
    return np.asarray(merged_boxes, dtype=float), np.asarray(merged_scores, dtype=float)
