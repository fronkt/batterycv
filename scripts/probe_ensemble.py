"""Fuse the five detector checkpoints already on disk — after first asking whether fusion CAN help.

WHY: five single-class 'battery' detectors sit unused on disk (SAM-pseudo-labeled,
YOLO-World-pseudo-labeled at two input sizes, and two human-fine-tuned). They were only ever
compared head-to-head; the winner (ft1, accept-all recall ceiling 0.489) was kept and the rest
shelved. Two were trained on pseudo-labels from *genuinely different* labelers, which produce
different box statistics, so their failure modes are plausibly decorrelated — and fusing costs
zero training.

The decisive number is therefore computed and reported FIRST, before any fusion variant: the
ORACLE UNION CEILING, i.e. the recall of the union of every model's accept-all coverage mask.
No fusion rule can exceed it, because no fusion rule can invent a box no model proposed. If
the ceiling sits on top of the best single model's 0.489, the models miss the *same* objects
and Track B is closed no matter how clever the fusion rule is. Only if there is headroom do
the fusion variants mean anything, so they are printed after it, not before.

Then: a pairwise complementarity matrix (which models disagree about which GT objects, per
class — the deficit is concentrated in ni_mh_all and li_ion_mobile), and three real fusion
rules — Weighted Box Fusion (implemented here, no new dependency), NMS over the pooled boxes,
and batterycv.merge.merge_boxes' containment+agglomerate pass — swept over IoU clustering
thresholds and two model subsets (all five, and the most-decorrelated triple).

Each checkpoint runs at its OWN training imgsz. Ultralytics rescales predictions back to the
original 1280x1024 frame; this script VERIFIES that instead of assuming it (a box with x2 >
imgsz on a 1024-input model can only exist in original-image coordinates) and refuses to
continue if any model's boxes fall outside the frame. Inference is the expensive part, so raw
predictions are cached under results/phase4/cache/ and every fusion configuration is scored
from that cache — the sweep costs no extra forward passes.

All scoring goes through batterycv.evalutil so the numbers are comparable to the published
table in docs/recall_ceiling_findings.md.

    python scripts/probe_ensemble.py --limit 4 --out results/phase4/ensemble_smoke.json
    python scripts/probe_ensemble.py            # full 72-frame sweep -> results/phase4/ensemble.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.evalutil import (
    CLASSES,
    IMG_H,
    IMG_W,
    Accumulator,
    class_of,
    coverage,
    iou_mat,
    load_gt,
    per_class_coverage,
    union_recall,
)
from batterycv.merge import merge_boxes

# short name -> (weights relative to repo root, native training imgsz, provenance)
MODELS: dict[str, tuple[str, int, str]] = {
    "sam":    ("runs/detect/battery_yolo11/weights/best.pt",    1024, "SAM auto-mask pseudo-labels"),
    "yw":     ("runs/detect/battery_yolo11_yw/weights/best.pt", 1024, "YOLO-World pseudo-labels"),
    "yw1280": ("runs/detect/battery_yw_s1280/weights/best.pt",  1280, "YOLO-World labels, imgsz 1280"),
    "ft1":    ("runs/detect/battery_ft1/weights/best.pt",       1024, "+36 human frames (KEEPER)"),
    "ft3":    ("runs/detect/battery_ft3/weights/best.pt",       1024, "+201 human frames"),
}

IOU_SWEEP = (0.4, 0.55, 0.7)
DEPLOY_CONF = 0.25

# Candidate-set definitions the union ceiling is computed under. `accept_all` is the headline
# number, but it is NOT budget-matched: a model that sprays hundreds of boxes per frame at
# conf 0.001 can "cover" a GT object by accident, which inflates any union it takes part in.
# The other two re-ask the same question at a comparable box budget, so genuine complementarity
# can be told apart from box spam.
BUDGETS: dict[str, object] = {"accept_all": None, "top20": 20, "conf25": "conf"}


def budget_boxes(preds_n: dict[str, tuple[np.ndarray, np.ndarray]],
                 mode: object) -> dict[str, np.ndarray]:
    """Restrict one model's per-frame candidates to a budget (all / top-K by score / conf>=0.25)."""
    out: dict[str, np.ndarray] = {}
    for s, (b, sc) in preds_n.items():
        if mode == "conf":
            out[s] = b[sc >= DEPLOY_CONF]
        elif isinstance(mode, int):
            out[s] = b[np.argsort(-sc, kind="stable")[:mode]]
        else:
            out[s] = b
    return out


# --------------------------------------------------------------------------- inference cache
def _cache_path(cache_dir: Path, name: str, imgsz: int, conf: float) -> Path:
    return cache_dir / f"preds_{name}_sz{imgsz}_conf{conf:g}.npz"


def _load_cache(path: Path, stems: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Return cached predictions only if the cache covers EVERY requested stem.

    A 4-frame smoke cache must never be silently reused by the 72-frame sweep, so coverage is
    checked as a superset, not just for file existence.
    """
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    cached = [str(s) for s in z["stems"]]
    if not set(stems).issubset(cached):
        return None
    counts = z["counts"].astype(int)
    boxes, scores = z["boxes"].reshape(-1, 4), z["scores"].reshape(-1)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    off = 0
    for stem, n in zip(cached, counts):
        out[stem] = (boxes[off:off + n].copy(), scores[off:off + n].copy())
        off += n
    return {s: out[s] for s in stems}


def _save_cache(path: Path, preds: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    stems = sorted(preds)
    counts = np.array([len(preds[s][0]) for s in stems], int)
    boxes = np.concatenate([preds[s][0] for s in stems]) if stems else np.zeros((0, 4))
    scores = np.concatenate([preds[s][1] for s in stems]) if stems else np.zeros(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, stems=np.array(stems), counts=counts,
             boxes=boxes.astype(np.float32), scores=scores.astype(np.float32))


def run_model(name: str, imgs: list[Path], conf: float, device: str,
              cache_dir: Path, refresh: bool) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    """Predict (or load cached) accept-all boxes for one checkpoint at its native imgsz."""
    weights, imgsz, _ = MODELS[name]
    stems = [p.stem for p in imgs]
    cpath = _cache_path(cache_dir, name, imgsz, conf)
    if not refresh:
        hit = _load_cache(cpath, stems)
        if hit is not None:
            print(f"  {name:<7} cache HIT  {cpath.name}")
            return hit, 0.0

    from ultralytics import YOLO

    repo = Path(__file__).resolve().parents[1]
    model = YOLO(str(repo / weights))
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    t0 = time.time()
    for p in imgs:
        res = model.predict(str(p), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            preds[p.stem] = (np.zeros((0, 4)), np.zeros(0))
            continue
        preds[p.stem] = (res.boxes.xyxy.cpu().numpy().astype(float),
                         res.boxes.conf.cpu().numpy().astype(float))
    dt = time.time() - t0
    print(f"  {name:<7} inferred {len(imgs)} frames @ imgsz {imgsz} in {dt:.1f}s "
          f"({dt / max(len(imgs), 1):.2f}s/frame)")
    _save_cache(cpath, preds)
    return preds, dt


def verify_coord_space(name: str, preds: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Prove boxes are in ORIGINAL 1280x1024 pixels, not letterboxed model-input pixels.

    Two checks: (a) nothing escapes the frame by more than a rounding tolerance, and (b) for a
    model whose imgsz is smaller than the frame width, count boxes with x2 > imgsz — such a box
    is impossible in model-input coordinates, so a non-zero count is positive evidence of
    rescaling. A zero count on such a model is inconclusive, not a failure.
    """
    imgsz = MODELS[name][1]
    all_b = [b for b, _ in preds.values() if len(b)]
    if not all_b:
        return {"n_boxes": 0, "max_x": 0.0, "max_y": 0.0, "beyond_imgsz_x": 0, "in_frame": True}
    B = np.concatenate(all_b)
    tol = 1.5
    return {
        "n_boxes": int(len(B)),
        "max_x": float(B[:, 2].max()),
        "max_y": float(B[:, 3].max()),
        "beyond_imgsz_x": int((B[:, 2] > imgsz + tol).sum()) if imgsz < IMG_W else -1,
        "in_frame": bool(B[:, 0].min() >= -tol and B[:, 1].min() >= -tol
                         and B[:, 2].max() <= IMG_W + tol and B[:, 3].max() <= IMG_H + tol),
    }


# --------------------------------------------------------------------------- fusion rules
def weighted_box_fusion(boxes_list: list[np.ndarray], scores_list: list[np.ndarray],
                        iou_thr: float, n_models: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted Box Fusion over per-model box sets. Returns (boxes, score_avg, score_agree).

    Boxes are consumed high-score-first; each is either absorbed into the existing cluster whose
    *current fused box* it overlaps most (IoU >= iou_thr) or seeds a new cluster. A cluster's
    box is the score-weighted coordinate average of its members, recomputed on each absorption —
    that is what distinguishes WBF from NMS, which discards the losers instead of averaging.

    Two fused scores are returned because they mean different things at a deployment threshold:
      score_avg   mean member score — comparable to a single model's raw confidence.
      score_agree mean * (distinct models in cluster / n_models) — the standard WBF weighting,
                  which demotes a box only one model proposed. With n_models=5 a solo box can
                  never exceed 0.2, so this is a genuinely different operating point, not a
                  rescaling of the same one.
    Distinct *models* (not member count) are used so one model firing twice cannot fake consensus.
    """
    bs, ss, ms = [], [], []
    for mi, (b, s) in enumerate(zip(boxes_list, scores_list)):
        b = np.asarray(b, float).reshape(-1, 4)
        if len(b) == 0:
            continue
        bs.append(b)
        ss.append(np.asarray(s, float).reshape(-1))
        ms.append(np.full(len(b), mi, int))
    if not bs:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    B, S, M = np.concatenate(bs), np.concatenate(ss), np.concatenate(ms)

    fused: list[list[float]] = []
    members: list[list[int]] = []
    for i in np.argsort(-S, kind="stable"):          # stable => deterministic on score ties
        j = -1
        if fused:
            ious = iou_mat(B[i:i + 1], np.asarray(fused, float))[0]
            k = int(np.argmax(ious))
            if ious[k] >= iou_thr:
                j = k
        if j < 0:
            fused.append(B[i].tolist())
            members.append([int(i)])
        else:
            members[j].append(int(i))
            idx = np.asarray(members[j], int)
            w = S[idx]
            fused[j] = ((B[idx] * w[:, None]).sum(0) / max(w.sum(), 1e-9)).tolist()

    out = np.asarray(fused, float).reshape(-1, 4)
    avg = np.array([S[np.asarray(m, int)].mean() for m in members], float)
    n_mod = np.array([len(set(M[np.asarray(m, int)].tolist())) for m in members], float)
    return out, avg, avg * np.minimum(n_mod, n_models) / n_models


def nms_union(boxes_list: list[np.ndarray], scores_list: list[np.ndarray],
              iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Pool every model's boxes, then greedy score-ordered NMS. The naive fusion baseline."""
    bs = [np.asarray(b, float).reshape(-1, 4) for b in boxes_list]
    ss = [np.asarray(s, float).reshape(-1) for s in scores_list]
    B = np.concatenate([b for b in bs if len(b)]) if any(len(b) for b in bs) else np.zeros((0, 4))
    S = np.concatenate([s for s, b in zip(ss, bs) if len(b)]) if len(B) else np.zeros(0)
    keep: list[int] = []
    for i in np.argsort(-S, kind="stable"):
        if keep and iou_mat(B[i:i + 1], B[keep])[0].max() > iou_thr:
            continue
        keep.append(int(i))
    return B[keep], S[keep]


def merge_union(boxes_list: list[np.ndarray], scores_list: list[np.ndarray],
                iou_thr: float, agglomerate: bool) -> tuple[np.ndarray, np.ndarray]:
    """Pool every model's boxes, then batterycv.merge.merge_boxes.

    Two variants, because they behave very differently on a multi-model pool: with
    `agglomerate=True` (the pseudo-label-time setting) any survivors overlapping at IoU >= 0.2
    are unioned, and a dense five-model pool chains those unions into blobs far larger than any
    battery. The containment+NMS-only variant is the honest fusion analogue.
    """
    bs = [np.asarray(b, float).reshape(-1, 4) for b in boxes_list]
    ss = [np.asarray(s, float).reshape(-1) for s in scores_list]
    if not any(len(b) for b in bs):
        return np.zeros((0, 4)), np.zeros(0)
    B = np.concatenate([b for b in bs if len(b)])
    S = np.concatenate([s for s, b in zip(ss, bs) if len(b)])
    return merge_boxes(B, S, iou_thresh=iou_thr, contain_thresh=0.75,
                       agglomerate=agglomerate, agg_iou=0.2)


# --------------------------------------------------------------------------- scoring helpers
def per_class_stats(acc: Accumulator) -> dict[str, dict | None]:
    """Per-class tallies read directly, WITHOUT touching Accumulator.stats via _agg/recall.

    `Accumulator.stats` is a defaultdict, so querying an unseen class through the public
    `recall()`/`precision()` would silently insert it and make it appear in later `table()`
    output. A class with zero GT reports recall None, not 0.0 — 0/0 is not a miss.
    """
    out: dict[str, dict | None] = {}
    for c in CLASSES:
        if c not in acc.stats:
            out[c] = None
            continue
        tp, fp, ngt, nbox = acc.stats[c]
        out[c] = {"recall": tp / ngt if ngt else None, "precision": tp / max(tp + fp, 1),
                  "tp": int(tp), "fp": int(fp), "n_gt": int(ngt), "n_box": int(nbox)}
    return out


def score_boxes(boxes_by_stem: dict[str, tuple[np.ndarray, np.ndarray]],
                gt: dict[str, np.ndarray], title: str) -> dict:
    """Run every frame through the shared Accumulator and return a JSON-able summary."""
    acc = Accumulator()
    for stem, g in gt.items():
        b, s = boxes_by_stem.get(stem, (np.zeros((0, 4)), np.zeros(0)))
        acc.add(stem, b, g, s)
    tp, fp, ngt, nbox = acc.agg(None)
    ap = acc.ap50()
    pcs = per_class_stats(acc)
    return {
        "recall": tp / max(ngt, 1),
        "precision": tp / max(tp + fp, 1),
        "boxes_per_frame": nbox / max(acc.n_frames, 1),
        "ap50": None if ap != ap else float(ap),
        "n_gt": int(ngt),
        "per_class": pcs,
        "per_class_recall": {c: (pcs[c]["recall"] if pcs[c] else None) for c in CLASSES},
        "table": acc.table(title),
    }


def row(label: str, r: dict) -> str:
    pc = r["per_class_recall"]
    cells = "".join(f"{pc[c]:>7.3f}" if pc[c] is not None else f"{'-':>7}" for c in CLASSES)
    ap = f"{r['ap50']:>6.3f}" if r["ap50"] is not None else f"{'-':>6}"
    return (f"{label:<34}{r['boxes_per_frame']:>7.1f}{r['recall']:>7.3f}"
            f"{r['precision']:>7.3f}{ap} |{cells}")


HEADER = (f"{'config':<34}{'box/f':>7}{'R':>7}{'P':>7}{'AP50':>6} |"
          + "".join(f"{c.replace('li_ion_', '').replace('ni_cd_', '').replace('_all', '')[:6]:>7}"
                   for c in CLASSES))


def union_cov(covs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """OR the per-frame coverage masks of several models together."""
    out: dict[str, np.ndarray] = {}
    keys = set().union(*[set(c) for c in covs]) if covs else set()
    for k in sorted(keys):
        masks = [c[k] for c in covs if k in c]
        u = masks[0].copy()
        for m in masks[1:]:
            u |= m
        out[k] = u
    return out


def cov_stats(cov: dict[str, np.ndarray]) -> dict:
    pc = per_class_coverage(cov)
    tot = sum(v[1] for v in pc.values())
    hit = sum(v[0] for v in pc.values())
    return {"recall": hit / max(tot, 1), "covered": hit, "total": tot,
            "per_class": {c: {"covered": pc[c][0], "total": pc[c][1],
                              "recall": pc[c][0] / max(pc[c][1], 1)}
                          for c in sorted(pc)}}


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma-separated subset of: " + ", ".join(MODELS))
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test on N frames, picked ROUND-ROBIN across classes (GT-bearing "
                         "frames first) so a 6-frame smoke touches all six classes (0 = all 72)")
    ap.add_argument("--conf", type=float, default=0.001, help="accept-all floor; threshold later in numpy")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/phase4/ensemble.json")
    ap.add_argument("--cache-dir", default="results/phase4/cache")
    ap.add_argument("--refresh", action="store_true", help="ignore cached predictions and re-infer")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    bad = [n for n in names if n not in MODELS]
    if bad:
        ap.error(f"unknown model(s) {bad}; choose from {list(MODELS)}")

    paths = load_paths()
    img_dir, lbl_dir = paths["eval_dir"] / "images", paths["eval_dir"] / "labels"
    imgs = sorted(img_dir.glob("*.jpg"))
    gt_all = load_gt(lbl_dir)
    if args.limit and args.limit < len(imgs):
        # round-robin over classes, GT-bearing frames first, so a tiny smoke run still
        # exercises the classes the whole investigation is about (ni_mh_all, li_ion_mobile)
        by_cls: dict[str, list[Path]] = {}
        for p in imgs:
            by_cls.setdefault(class_of(p.stem), []).append(p)
        for c in by_cls:
            by_cls[c].sort(key=lambda p: (0 if len(gt_all.get(p.stem, [])) else 1, p.stem))
        picked: list[Path] = []
        r = 0
        while len(picked) < args.limit and any(r < len(v) for v in by_cls.values()):
            for c in sorted(by_cls):
                if r < len(by_cls[c]) and len(picked) < args.limit:
                    picked.append(by_cls[c][r])
            r += 1
        imgs = sorted(picked)
    gt = {p.stem: gt_all.get(p.stem, np.zeros((0, 4))) for p in imgs}
    n_gt = sum(len(v) for v in gt.values())

    print(f"probe_ensemble: {len(imgs)} frames, {n_gt} GT boxes, models={names}")
    if args.limit:
        print("  SMOKE MODE — classes touched:",
              ", ".join(f"{c}({sum(len(gt[p.stem]) for p in imgs if class_of(p.stem) == c)} GT)"
                        for c in sorted({class_of(p.stem) for p in imgs})))
        print("  SMOKE MODE — numbers below are from a handful of frames; they exist to prove the")
        print("  code path, NOT to support any conclusion. Only the full 72-frame run counts.")
    missing = [n for n in names if not (repo / MODELS[n][0]).exists()]
    if missing:
        ap.error(f"missing weights for {missing}")

    # ---- 1. inference (once per model, cached) -------------------------------------------
    print("\n[1] inference")
    cache_dir = repo / args.cache_dir if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)
    preds: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    infer_secs = 0.0
    for n in names:
        preds[n], dt = run_model(n, imgs, args.conf, args.device, cache_dir, args.refresh)
        infer_secs += dt

    print("\n[1b] coordinate-space verification (boxes must be in ORIGINAL 1280x1024 pixels)")
    scale_check = {}
    for n in names:
        v = verify_coord_space(n, preds[n])
        scale_check[n] = v
        ev = (f"no box exceeds y={IMG_H} though letterboxing to {MODELS[n][1]} would allow it"
              if v["beyond_imgsz_x"] < 0
              else f"{v['beyond_imgsz_x']} boxes with x2>imgsz {MODELS[n][1]}")
        print(f"  {n:<7} imgsz {MODELS[n][1]:<5} n={v['n_boxes']:<6} max_x={v['max_x']:7.1f} "
              f"max_y={v['max_y']:7.1f} in_frame={v['in_frame']}  proof: {ev}")
        if not v["in_frame"]:
            raise SystemExit(f"ABORT: {n} produced boxes outside the 1280x1024 frame — "
                             "Ultralytics did NOT rescale to original coords; fix before trusting anything.")

    # ---- 2. per-model baselines ------------------------------------------------------------
    print("\n[2] single models (accept-all = recall ceiling; conf>=0.25 = deployment)")
    print(HEADER)
    single: dict[str, dict] = {}
    for n in names:
        allb = preds[n]
        d25 = {s: (b[sc >= DEPLOY_CONF], sc[sc >= DEPLOY_CONF]) for s, (b, sc) in allb.items()}
        single[n] = {"weights": MODELS[n][0], "imgsz": MODELS[n][1], "provenance": MODELS[n][2],
                     "accept_all": score_boxes(allb, gt, f"{n} accept-all"),
                     "conf25": score_boxes(d25, gt, f"{n} conf>=0.25")}
        print(row(f"{n} accept-all", single[n]["accept_all"]))
        print(row(f"{n} conf>=0.25", single[n]["conf25"]))

    # ---- 3. THE ORACLE UNION CEILING -------------------------------------------------------
    print("\n" + "=" * 96)
    print("[3] ORACLE UNION CEILING — upper bound on ANY fusion rule over these checkpoints")
    print("=" * 96)
    ceilings: dict[str, dict] = {}
    covs_by_budget: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for bname, bmode in BUDGETS.items():
        bcovs = {n: coverage(budget_boxes(preds[n], bmode), gt) for n in names}
        covs_by_budget[bname] = bcovs
        bcs = {n: cov_stats(bcovs[n]) for n in names}
        us = cov_stats(union_cov([bcovs[n] for n in names]))
        shared = union_recall(*[bcovs[n] for n in names])   # cross-check vs the shared helper
        bpf = {n: sum(len(v) for v in budget_boxes(preds[n], bmode).values()) / max(len(gt), 1)
               for n in names}
        b1_name = max(names, key=lambda n: bcs[n]["recall"])
        ceilings[bname] = {"union": us, "per_model": bcs, "boxes_per_frame": bpf,
                           "best_single": b1_name, "best_single_recall": bcs[b1_name]["recall"],
                           "headroom": us["recall"] - bcs[b1_name]["recall"],
                           "cross_check_union_recall": shared}

        tag = {"accept_all": "every box (conf>=%g) — NOT budget-matched" % args.conf,
               "top20": "top-20 boxes/model/frame — budget-matched",
               "conf25": "conf>=0.25 only — deployment budget"}[bname]
        print(f"\n  budget '{bname}': {tag}")
        print(f"  {'class':<16}{'union':>8}{'best-1':>8}{'gain':>8}{'nGT':>6}   per-model coverage")
        for c in CLASSES:
            if us["per_class"].get(c, {}).get("total", 0) == 0:
                continue   # class present only via an empty (bare-belt) frame — 0/0 is not a miss
            ur = us["per_class"][c]["recall"]
            per = {n: bcs[n]["per_class"].get(c, {}).get("recall", 0.0) for n in names}
            print(f"  {c:<16}{ur:>8.3f}{max(per.values()):>8.3f}{ur - max(per.values()):>+8.3f}"
                  f"{us['per_class'][c]['total']:>6}   " + " ".join(f"{n}={per[n]:.2f}" for n in names))
        print(f"  {'TOTAL':<16}{us['recall']:>8.3f}{bcs[b1_name]['recall']:>8.3f}"
              f"{us['recall'] - bcs[b1_name]['recall']:>+8.3f}{us['total']:>6}   best single = {b1_name}"
              f" | box/f " + " ".join(f"{n}={bpf[n]:.0f}" for n in names))
        if abs(shared - us["recall"]) > 1e-9:
            print(f"  WARNING: evalutil.union_recall={shared:.4f} disagrees with this table — investigate")

    covs = covs_by_budget["accept_all"]
    cstats = ceilings["accept_all"]["per_model"]
    u_stats = ceilings["accept_all"]["union"]
    u_shared = ceilings["accept_all"]["cross_check_union_recall"]
    best_single = max(names, key=lambda n: single[n]["accept_all"]["recall"])
    best_r = single[best_single]["accept_all"]["recall"]
    print(f"\n  cross-check evalutil.union_recall = {u_shared:.4f} (agrees with the accept_all table)")
    print(f"  HEADROOM for all fusion work on these five checkpoints (accept_all): "
          f"{ceilings['accept_all']['headroom']:+.3f} recall")
    print(f"  HEADROOM at a matched 20-box budget:                                "
          f"{ceilings['top20']['headroom']:+.3f} recall")
    print("  If the matched-budget headroom is ~0 while accept_all looks generous, the apparent")
    print("  complementarity is one model spraying boxes, not two models seeing different things.")

    # ---- 4. pairwise complementarity -------------------------------------------------------
    print("\n[4] pairwise complementarity (accept-all coverage masks, IoU 0.5). 'A only' = GT")
    print("    objects A covers and B misses. u@20 repeats the union at a matched 20-box budget,")
    print("    so a pair that only looks complementary because one model sprays boxes shows up.")
    print(f"{'A / B':<18}{'A only':>8}{'B only':>8}{'both':>8}{'neither':>9}{'union':>8}{'u@20':>8}"
          f"{'   ni_mh A|B|both':>20}{'   mobile A|B|both':>20}")
    pairs = []
    for a, b in itertools.combinations(names, 2):
        cnt = {"a_only": 0, "b_only": 0, "both": 0, "neither": 0}
        pc: dict[str, dict[str, int]] = {}
        for stem, g in gt.items():
            if len(g) == 0:
                continue
            ma, mb = covs[a][stem], covs[b][stem]
            c = class_of(stem)
            d = pc.setdefault(c, {"a_only": 0, "b_only": 0, "both": 0, "neither": 0})
            for key, m in (("a_only", ma & ~mb), ("b_only", mb & ~ma),
                           ("both", ma & mb), ("neither", ~ma & ~mb)):
                cnt[key] += int(m.sum())
                d[key] += int(m.sum())
        un = union_recall(covs[a], covs[b])
        un20 = union_recall(covs_by_budget["top20"][a], covs_by_budget["top20"][b])
        pairs.append({"a": a, "b": b, **cnt, "union_recall": un,
                      "union_recall_top20": un20, "per_class": pc})
        nm = pc.get("ni_mh_all", {})
        mo = pc.get("li_ion_mobile", {})

        def cell(x: dict) -> str:
            return f"{x.get('a_only', 0)}|{x.get('b_only', 0)}|{x.get('both', 0)}"

        print(f"{a + ' / ' + b:<18}{cnt['a_only']:>8}{cnt['b_only']:>8}{cnt['both']:>8}"
              f"{cnt['neither']:>9}{un:>8.3f}{un20:>8.3f}{cell(nm):>20}{cell(mo):>20}")

    # ---- 5. pick the most decorrelated triple (max union recall over all C(n,3)) -----------
    subsets: dict[str, list[str]] = {"all%d" % len(names): list(names)}
    triple_search = []
    if len(names) >= 4:
        for tri in itertools.combinations(names, 3):
            triple_search.append({"models": list(tri),
                                  "union_recall": union_recall(*[covs[n] for n in tri])})
        triple_search.sort(key=lambda d: -d["union_recall"])
        best_tri = triple_search[0]["models"]
        subsets["top3"] = best_tri
        print(f"\n[5] most-decorrelated triple by union recall: {best_tri} "
              f"({triple_search[0]['union_recall']:.3f}); worst triple "
              f"{triple_search[-1]['models']} ({triple_search[-1]['union_recall']:.3f})")
        print("    NOTE: this triple is selected ON the eval set — its edge is optimistic.")

    # ---- 6. fusion sweep -------------------------------------------------------------------
    print("\n[6] fusion sweep. slices: accept-all | post25 = fuse then threshold fused score |")
    print("    pre25 = threshold each model at 0.25 THEN fuse (the deployment-realistic order)")
    print(HEADER)
    fusion: list[dict] = []
    t0 = time.time()
    for sub_name, sub in subsets.items():
        for pre in (False, True):
            src = {}
            for n in sub:
                src[n] = {s: ((b[sc >= DEPLOY_CONF], sc[sc >= DEPLOY_CONF]) if pre else (b, sc))
                          for s, (b, sc) in preds[n].items()}
            for iou in IOU_SWEEP:
                out: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
                    "wbf_agree": {}, "wbf_avg": {}, "nms": {}, "merge_agg": {}, "merge_noagg": {}}
                for stem in gt:
                    bl = [src[n][stem][0] for n in sub]
                    sl = [src[n][stem][1] for n in sub]
                    wb, w_avg, w_agr = weighted_box_fusion(bl, sl, iou, len(sub))
                    out["wbf_agree"][stem] = (wb, w_agr)
                    out["wbf_avg"][stem] = (wb, w_avg)
                    out["nms"][stem] = nms_union(bl, sl, iou)
                    out["merge_agg"][stem] = merge_union(bl, sl, iou, agglomerate=True)
                    out["merge_noagg"][stem] = merge_union(bl, sl, iou, agglomerate=False)
                for rule, per_stem in out.items():
                    if pre:
                        slices = {"pre25": per_stem}
                    else:
                        slices = {
                            "accept_all": per_stem,
                            "post25": {s: (b[sc >= DEPLOY_CONF], sc[sc >= DEPLOY_CONF])
                                       for s, (b, sc) in per_stem.items()},
                        }
                    for sl_name, boxes in slices.items():
                        label = f"{sub_name} {rule} iou{iou:g} {sl_name}"
                        r = score_boxes(boxes, gt, label)
                        fusion.append({"subset": sub_name, "models": sub, "rule": rule,
                                       "iou": iou, "slice": sl_name, **r})
                        print(row(label, r))
    print(f"    (fusion sweep {time.time() - t0:.1f}s over cached predictions, 0 extra forward passes)")

    # ---- 7. verdict ------------------------------------------------------------------------
    def _fmt(d: dict) -> str:
        return (f"{d['recall']:.3f}  ({d['subset']} {d['rule']} iou{d['iou']:g} {d['slice']}, "
                f"prec {d['precision']:.3f}, {d['boxes_per_frame']:.1f} box/f, "
                f"AP50 {d['ap50']:.3f})" if d else "n/a")

    best_ceil = max((f for f in fusion if f["slice"] == "accept_all"),
                    key=lambda d: d["recall"], default=None)
    deploy = [f for f in fusion if f["slice"] in ("post25", "pre25")]
    best_dep_r = max(deploy, key=lambda d: d["recall"], default=None)
    best_dep_ap = max(deploy, key=lambda d: (d["ap50"] or 0.0), default=None)
    dep_base = single[best_single]["conf25"]

    print("\n" + "=" * 96)
    print("[7] VERDICT")
    print(f"  best single model, accept-all recall : {best_r:.3f} ({best_single})")
    print(f"  ORACLE UNION CEILING                 : {u_stats['recall']:.3f} "
          f"({u_stats['recall'] - best_r:+.3f} headroom — nothing below can exceed this)")
    print(f"  best fused, accept-all               : {_fmt(best_ceil)}")
    print(f"  best fused, deployment slice         : {_fmt(best_dep_r)}")
    print(f"  best fused by AP50, deployment slice : {_fmt(best_dep_ap)}")
    print(f"  same-model deployment reference      : {dep_base['recall']:.3f}  "
          f"({best_single} conf>=0.25, prec {dep_base['precision']:.3f}, "
          f"{dep_base['boxes_per_frame']:.1f} box/f, AP50 {dep_base['ap50']:.3f})")
    print("  Read the ceiling first. If it sits on top of the best single model, the five")
    print("  checkpoints miss the SAME objects, no fusion rule can rescue recall, and Track B")
    print("  is a settled negative rather than a tuning problem. A fused recall that beats a")
    print("  single model only by emitting far more boxes per frame is not a win either —")
    print("  check the box/f and precision columns before claiming one.")
    print("=" * 96)

    out_path = repo / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"frames": [p.stem for p in imgs], "n_frames": len(imgs), "n_gt": int(n_gt),
                 "models": names, "conf_floor": args.conf, "deploy_conf": DEPLOY_CONF,
                 "device": args.device, "limit": args.limit, "iou_sweep": list(IOU_SWEEP),
                 "inference_seconds": infer_secs, "smoke": bool(args.limit)},
        "coord_space_check": scale_check,
        "single_models": single,
        "union_ceiling": {"by_budget": ceilings, "headline_budget": "accept_all",
                          "total": u_stats, "cross_check_union_recall": u_shared,
                          "best_single": best_single, "best_single_recall": best_r,
                          "headroom": u_stats["recall"] - best_r},
        "per_model_coverage": cstats,
        "complementarity": pairs,
        "subset_union_search": triple_search,
        "subsets": subsets,
        "fusion": fusion,
        "verdict": {"best_fused_accept_all": best_ceil, "best_fused_deploy_recall": best_dep_r,
                    "best_fused_deploy_ap50": best_dep_ap,
                    "deploy_reference": {"model": best_single, **dep_base}},
    }
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
