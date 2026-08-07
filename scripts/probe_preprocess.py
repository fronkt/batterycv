"""Sweep the illumination-preprocessing knobs that have been frozen since day one.

WHY: ``batterycv/preprocess.normalize_illumination`` has run at ``clip=2.5, grid=8`` through
every experiment in this project — labeler swaps, resolution sweeps, backbone sweeps,
fine-tuning — while imagery quality is the *documented* recall bottleneck
(``docs/recall_ceiling_findings.md``). It is the one knob nobody ever turned.

The eval JPEGs are ALREADY CLAHE(2.5,8) copies of raw frames, so they cannot be used to test
alternative preprocessing. This probe maps each eval label stem back to its lossless RAW BMP
(stem is ``<class>__<raw stem>``; the raw path comes from ``batterycv.io.build_manifest``),
applies a candidate transform itself, and scores the result with the shared matcher in
``batterycv/evalutil.py`` so the numbers are comparable to the published tables.

THE CONFOUND, stated up front and repeated in every output: ``battery_ft1`` was TRAINED on
CLAHE(2.5,8). Any other preprocessing is a train/test distribution shift, so a recall DROP is
ambiguous — "worse imagery" and "merely different from what the weights saw" look identical
from the recall column alone. Two partial mitigations are built in:

  (a) a SECOND detector (``battery_yolo11_yw``: same CLAHE(2.5,8) training imagery, different
      pseudo-labels). A genuine imagery effect should move both detectors the same way; an
      effect visible in only one is that model's quirk.
  (b) a DETECTOR-FREE contrast diagnostic computed straight from the pixels: how far the
      GT-box interior sits from the surrounding belt (``sep_dprime``, pooled-sigma units —
      invariant to any affine change in intensity), plus Weber contrast and local RMS
      contrast. A variant that measurably RAISES object/belt separation while LOWERING recall
      is evidence of train/test mismatch rather than worse imagery. That distinction is the
      entire point of this probe.

All variants operate on the LAB L channel, exactly like ``normalize_illumination``, so chroma
is never touched and luminance processing is the only thing that varies. The
``clahe_c2.5_g8`` variant is asserted byte-identical to ``normalize_illumination`` output and
doubles as the SANITY GATE: its score must land near the documented accept-all TOTAL recall
0.489 / conf>=0.25 recall 0.452. A large gap means the raw->eval mapping is wrong and every
number downstream is invalid. (It will not match to 3 decimals: eval/images went through JPEG
compression, the raw BMPs are lossless.)

    python scripts/probe_preprocess.py --limit 4 --variants clahe_c2.5_g8,clahe_c8.0_g8,msr
    python scripts/probe_preprocess.py            # full 72-frame sweep, all variants
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batterycv.config import load_paths  # noqa: E402
from batterycv.evalutil import CLASSES, Accumulator, class_of, load_gt  # noqa: E402
from batterycv.io import build_manifest  # noqa: E402
from batterycv.preprocess import normalize_illumination, read_bgr  # noqa: E402

LTransform = Callable[[np.ndarray], np.ndarray]

BASELINE = "clahe_c2.5_g8"

# Weights already on disk; both native imgsz 1024, both trained on CLAHE(2.5,8) imagery.
DET_A = ("battery_ft1", ROOT / "runs/detect/battery_ft1/weights/best.pt")
DET_B = ("battery_yolo11_yw", ROOT / "runs/detect/battery_yolo11_yw/weights/best.pt")

# Detector B is the cross-detector consistency check, not a second full sweep — running it on
# every variant would roughly double a ~1 h CPU job for little extra information. This subset
# spans the families (identity / clip axis / grid axis / gamma / global EQ / retinex).
DET_B_DEFAULT = [
    "none", BASELINE, "clahe_c1.0_g8", "clahe_c8.0_g8", "clahe_c2.5_g16",
    "gamma0.7_clahe", "histeq", "msr",
]

# Documented battery_ft1 @1024 baseline, measured with THIS harness on eval/images.
REFERENCE = {
    "accept_all": {
        "recall": 0.489, "precision": 0.079, "box_per_frame": 16.1, "ap50": 0.239,
        "per_class_recall": {"li_ion_laptop": 0.733, "li_ion_mobile": 0.449, "liso2": 0.478,
                             "ni_cd_bulk": 0.875, "ni_cd_small": 0.435, "ni_mh_all": 0.120},
    },
    "conf25": {
        "recall": 0.452, "precision": 0.410, "box_per_frame": 2.8, "ap50": 0.233,
        "per_class_recall": {"li_ion_laptop": 0.733, "li_ion_mobile": 0.406, "liso2": 0.435,
                             "ni_cd_bulk": 0.750, "ni_cd_small": 0.435, "ni_mh_all": 0.080},
    },
}

CONFOUND_NOTE = (
    "Both detectors were TRAINED on CLAHE(2.5,8) imagery. Every non-baseline variant is a "
    "train/test distribution shift, so a recall DROP does not by itself mean worse imagery. "
    "Read the recall columns together with sep_dprime: separation UP + recall DOWN => "
    "train/test mismatch; separation DOWN + recall DOWN => genuinely worse imagery."
)


# --------------------------------------------------------------------------- L transforms
def _clahe(l: np.ndarray, clip: float, grid: int) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)


def _gamma(l: np.ndarray, g: float) -> np.ndarray:
    """out = 255*(in/255)**g. g < 1 brightens (these frames sit at mean L ~71/255)."""
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(l, lut)


def _gamma_then_clahe(l: np.ndarray, g: float) -> np.ndarray:
    return _clahe(_gamma(l, g), 2.5, 8)


def _histeq(l: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(l)


def _msr(l: np.ndarray, sigmas: tuple[float, ...] = (15.0, 80.0, 250.0)) -> np.ndarray:
    """Multi-scale Retinex on L, rescaled by a 1/99-percentile stretch (deterministic)."""
    f = l.astype(np.float32) + 1.0
    logf = np.log(f)
    acc = np.zeros_like(logf)
    for s in sigmas:
        acc += logf - np.log(cv2.GaussianBlur(f, (0, 0), s) + 1e-6)
    acc /= len(sigmas)
    lo, hi = np.percentile(acc, (1.0, 99.0))
    return np.clip((acc - lo) / max(float(hi - lo), 1e-6) * 255.0, 0, 255).astype(np.uint8)


def _clahe_unsharp(l: np.ndarray, sigma: float = 3.0, amount: float = 1.0) -> np.ndarray:
    c = _clahe(l, 2.5, 8).astype(np.float32)
    blur = cv2.GaussianBlur(c, (0, 0), sigma)
    return np.clip(c + amount * (c - blur), 0, 255).astype(np.uint8)


def _bilateral_clahe(l: np.ndarray) -> np.ndarray:
    return _clahe(cv2.bilateralFilter(l, 7, 25, 7), 2.5, 8)


def _nlm_clahe(l: np.ndarray) -> np.ndarray:
    return _clahe(cv2.fastNlMeansDenoising(l, None, 7, 7, 21), 2.5, 8)


def build_variants() -> dict[str, LTransform]:
    """name -> (uint8 L) -> (uint8 L). Order here is the report order."""
    v: dict[str, LTransform] = {"none": lambda l: l}
    for clip in (1.0, 2.5, 4.0, 8.0):
        for grid in (4, 8, 16):
            v[f"clahe_c{clip}_g{grid}"] = partial(_clahe, clip=clip, grid=grid)
    for g in (0.5, 0.7):
        v[f"gamma{g}"] = partial(_gamma, g=g)
        v[f"gamma{g}_clahe"] = partial(_gamma_then_clahe, g=g)
    v["histeq"] = _histeq
    v["msr"] = _msr
    v["clahe_unsharp"] = _clahe_unsharp
    v["bilateral_clahe"] = _bilateral_clahe
    v["nlm_clahe"] = _nlm_clahe
    return v


def apply_variant(bgr: np.ndarray, fn: LTransform) -> np.ndarray:
    """LAB split -> transform L -> merge -> BGR. Same code path as normalize_illumination."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((fn(l), a, b)), cv2.COLOR_LAB2BGR)


# ------------------------------------------------------------------- detector-free contrast
def contrast_stats(bgr: np.ndarray, gt_boxes: np.ndarray) -> list[dict[str, float]]:
    """Per-GT-box object/belt separation, computed from pixels alone (no detector involved).

    object = GT box shrunk 10% per side (drops the half-belt boundary ring);
    belt   = a dilated window around the box MINUS every GT box on the frame, so a neighbouring
             battery can never be mistaken for belt.

    ``sep_dprime`` is the headline number: |mu_obj - mu_belt| / pooled sigma, which is
    invariant to any affine intensity change and therefore comparable across variants that
    change overall brightness. ``weber`` is gain-invariant but not gamma-invariant, and
    ``rms_local`` (std/mean of the window) is neither — read those two as secondary.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    occ = np.zeros((h, w), bool)
    ints = []
    for x0, y0, x1, y1 in np.asarray(gt_boxes, float).reshape(-1, 4):
        bx = (max(0, int(round(x0))), max(0, int(round(y0))),
              min(w, int(round(x1))), min(h, int(round(y1))))
        if bx[2] > bx[0] and bx[3] > bx[1]:
            occ[bx[1]:bx[3], bx[0]:bx[2]] = True
            ints.append(bx)

    out: list[dict[str, float]] = []
    for x0, y0, x1, y1 in ints:
        bw, bh = x1 - x0, y1 - y0
        sx, sy = int(round(0.1 * bw)), int(round(0.1 * bh))
        obj = gray[y0 + sy:y1 - sy, x0 + sx:x1 - sx]
        if obj.size < 16:
            obj = gray[y0:y1, x0:x1]
        m = max(8, int(round(0.35 * min(bw, bh))))
        rx0, ry0 = max(0, x0 - m), max(0, y0 - m)
        rx1, ry1 = min(w, x1 + m), min(h, y1 + m)
        win = gray[ry0:ry1, rx0:rx1]
        belt = win[~occ[ry0:ry1, rx0:rx1]]
        if obj.size < 16 or belt.size < 50:
            continue  # degenerate box or fully enclosed by neighbours; not scorable
        mo, mb = float(obj.mean()), float(belt.mean())
        pooled = float(np.sqrt(0.5 * (obj.var() + belt.var())) + 1e-6)
        out.append({
            "sep_dprime": abs(mo - mb) / pooled,
            "weber": abs(mo - mb) / (mb + 1e-6),
            "rms_local": float(win.std()) / (float(win.mean()) + 1e-6),
            "obj_mean": mo,
            "belt_mean": mb,
        })
    return out


def agg_contrast(per_box: dict[str, list[dict[str, float]]]) -> dict:
    """{stem: [box dicts]} -> {'total': {...}, 'per_class': {...}, 'n_boxes': int}."""
    keys = ["sep_dprime", "weber", "rms_local", "obj_mean", "belt_mean"]
    by_cls: dict[str, list[dict[str, float]]] = defaultdict(list)
    allb: list[dict[str, float]] = []
    for stem, boxes in per_box.items():
        by_cls[class_of(stem)].extend(boxes)
        allb.extend(boxes)

    def mean_of(rows):
        return {k: (float(np.mean([r[k] for r in rows])) if rows else float("nan")) for k in keys}

    return {
        "total": mean_of(allb),
        "n_boxes": len(allb),
        "per_class": {c: dict(mean_of(by_cls[c]), n_boxes=len(by_cls[c]))
                      for c in CLASSES if by_cls.get(c)},
    }


# ------------------------------------------------------------------------------- scoring
def summarize(acc: Accumulator) -> dict:
    """Accumulator -> plain dict. ``Accumulator.agg`` is now read-safe (it no longer inserts a
    phantom row for an unseen class), but the membership check is kept so absent classes stay
    absent from the JSON rather than appearing as zeros."""
    tp, fp, ngt, nbox = acc.agg(None)
    per_class = {}
    for c in CLASSES:
        if c not in acc.stats:
            continue
        ctp, cfp, cngt, cnb = acc.agg(c)
        # per-class box/frame divides by THAT class's frame count; dividing by the total
        # understates every per-class row by the number of classes (6x on this eval set)
        per_class[c] = {"recall": ctp / max(cngt, 1), "precision": ctp / max(ctp + cfp, 1),
                        "n_gt": int(cngt),
                        "box_per_frame": cnb / max(acc.frames_by_cls.get(c, 0), 1)}
    ap = acc.ap50()
    return {
        "recall": tp / max(ngt, 1),
        "precision": tp / max(tp + fp, 1),
        "n_gt": int(ngt),
        "n_box": int(nbox),
        "box_per_frame": nbox / max(acc.n_frames, 1),
        "ap50": None if ap != ap else float(ap),
        "per_class_recall": {c: v["recall"] for c, v in per_class.items()},
        "per_class": per_class,
    }


def select_stems(gt: dict[str, np.ndarray], limit: int) -> list[str]:
    """Deterministic, class-stratified subset so a small --limit still spans classes.

    Within a class, GT-bearing frames come before the 4 bare-belt frames — otherwise a
    2-frame smoke run can draw an empty frame and report recall over zero objects.
    """
    stems = sorted(gt)
    if not limit or limit >= len(stems):
        return stems
    pools = {c: sorted([s for s in stems if class_of(s) == c],
                       key=lambda s: (len(gt[s]) == 0, s)) for c in CLASSES}
    order = [c for c in CLASSES if pools[c]]
    picked: list[str] = []
    i = 0
    while len(picked) < limit and any(pools[c] for c in order):
        c = order[i % len(order)]
        if pools[c]:
            picked.append(pools[c].pop(0))
        i += 1
    return sorted(picked)


def raw_path_map(raw_dir: Path, stems: list[str]) -> dict[str, Path]:
    """Eval stem ``<class>__<raw stem>`` -> raw BMP path (frame stems are globally unique)."""
    df = build_manifest(raw_dir)
    if df.empty:
        sys.exit(f"empty manifest under {raw_dir}")
    by_stem = {Path(p).stem: Path(p) for p in df["path"]}
    out, missing = {}, []
    for s in stems:
        raw_stem = s.split("__", 1)[-1]
        if raw_stem in by_stem:
            out[s] = by_stem[raw_stem]
        else:
            missing.append(s)
    if missing:
        sys.exit(f"{len(missing)} eval stems have no raw frame, e.g. {missing[:3]} — "
                 "the raw->eval mapping is broken, refusing to produce numbers")
    return out


def predict_boxes(model, images: list[np.ndarray], imgsz: int, conf: float, batch: int):
    """-> list of (boxes_xyxy (N,4), scores (N,)) aligned with ``images``."""
    out = []
    for i in range(0, len(images), batch):
        for r in model.predict(images[i:i + batch], conf=conf, imgsz=imgsz,
                               device="cpu", verbose=False):
            b = r.boxes
            out.append((b.xyxy.cpu().numpy().reshape(-1, 4),
                        b.conf.cpu().numpy().reshape(-1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="frames to score (0 = all 72)")
    ap.add_argument("--variants", default="all", help="comma list of variant names, or 'all'")
    ap.add_argument("--det-b-variants", default="default",
                    help="'default' (8-variant consistency subset) | 'all' | 'none' | comma list")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf-lo", type=float, default=0.001, help="accept-all floor")
    ap.add_argument("--conf-hi", type=float, default=0.25, help="deployment threshold")
    ap.add_argument("--skip-jpeg-gate", action="store_true",
                    help="skip the eval-JPEG cross-check (saves ~1 predict/frame)")
    ap.add_argument("--out", default=str(ROOT / "results/phase4/preprocess.json"))
    args = ap.parse_args()

    all_variants = build_variants()
    names = list(all_variants) if args.variants == "all" else \
        [n.strip() for n in args.variants.split(",") if n.strip()]
    unknown = [n for n in names if n not in all_variants]
    if unknown:
        sys.exit(f"unknown variant(s) {unknown}; available: {list(all_variants)}")
    if BASELINE not in names:
        print(f"[warn] {BASELINE} not in --variants: the sanity gate will not run")

    if args.det_b_variants == "none":
        det_b_names: list[str] = []
    elif args.det_b_variants == "all":
        det_b_names = list(names)
    elif args.det_b_variants == "default":
        det_b_names = [n for n in DET_B_DEFAULT if n in names]
    else:
        det_b_names = [n.strip() for n in args.det_b_variants.split(",")
                       if n.strip() and n.strip() in names]

    paths = load_paths()
    gt_all = load_gt(paths["eval_dir"] / "labels")
    stems = select_stems(gt_all, args.limit)
    gt = {s: gt_all[s] for s in stems}
    raws = raw_path_map(paths["raw_dir"], stems)
    n_gt = int(sum(len(v) for v in gt.values()))
    print(f"probe_preprocess: {len(stems)} frames / {n_gt} GT boxes, "
          f"{len(names)} variants, imgsz={args.imgsz}")
    print(f"  detector A {DET_A[0]}: all {len(names)} variants")
    print(f"  detector B {DET_B[0]}: {len(det_b_names)} variants {det_b_names}")

    from ultralytics import YOLO

    model_a = YOLO(str(DET_A[1]))
    dets = [(DET_A[0], model_a, names)]
    if det_b_names:
        dets.append((DET_B[0], YOLO(str(DET_B[1])), det_b_names))

    # acc[det][variant][slice] ; slices are the two DISTINCT metrics, never merged
    acc = {d: {v: {"accept_all": Accumulator(), "conf25": Accumulator()} for v in vn}
           for d, _, vn in dets}
    diag: dict[str, dict[str, list[dict[str, float]]]] = {v: {} for v in names}
    baseline_identical: bool | None = None

    # The real gate: score detector A on the published eval JPEG for the SAME frames. Sample
    # size cancels exactly, so any gap is attributable to raw-BMP-vs-JPEG, not to the subset.
    jpeg_gate = BASELINE in names and not args.skip_jpeg_gate
    jpeg_acc = {"accept_all": Accumulator(), "conf25": Accumulator()}
    jpeg_mae: list[float] = []

    t0 = time.time()
    for k, stem in enumerate(stems):
        bgr = read_bgr(str(raws[stem]))
        if bgr.shape[:2] != (1024, 1280):
            sys.exit(f"{raws[stem]} is {bgr.shape[:2]}, expected (1024,1280) — GT pixel "
                     "coords would not align")
        imgs = {v: apply_variant(bgr, all_variants[v]) for v in names}

        if BASELINE in imgs and baseline_identical is None:
            baseline_identical = bool(np.array_equal(imgs[BASELINE],
                                                     normalize_illumination(bgr)))
            if not baseline_identical:
                sys.exit("clahe_c2.5_g8 does not reproduce normalize_illumination — the "
                         "sanity gate is meaningless, refusing to continue")

        gtb = gt[stem]
        for v in names:
            diag[v][stem] = contrast_stats(imgs[v], gtb)

        if jpeg_gate:
            jp = paths["eval_dir"] / "images" / f"{stem}.jpg"
            ej = read_bgr(str(jp))
            jpeg_mae.append(float(np.abs(ej.astype(np.int16)
                                         - imgs[BASELINE].astype(np.int16)).mean()))
            (jb, js), = predict_boxes(model_a, [ej], args.imgsz, args.conf_lo, 1)
            jpeg_acc["accept_all"].add(stem, jb, gtb, js)
            jkeep = js >= args.conf_hi
            jpeg_acc["conf25"].add(stem, jb[jkeep], gtb, js[jkeep])

        for dname, model, vn in dets:
            preds = predict_boxes(model, [imgs[v] for v in vn],
                                  args.imgsz, args.conf_lo, args.batch)
            for v, (boxes, scores) in zip(vn, preds):
                acc[dname][v]["accept_all"].add(stem, boxes, gtb, scores)
                keep = scores >= args.conf_hi
                acc[dname][v]["conf25"].add(stem, boxes[keep], gtb, scores[keep])

        el = time.time() - t0
        print(f"  [{k+1}/{len(stems)}] {stem[:52]:<52} {el:6.1f}s "
              f"({el/(k+1):.1f}s/frame)")

    # ------------------------------------------------------------------ report + persist
    results = {d: {v: {sl: summarize(a[sl]) for sl in ("accept_all", "conf25")}
                   for v, a in vs.items()} for d, vs in acc.items()}
    diagnostics = {v: agg_contrast(diag[v]) for v in names}

    gate = None
    if BASELINE in names:
        g = results[DET_A[0]][BASELINE]
        jr = {sl: summarize(jpeg_acc[sl]) for sl in ("accept_all", "conf25")} if jpeg_gate else None
        gate = {
            "variant": BASELINE, "detector": DET_A[0],
            "identical_to_normalize_illumination": baseline_identical,
            "raw_bmp": g,
            "eval_jpeg_same_frames": jr,
            "jpeg_vs_raw_pixel_mae": (float(np.mean(jpeg_mae)) if jpeg_mae else None),
            "delta_vs_eval_jpeg_accept_all": (
                g["accept_all"]["recall"] - jr["accept_all"]["recall"]) if jr else None,
            "delta_vs_eval_jpeg_conf25": (
                g["conf25"]["recall"] - jr["conf25"]["recall"]) if jr else None,
            "documented_reference_full_72_frames": REFERENCE,
            "delta_vs_reference_accept_all":
                g["accept_all"]["recall"] - REFERENCE["accept_all"]["recall"],
            "delta_vs_reference_conf25": g["conf25"]["recall"] - REFERENCE["conf25"]["recall"],
            "note": ("Two gates. STRONG: raw-BMP-CLAHE vs the eval JPEG scored on the SAME "
                     "frames — sample size cancels, so a gap means the raw->eval mapping or "
                     "the CLAHE reproduction is wrong; pixel MAE should be JPEG-artifact "
                     "sized (single-digit gray levels). WEAK: comparison to the documented "
                     "full-72-frame reference, which is only meaningful at --limit 0."),
        }
        print(acc[DET_A[0]][BASELINE]["accept_all"].table(
            f"SANITY GATE A  {DET_A[0]} / {BASELINE} / RAW BMP / accept-all "
            f"(full-72 ref {REFERENCE['accept_all']['recall']:.3f})"))
        if jpeg_gate:
            print(jpeg_acc["accept_all"].table(
                f"SANITY GATE B  {DET_A[0]} / EVAL JPEG, same frames / accept-all "
                f"(full-72 ref {REFERENCE['accept_all']['recall']:.3f})"))
            print(f"\n  raw-BMP-CLAHE vs eval-JPEG: accept-all recall "
                  f"{g['accept_all']['recall']:.3f} vs {jr['accept_all']['recall']:.3f} "
                  f"(delta {gate['delta_vs_eval_jpeg_accept_all']:+.3f}) | conf>={args.conf_hi} "
                  f"{g['conf25']['recall']:.3f} vs {jr['conf25']['recall']:.3f} "
                  f"(delta {gate['delta_vs_eval_jpeg_conf25']:+.3f}) | "
                  f"pixel MAE {gate['jpeg_vs_raw_pixel_mae']:.2f}/255")
        print(acc[DET_A[0]][BASELINE]["conf25"].table(
            f"SANITY GATE A  {DET_A[0]} / {BASELINE} / RAW BMP / conf>={args.conf_hi} "
            f"(full-72 ref {REFERENCE['conf25']['recall']:.3f})"))

    base_d = diagnostics.get(BASELINE, {}).get("total", {}).get("sep_dprime")
    for dname, _, vn in dets:
        # delta is always vs THIS detector's own baseline, so the two sweeps are comparable
        base_r = results[dname].get(BASELINE, {}).get("accept_all", {}).get("recall")
        print(f"\n=== {dname}: variant sweep (accept-all recall | conf>= {args.conf_hi} "
              f"recall | sep_dprime); deltas vs this detector's {BASELINE} ===")
        print(f"{'variant':<20}{'accAll':>8}{'d_acc':>8}{'conf25':>8}{'prec25':>8}"
              f"{'AP50':>8}{'dprime':>8}{'d_dpr':>8}   per-class accept-all recall")
        rows = sorted(vn, key=lambda v: -results[dname][v]["accept_all"]["recall"])
        for v in rows:
            a = results[dname][v]["accept_all"]
            c = results[dname][v]["conf25"]
            dp = diagnostics[v]["total"]["sep_dprime"]
            dr = "" if base_r is None else f"{a['recall'] - base_r:+8.3f}"
            dd = "" if base_d is None else f"{dp - base_d:+8.3f}"
            pc = " ".join(f"{k.split('_')[-1][:4]}:{x:.2f}"
                          for k, x in a["per_class_recall"].items())
            ap50 = a["ap50"]
            print(f"{v:<20}{a['recall']:>8.3f}{dr:>8}{c['recall']:>8.3f}"
                  f"{c['precision']:>8.3f}{(ap50 if ap50 is not None else float('nan')):>8.3f}"
                  f"{dp:>8.3f}{dd:>8}   {pc}")

    payload = {
        "script": "scripts/probe_preprocess.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "limit": args.limit, "imgsz": args.imgsz, "batch": args.batch,
            "conf_lo": args.conf_lo, "conf_hi": args.conf_hi, "jpeg_gate": jpeg_gate,
            "variants": names, "det_b_variants": det_b_names,
            "detector_a": {"name": DET_A[0], "weights": str(DET_A[1])},
            "detector_b": {"name": DET_B[0], "weights": str(DET_B[1])} if det_b_names else None,
            "source": "RAW BMP (lossless) via build_manifest, NOT the pre-CLAHE'd eval JPEGs",
        },
        "frames": stems,
        "n_frames": len(stems),
        "n_gt_boxes": n_gt,
        "elapsed_s": round(time.time() - t0, 1),
        "sanity_gate": gate,
        "diagnostics": diagnostics,
        "detectors": results,
        "confound": CONFOUND_NOTE,
        "diagnostic_definitions": {
            "sep_dprime": "|mu_obj - mu_belt| / sqrt(0.5*(var_obj+var_belt)); affine-invariant",
            "weber": "|mu_obj - mu_belt| / mu_belt; gain-invariant, not gamma-invariant",
            "rms_local": "std/mean of the dilated box window; not intensity-invariant",
            "regions": "obj = GT box shrunk 10%/side; belt = 35%-dilated window minus ALL GT boxes",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n{CONFOUND_NOTE}\n\nwrote {out}  ({payload['elapsed_s']}s, "
          f"{payload['elapsed_s']/max(len(stems),1):.1f}s/frame)")


if __name__ == "__main__":
    main()
