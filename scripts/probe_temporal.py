"""Buy detection recall with TIME — two physically independent uses of the frame sequence.

WHY: ft1's accept-all recall ceiling is 0.489, and the deficit is concentrated in the two
dark-body classes (ni_mh_all 0.12, li_ion_mobile 0.45). Those objects fail not because the
detector is weak but because the *signal* is weak: a battery-vs-belt contrast of a few grey
levels sitting under a measured per-frame temporal noise of ~6 grey levels mean|delta|, on a
belt whose illumination varies 1.9x from center to edge. Both of those are attackable with
frames the camera already recorded, at zero labelling and zero training cost.

The two ideas are INDEPENDENT and are evaluated separately and combined:

A1  motion-aligned temporal stacking. The belt moves, and objects ride it, so in belt-aligned
    coordinates the scene is STATIC. Aligning neighbours on the belt shift lines up belt AND
    object together, which makes averaging pure noise reduction (sigma / sqrt(k)) rather than
    motion blur. Note what this is NOT: frame differencing to find "things that move" cannot
    work here, because object and belt share a motion — differencing cancels both.

A2  temporal-median flat-field. In CAMERA coordinates (no alignment) the belt slides past, so
    each pixel sees many different belt patches over a run. The per-pixel temporal median
    therefore keeps only what is STATIC in camera space: vignetting, fixed-pattern noise, lens
    dirt, mean belt brightness. Measured on run 92 this is a 1.89x center/corner falloff that is
    almost purely horizontal (column means: left 44, center 90, right 39; row means flat within
    1 grey level) — a light bar across the belt. Dividing it out equalises an object's
    appearance across the belt width, which is exactly the nuisance variable that makes a dark
    battery at the frame edge invisible.

SHIFT ESTIMATION — the part that decides whether A1 denoises or blurs. Whole-frame mean-absdiff
minimisation is a documented trap here and is REPRODUCED in this script as a control: the belt
is nearly featureless and objects cover a tiny area fraction, so that objective is insensitive
to the true shift and returns garbage (0 / 80 / 110 px on pairs whose true shift is ~155 px).
This script uses Hanning-windowed `cv2.phaseCorrelate`, and rather than trusting it, VALIDATES
it against detected-object centroid displacement — the same check that exposed the trap. Both
axes are compensated: dy is not 0, it is a consistent ~+4 px/frame, confirmed independently by
phase correlation AND by the detector centroids, and over an 8-frame stack that is ~32 px of
uncorrected vertical drift if ignored.

Everything is scored through `batterycv.evalutil` so the numbers are directly comparable to the
published ft1 table, and every method is additionally scored by UNION COVERAGE with the
unmodified baseline: a method that merely re-finds the objects the baseline already found is
worthless even when its aggregate recall ties.

    python scripts/probe_temporal.py --limit 6                 # smoke test (do this first)
    python scripts/probe_temporal.py                           # full 72-frame run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.evalutil import (
    CLASSES,
    Accumulator,
    class_of,
    coverage,
    load_gt,
    per_class_coverage,
    union_recall,
)
from batterycv.io import build_manifest, segment_runs
from batterycv.preprocess import normalize_illumination, read_bgr

WEIGHTS = "runs/detect/battery_ft1/weights/best.pt"
IMGSZ = 1024
DEPLOY_CONF = 0.25

# Published ft1 numbers on eval/images with this harness — the sanity gate's target.
PUBLISHED = {
    "accept_all": {"recall": 0.489, "precision": 0.079, "ap50": 0.239},
    "conf25": {"recall": 0.452, "precision": 0.410, "ap50": 0.233},
}


# --------------------------------------------------------------------------- run indexing
def build_run_index(raw_dir: Path, gap_s: float = 2.0) -> dict[str, tuple[int, int, list[str]]]:
    """{raw stem: (run_id, index within run, ordered frame paths of that run)}.

    The eval stem is `<class>__<raw stem>`, so stripping the class prefix gives the key. This is
    the only bridge between the hand-labelled eval set and the raw capture sequence, so the
    caller sanity-gates it by rebuilding the baseline through it.
    """
    runs = segment_runs(build_manifest(raw_dir), gap_s=gap_s)
    index: dict[str, tuple[int, int, list[str]]] = {}
    for run_id, grp in runs.groupby("run_id"):
        paths = [str(p) for p in grp.sort_values("timestamp")["path"]]
        for k, p in enumerate(paths):
            index[Path(p).stem] = (int(run_id), k, paths)
    return index


def neighbor_indices(center: int, n_run: int, k_total: int) -> list[int]:
    """Pick `k_total` frame indices centred on `center`, alternating after/before.

    Symmetric selection keeps the MAXIMUM shift (and hence the invalid border and the
    accumulated shift error) as small as possible for a given stack depth: +-4 frames rather
    than +7. Falls back to whatever the run actually offers near its ends.
    """
    chosen = [center]
    step = 1
    while len(chosen) < k_total and step < n_run:
        for cand in (center + step, center - step):
            if 0 <= cand < n_run and cand not in chosen and len(chosen) < k_total:
                chosen.append(cand)
        step += 1
    return sorted(chosen)


# --------------------------------------------------------------------------- shift estimation
def _gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def pairwise_shift(g_a: np.ndarray, g_b: np.ndarray,
                   window: np.ndarray) -> tuple[float, float, float]:
    """Shift APPLIED TO `g_a` that produces `g_b`, via Hanning-windowed phase correlation.

    Sign convention verified empirically against synthetic translations: warping `g_a` by
    (+dx,+dy) reproduces `g_b`, so warping `g_b` by (-dx,-dy) brings it into `g_a`'s frame.
    The returned response is the correlation peak sharpness, used here only as a reported
    quality signal (it drops to ~0.2 on pairs where an object is entering the frame).
    """
    (dx, dy), resp = cv2.phaseCorrelate(g_a, g_b, window)
    return float(dx), float(dy), float(resp)


def cumulative_shifts(grays: list[np.ndarray], ref_pos: int,
                      window: np.ndarray) -> tuple[list[tuple[float, float]], list[float]]:
    """Shift of every frame relative to the reference, by CHAINING consecutive pairs.

    Chaining rather than estimating ref->neighbour directly: consecutive frames overlap ~88%
    (155 px of 1280), while a 4-frame gap overlaps only ~half the frame, which is where phase
    correlation starts to fail. The cost is accumulated error, which is why the accumulated
    shift is cross-checked against detector centroid displacement over the full span.
    """
    n = len(grays)
    shifts: list[tuple[float, float]] = [(0.0, 0.0)] * n
    resps: list[float] = [1.0] * n
    for j in range(ref_pos + 1, n):
        dx, dy, r = pairwise_shift(grays[j - 1], grays[j], window)
        shifts[j] = (shifts[j - 1][0] + dx, shifts[j - 1][1] + dy)
        resps[j] = r
    for j in range(ref_pos - 1, -1, -1):
        dx, dy, r = pairwise_shift(grays[j], grays[j + 1], window)
        shifts[j] = (shifts[j + 1][0] - dx, shifts[j + 1][1] - dy)
        resps[j] = r
    return shifts, resps


def brute_force_shift(g_a: np.ndarray, g_b: np.ndarray, rng: int = 260,
                      step: int = 10) -> float:
    """CONTROL, expected to FAIL: shift minimising whole-frame mean absdiff.

    Kept in the script (not deleted after being disproved) because it is the reason to trust
    phase correlation. If this ever agreed with phase correlation the alignment story would need
    re-examining; it does not, and the disagreement is reported.
    """
    h, w = g_a.shape
    best_val, best_s = np.inf, 0.0
    for s in range(-rng, rng + 1, step):
        warped = cv2.warpAffine(g_b, np.float32([[1, 0, -s], [0, 1, 0]]), (w, h))
        val = float(np.abs(warped - g_a)[:, 300:900].mean())
        if val < best_val:
            best_val, best_s = val, float(s)
    return best_s


# --------------------------------------------------------------------------- A1 stacking
def warp_into_ref(bgr: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Translate a neighbour into the reference frame's coordinates. Returns (warped, valid).

    INTER_LINEAR, not nearest: the shifts are sub-pixel and rounding them to whole pixels would
    inject up to 0.5 px of avoidable misalignment into every layer of the stack.
    """
    h, w = bgr.shape[:2]
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    warped = cv2.warpAffine(bgr.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(np.ones((h, w), np.float32), m, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    return warped, valid


def _masked_median(stack: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    """Per-pixel median over only the VALID layers. Equivalent to nanmedian, ~7x faster.

    `np.nanmedian` takes ~10 s on an 8-frame 1280x1024x3 stack, which would dominate the probe.
    Pushing invalid entries to +inf and sorting puts every valid value first, so the median is
    just the middle element(s) of the per-pixel valid count. The reference layer is always valid,
    so the count is never 0.
    """
    filled = np.where(mask3, stack, np.inf)
    s = np.sort(filled, axis=0)
    cnt = mask3.sum(0)                                   # (H,W,1), >= 1
    shape = (1,) + stack.shape[1:]
    lo = np.broadcast_to(((cnt - 1) // 2)[None], shape)
    hi = np.broadcast_to((cnt // 2)[None], shape)
    return (np.take_along_axis(s, lo, 0)[0] + np.take_along_axis(s, hi, 0)[0]) / 2.0


def stack_frames(ref_bgr: np.ndarray, neighbors: list[tuple[np.ndarray, float, float]],
                 how: str) -> tuple[np.ndarray, np.ndarray]:
    """Combine reference + motion-aligned neighbours. Returns (composite uint8, neighbour count).

    The composite stays in the REFERENCE frame's coordinate system, which is what keeps the GT
    boxes valid. Where a pixel received no neighbour contribution (the border swept in/out by
    belt motion) the reference's own value is used verbatim rather than averaging in the zeros
    that `warpAffine` pads with — that fallback is the difference between a clean border and a
    black wedge the detector would happily fire on.
    """
    h, w = ref_bgr.shape[:2]
    layers = [ref_bgr.astype(np.float32)]
    masks = [np.ones((h, w), bool)]
    for nb, dx, dy in neighbors:
        warped, valid = warp_into_ref(nb, dx, dy)
        layers.append(warped)
        masks.append(valid)
    n_neighbors = np.sum(np.stack(masks[1:], 0), axis=0).astype(np.int16) if neighbors \
        else np.zeros((h, w), np.int16)

    stack = np.stack(layers, 0)                      # (k,H,W,3)
    mask3 = np.stack(masks, 0)[..., None]            # (k,H,W,1)
    if how == "mean":
        tot = (stack * mask3).sum(0)
        cnt = mask3.sum(0).astype(np.float32)
        comp = tot / np.maximum(cnt, 1.0)
    elif how == "median":
        comp = _masked_median(stack, mask3)
    else:
        raise ValueError(f"unknown combine mode {how!r}")

    # pixels with no neighbour contribution fall back to the reference exactly
    fallback = n_neighbors == 0
    comp = np.where(fallback[..., None], ref_bgr.astype(np.float32), comp)
    return np.clip(comp, 0, 255).astype(np.uint8), n_neighbors


# --------------------------------------------------------------------------- A2 flat field
def compute_flat_field(paths: list[str], max_frames: int, min_frames: int) -> np.ndarray | None:
    """Per-pixel temporal median over a run, in CAMERA coordinates. None if the run is too short.

    No alignment on purpose: the belt sliding past is precisely what makes each pixel see many
    different belt patches, so the median collapses to what is static in camera space. It only
    works if the object is swept clear of every pixel — at ~155 px/frame an object crosses the
    1280 px frame in ~9 frames, hence the `min_frames` floor. Frames are sub-sampled evenly
    (not taken from the head of the run) so the estimate spans the whole traverse.
    """
    if len(paths) < min_frames:
        return None
    if len(paths) > max_frames:
        idx = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
        paths = [paths[i] for i in sorted(set(idx.tolist()))]
    stack = np.stack([read_bgr(p) for p in paths], 0)     # uint8, memory-conscious
    return np.median(stack, axis=0).astype(np.float32)


def apply_flat_field(bgr: np.ndarray, flat: np.ndarray, mode: str) -> np.ndarray:
    """Divide or subtract the flat field, renormalising to preserve mean brightness."""
    f = bgr.astype(np.float32)
    if mode == "divide":
        out = f / np.maximum(flat, 1.0) * float(flat.mean())
    elif mode == "subtract":
        out = f - flat + float(flat.mean())
    else:
        raise ValueError(f"unknown flat-field mode {mode!r}")
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- noise measurement
def flat_patch(gt_boxes: np.ndarray, h: int, w: int, size: int = 128) -> tuple[int, int]:
    """Pick a deterministic belt-only patch: grid candidates, drop any overlapping GT, take the
    one with the least gradient energy proxy (most central-but-empty). Used to measure noise
    where there is no object, so the metric reflects sensor noise and not object structure."""
    best = (h // 2 - size // 2, w // 2 - size // 2)
    best_score = -np.inf
    for y in range(64, h - size - 64, 96):
        for x in range(64, w - size - 64, 96):
            box = np.array([x, y, x + size, y + size], float)
            if len(gt_boxes):
                ox = np.minimum(box[2], gt_boxes[:, 2]) - np.maximum(box[0], gt_boxes[:, 0])
                oy = np.minimum(box[3], gt_boxes[:, 3]) - np.maximum(box[1], gt_boxes[:, 1])
                if np.any((ox > 0) & (oy > 0)):
                    continue
            # prefer patches near the bright center band (where belt contrast is measurable)
            score = -abs((x + size / 2) - w / 2) - abs((y + size / 2) - h / 2)
            if score > best_score:
                best_score, best = score, (y, x)
    return best


def hf_noise(bgr: np.ndarray, yx: tuple[int, int], size: int = 128) -> float:
    """High-frequency residual std in a flat patch = std(patch - gaussian_blur(patch)).

    Plain spatial std would be dominated by real belt texture, which stacking does NOT remove
    and which would mask the effect. Subtracting a smoothed copy isolates the pixel-scale
    component, which is where the sqrt(k) reduction should show up if the stack is working.
    """
    y, x = yx
    patch = _gray(bgr)[y:y + size, x:x + size]
    return float((patch - cv2.GaussianBlur(patch, (5, 5), 0)).std())


def edge_energy(bgr: np.ndarray, gt_boxes: np.ndarray, patch: tuple[int, int],
                size: int = 128) -> tuple[float, float]:
    """Gradient ENERGY inside the GT boxes and in an object-free belt patch. The blur test.

    Returns (E_box, E_flat) where E = mean(gx^2 + gy^2). Reported as energy, not magnitude,
    because gradient energy is additive over independent components:

        E_box = E_signal + E_noise      and      E_flat = E_noise

    so `E_box - E_flat` estimates the OBJECT's own edge energy with the sensor-noise (and belt
    texture) contribution removed. That subtraction is the whole point. A raw mean-|Sobel| inside
    the box cannot answer the question, because noise inflates it: denoising alone would lower it
    even under zero blur, making a perfect denoiser look exactly like a blur. With the correction:

        pure denoising  -> E_flat falls,  (E_box - E_flat) stays flat
        blurring        -> E_flat falls,  (E_box - E_flat) falls too

    Most of a GT box is object interior rather than boundary, so the noise term is a large share
    of the raw number — this correction is not a refinement, it decides the conclusion.
    """
    if not len(gt_boxes):
        return float("nan"), float("nan")
    g = _gray(bgr)
    h, w = g.shape
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    e = gx * gx + gy * gy
    vals = []
    for b in gt_boxes:
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(w, int(b[2])), min(h, int(b[3]))
        if x2 > x1 and y2 > y1:
            vals.append(float(e[y1:y2, x1:x2].mean()))
    y, x = patch
    return (float(np.mean(vals)) if vals else float("nan"),
            float(e[y:y + size, x:x + size].mean()))


def match_boxes_across_frames(rb: np.ndarray, nb: np.ndarray, max_dx: float = 380.0,
                              max_dy: float = 60.0, area_tol: float = 1.6
                              ) -> list[tuple[float, float]]:
    """Displacements of objects matched between ADJACENT frames, by size + plausible motion.

    Naively pairing the highest-confidence box in each frame is wrong when a frame holds several
    objects — it silently compares two DIFFERENT batteries and reports a bogus residual (this
    produced a spurious 78 px "error" in an earlier version). Matching on box-area similarity
    within a physically plausible displacement window fixes that. Only adjacent frames are used
    so the true displacement is bounded and the correspondence is unambiguous.
    """
    pairs: list[tuple[float, float]] = []
    if not len(rb) or not len(nb):
        return pairs
    ra = (rb[:, 2] - rb[:, 0]) * (rb[:, 3] - rb[:, 1])
    na = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
    rc = np.stack([(rb[:, 0] + rb[:, 2]) / 2, (rb[:, 1] + rb[:, 3]) / 2], 1)
    nc = np.stack([(nb[:, 0] + nb[:, 2]) / 2, (nb[:, 1] + nb[:, 3]) / 2], 1)
    for i in range(len(rb)):
        best, best_cost = None, np.inf
        for j in range(len(nb)):
            dx, dy = nc[j, 0] - rc[i, 0], nc[j, 1] - rc[i, 1]
            if abs(dx) > max_dx or abs(dy) > max_dy:
                continue
            ar = na[j] / max(ra[i], 1e-9)
            if ar > area_tol or ar < 1.0 / area_tol:
                continue
            cost = abs(np.log(ar)) + abs(dy) / max_dy
            if cost < best_cost:
                best_cost, best = cost, (float(dx), float(dy))
        if best is not None:
            pairs.append(best)
    return pairs


# --------------------------------------------------------------------------- detection/scoring
def detect(model, bgr: np.ndarray, conf: float, device: str) -> tuple[np.ndarray, np.ndarray]:
    res = model.predict(bgr, conf=conf, imgsz=IMGSZ, device=device, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    return (res.boxes.xyxy.cpu().numpy().astype(float),
            res.boxes.conf.cpu().numpy().astype(float))


def per_class_recall(acc: Accumulator) -> dict[str, float | None]:
    """Per-class recall without mutating Accumulator.stats (it is a defaultdict)."""
    out: dict[str, float | None] = {}
    for c in CLASSES:
        if c in acc.stats:
            tp, _, ngt, _ = acc.stats[c]
            out[c] = tp / max(ngt, 1)
        else:
            out[c] = None
    return out


def score(preds: dict[str, tuple[np.ndarray, np.ndarray]], gt: dict[str, np.ndarray]) -> dict:
    acc = Accumulator()
    for stem, g in gt.items():
        b, s = preds.get(stem, (np.zeros((0, 4)), np.zeros(0)))
        acc.add(stem, b, g, s)
    tp, fp, ngt, nbox = acc._agg(None)
    ap = acc.ap50()
    return {
        "recall": tp / max(ngt, 1),
        "precision": tp / max(tp + fp, 1),
        "boxes_per_frame": nbox / max(acc.n_frames, 1),
        "ap50": None if ap != ap else float(ap),
        "n_gt": int(ngt),
        "per_class_recall": per_class_recall(acc),
    }


HEADER = (f"{'method':<30}{'box/f':>7}{'R':>7}{'P':>7}{'AP50':>7} |"
          + "".join(f"{c.replace('li_ion_', '').replace('ni_cd_', '')[:6]:>7}" for c in CLASSES))


def row(label: str, r: dict) -> str:
    pc = r["per_class_recall"]
    cells = "".join(f"{pc[c]:>7.3f}" if pc[c] is not None else f"{'-':>7}" for c in CLASSES)
    ap = f"{r['ap50']:>7.3f}" if r["ap50"] is not None else f"{'-':>7}"
    return (f"{label:<30}{r['boxes_per_frame']:>7.1f}{r['recall']:>7.3f}"
            f"{r['precision']:>7.3f}{ap} |{cells}")


def cov_stats(cov: dict[str, np.ndarray]) -> dict:
    pc = per_class_coverage(cov)
    tot = sum(v[1] for v in pc.values())
    hit = sum(v[0] for v in pc.values())
    return {"recall": hit / max(tot, 1), "covered": hit, "total": tot,
            "per_class": {c: {"covered": pc[c][0], "total": pc[c][1],
                              "recall": pc[c][0] / max(pc[c][1], 1)} for c in sorted(pc)}}


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test on N eval frames spread EVENLY over the sorted set so "
                         "several classes are exercised (0 = all 72)")
    ap.add_argument("--stack-k", default="2,4,8",
                    help="comma-separated total frame counts for A1 (reference included)")
    ap.add_argument("--combine", default="mean,median", help="A1 combine modes to compare")
    ap.add_argument("--ff-mode", default="divide,subtract", help="A2 flat-field modes")
    ap.add_argument("--ff-max-frames", type=int, default=32,
                    help="cap on frames used for the per-run temporal median")
    ap.add_argument("--ff-min-frames", type=int, default=9,
                    help="a run shorter than this cannot sweep an object clear of every pixel")
    ap.add_argument("--best-k", type=int, default=4, help="stack depth used for the A1+A2 combo")
    ap.add_argument("--conf", type=float, default=0.001, help="accept-all floor")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads. Measured on this 8-logical-core box: 4 threads runs "
                         "4.2 s/inference vs 11.3 s at the default 8 — the default oversubscribes "
                         "hyperthreads. 0 leaves torch alone.")
    ap.add_argument("--align-frames", type=int, default=8,
                    help="validate the shift estimate against detector centroids on the first N "
                         "frames only (each check costs an extra inference + the brute-force control)")
    ap.add_argument("--out", default="results/phase4/temporal.json")
    args = ap.parse_args()

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    repo = Path(__file__).resolve().parents[1]
    ks = [int(x) for x in args.stack_k.split(",") if x.strip()]
    combines = [c.strip() for c in args.combine.split(",") if c.strip()]
    ff_modes = [c.strip() for c in args.ff_mode.split(",") if c.strip()]
    # the A1+A2 combo is built inside the stacking loop, so its depth/mode must be swept
    if args.best_k not in ks or "mean" not in combines:
        ap.error(f"--best-k {args.best_k} must be one of --stack-k {ks} and 'mean' must be in "
                 f"--combine {combines}, otherwise the A1+A2 combination is silently never built")

    paths = load_paths()
    img_dir, lbl_dir = paths["eval_dir"] / "images", paths["eval_dir"] / "labels"
    imgs = sorted(img_dir.glob("*.jpg"))
    if args.limit and args.limit < len(imgs):
        idx = np.linspace(0, len(imgs) - 1, args.limit).round().astype(int)
        imgs = [imgs[i] for i in sorted(set(idx.tolist()))]
    gt_all = load_gt(lbl_dir)
    gt = {p.stem: gt_all.get(p.stem, np.zeros((0, 4))) for p in imgs}
    n_gt = sum(len(v) for v in gt.values())

    print(f"probe_temporal: {len(imgs)} eval frames, {n_gt} GT boxes, "
          f"classes={sorted({class_of(p.stem) for p in imgs})}")
    if args.limit:
        print("  SMOKE MODE — full run is 72 frames")

    print("\n[0] indexing raw capture runs")
    index = build_run_index(paths["raw_dir"])
    mapped = [p for p in imgs if p.stem.split("__", 1)[1] in index]
    if len(mapped) != len(imgs):
        raise SystemExit(f"ABORT: only {len(mapped)}/{len(imgs)} eval stems map to raw frames — "
                         "the raw mapping is broken, everything downstream would be invalid.")
    print(f"  all {len(imgs)} eval stems map to raw frames")

    from ultralytics import YOLO
    model = YOLO(str(repo / args.weights) if not Path(args.weights).is_absolute() else args.weights)

    han = cv2.createHanningWindow((1280, 1024), cv2.CV_32F)
    flat_cache: dict[int, np.ndarray | None] = {}

    # predictions per method
    P: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    def put(method: str, stem: str, bs: tuple[np.ndarray, np.ndarray]) -> None:
        P.setdefault(method, {})[stem] = bs

    align_records: list[dict] = []
    noise_records: list[dict] = []
    sharp_records: list[dict] = []
    border_records: list[dict] = []
    ff_missing = 0
    t0 = time.time()

    for n, p in enumerate(imgs):
        stem = p.stem
        raw_stem = stem.split("__", 1)[1]
        run_id, pos, run_paths = index[raw_stem]
        gt_boxes = gt[stem]

        ref_bgr = read_bgr(run_paths[pos])
        h, w = ref_bgr.shape[:2]
        patch = flat_patch(gt_boxes, h, w)

        # ---- B0: baseline from eval/images (the published path) ---------------------------
        put("B0_evaljpg", stem, detect(model, normalize_illumination(read_bgr(str(p))),
                                       args.conf, args.device))
        # ---- B1: baseline rebuilt from the RAW BMP (the sanity gate) -----------------------
        ref_clahe = normalize_illumination(ref_bgr)
        put("B1_raw", stem, detect(model, ref_clahe, args.conf, args.device))

        # ---- A1: motion-aligned stacking ---------------------------------------------------
        kmax = max(ks)
        sel = neighbor_indices(pos, len(run_paths), kmax)
        bgrs = {j: (ref_bgr if j == pos else read_bgr(run_paths[j])) for j in sel}
        grays = [_gray(bgrs[j]) for j in sel]
        ref_pos = sel.index(pos)
        shifts, resps = cumulative_shifts(grays, ref_pos, han)

        # ---- validate the shift estimate against detected-object centroid displacement -------
        # Adjacent frame only: the correspondence is unambiguous there, so a disagreement is a
        # real shift error rather than a matching artefact.
        rb, rs = P["B1_raw"][stem]
        adj = pos + 1 if (pos + 1) in sel else (pos - 1 if (pos - 1) in sel else None)
        if n < args.align_frames and adj is not None and len(rb):
            k_at = sel.index(adj)
            nbb, nbs = detect(model, normalize_illumination(bgrs[adj]), DEPLOY_CONF, args.device)
            rb25 = rb[rs >= DEPLOY_CONF]
            pairs = match_boxes_across_frames(rb25, nbb)
            # every estimator here reports the SAME thing: the shift taking ref -> adj. That
            # holds for adj before or after the reference, so no sign flip belongs anywhere.
            pdx = (shifts[k_at][0] - shifts[ref_pos][0])
            pdy = (shifts[k_at][1] - shifts[ref_pos][1])
            rec = {"stem": stem, "run_id": run_id, "adj_offset": int(adj - pos),
                   "phase_dx": pdx, "phase_dy": pdy, "n_matched": len(pairs),
                   "min_response": float(min(resps)),
                   "brute_force_dx": brute_force_shift(grays[ref_pos], grays[k_at])}
            if pairs:
                cdx = float(np.median([p[0] for p in pairs]))
                cdy = float(np.median([p[1] for p in pairs]))
                rec.update({"centroid_dx": cdx, "centroid_dy": cdy,
                            "residual_dx": pdx - cdx, "residual_dy": pdy - cdy})
            align_records.append(rec)

        noise_rec = {"stem": stem, "ref": hf_noise(ref_bgr, patch)}
        sharp_rec = {"stem": stem, "ref": edge_energy(ref_bgr, gt_boxes, patch)}
        for k in ks:
            sub = neighbor_indices(pos, len(run_paths), k)
            neigh = [(bgrs[j], shifts[sel.index(j)][0], shifts[sel.index(j)][1])
                     for j in sub if j != pos]
            for how in combines:
                comp, ncnt = stack_frames(ref_bgr, neigh, how)
                name = f"A1_stack{k}_{how}"
                put(name, stem, detect(model, normalize_illumination(comp),
                                       args.conf, args.device))
                noise_rec[name] = hf_noise(comp, patch)
                sharp_rec[name] = edge_energy(comp, gt_boxes, patch)
                if how == combines[0]:
                    border_records.append({
                        "stem": stem, "k": k, "n_available": len(neigh) + 1,
                        "frac_no_neighbor": float((ncnt == 0).mean()),
                        "frac_partial": float((ncnt < len(neigh)).mean()),
                        "mean_depth": float(ncnt.mean() + 1),
                    })
                # ---- A1+A2 combined (only at --best-k) ------------------------------------
                if k == args.best_k and how == "mean":
                    if run_id not in flat_cache:
                        flat_cache[run_id] = compute_flat_field(
                            run_paths, args.ff_max_frames, args.ff_min_frames)
                    flat = flat_cache[run_id]
                    if flat is not None:
                        for fm in ff_modes:
                            put(f"A1A2_stack{k}_{fm}", stem,
                                detect(model, normalize_illumination(apply_flat_field(comp, flat, fm)),
                                       args.conf, args.device))
        noise_records.append(noise_rec)
        sharp_records.append(sharp_rec)

        # ---- A2 alone -----------------------------------------------------------------------
        if run_id not in flat_cache:
            flat_cache[run_id] = compute_flat_field(run_paths, args.ff_max_frames,
                                                    args.ff_min_frames)
        flat = flat_cache[run_id]
        if flat is None:
            ff_missing += 1
            for fm in ff_modes:
                put(f"A2_{fm}", stem, P["B1_raw"][stem])        # honest fallback: unmodified
                put(f"A1A2_stack{args.best_k}_{fm}", stem, P["B1_raw"][stem])
        else:
            for fm in ff_modes:
                put(f"A2_{fm}", stem, detect(model, normalize_illumination(
                    apply_flat_field(ref_bgr, flat, fm)), args.conf, args.device))

        # flush: the full run is ~1 min/frame, and a redirected stdout would otherwise buffer
        # the whole hour of progress into invisibility
        print(f"  [{n+1}/{len(imgs)}] {stem[:52]:<52} run={run_id:<4} "
              f"pos={pos}/{len(run_paths)} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0

    # ---- sanity gate -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("[1] SANITY GATE — baseline rebuilt from RAW BMP must land near the published numbers")
    print("=" * 110)
    gate: dict[str, dict] = {}
    print(HEADER)
    for m in ("B0_evaljpg", "B1_raw"):
        a = score(P[m], gt)
        d25 = {s: (b[sc >= DEPLOY_CONF], sc[sc >= DEPLOY_CONF]) for s, (b, sc) in P[m].items()}
        c = score(d25, gt)
        gate[m] = {"accept_all": a, "conf25": c}
        print(row(f"{m} accept-all", a))
        print(row(f"{m} conf>=0.25", c))
    print(f"\n  published (full 72, eval/images): accept-all R {PUBLISHED['accept_all']['recall']:.3f} "
          f"| conf25 R {PUBLISHED['conf25']['recall']:.3f}")
    d_all = gate["B1_raw"]["accept_all"]["recall"] - gate["B0_evaljpg"]["accept_all"]["recall"]
    d_25 = gate["B1_raw"]["conf25"]["recall"] - gate["B0_evaljpg"]["conf25"]["recall"]
    print(f"  RAW vs eval/jpg on THESE frames: accept-all {d_all:+.3f}, conf25 {d_25:+.3f}")
    print("  (a small gap is expected — eval/images went through JPEG; a large one means the raw"
          "\n   stem mapping is broken and every temporal number below is invalid)")

    # ---- alignment validation ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("[2] SHIFT-ESTIMATE VALIDATION — phase correlation vs detected-object centroid")
    print("=" * 110)
    matched = [a for a in align_records if a.get("n_matched")]
    if matched:
        rdx = np.array([a["residual_dx"] for a in matched])
        rdy = np.array([a["residual_dy"] for a in matched])
        print(f"{'stem':<42}{'off':>4}{'n':>3}{'phase dx':>10}{'cent dx':>9}{'res dx':>8}"
              f"{'phase dy':>10}{'cent dy':>9}{'res dy':>8}{'resp':>7}{'brute dx':>10}")
        for a in matched:
            print(f"{a['stem'][:42]:<42}{a['adj_offset']:>4}{a['n_matched']:>3}"
                  f"{a['phase_dx']:>10.1f}{a['centroid_dx']:>9.1f}{a['residual_dx']:>8.1f}"
                  f"{a['phase_dy']:>10.1f}{a['centroid_dy']:>9.1f}{a['residual_dy']:>8.1f}"
                  f"{a['min_response']:>7.3f}{a['brute_force_dx']:>10.1f}")
        print(f"\n  adjacent-frame |residual dx| median {np.median(np.abs(rdx)):.1f} px, "
              f"max {np.abs(rdx).max():.1f} px  (n={len(matched)} frames)")
        print(f"  adjacent-frame |residual dy| median {np.median(np.abs(rdy)):.1f} px, "
              f"max {np.abs(rdy).max():.1f} px")
        print(f"  median GT object width ~179 px => a {np.median(np.abs(rdx)):.1f} px residual is "
              f"~{100*np.median(np.abs(rdx))/179:.1f}% of an object. Stacking k frames chains up to "
              f"{(max(ks)-1)//2} such steps, so worst-case smear ~{(max(ks)-1)//2*np.median(np.abs(rdx)):.0f} px.")
        pf = np.array([a["phase_dy"] for a in matched])
        print(f"  dy is NOT zero: phase correlation says {np.median([a['phase_dy'] for a in matched]):+.1f} px/frame "
              f"and the detector centroids independently say {np.median([a['centroid_dy'] for a in matched]):+.1f} px/frame. "
              f"Both axes are compensated here.")
    else:
        print("  no frames yielded a confident box correspondence — alignment not independently "
              "validated on this subset.")
    if align_records:
        bf = np.array([a["brute_force_dx"] for a in align_records])
        pd = np.array([a["phase_dx"] for a in align_records])
        print(f"  CONTROL: whole-frame mean-absdiff brute force disagrees with phase correlation by "
              f"a median of {np.median(np.abs(bf - pd)):.0f} px over {len(align_records)} frames "
              f"— the documented trap, reproduced rather than assumed.")

    # ---- noise + border --------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("[3] DID THE STACK ACTUALLY DENOISE? high-frequency residual std in a GT-free belt patch")
    print("=" * 110)
    noise_summary: dict[str, dict] = {}
    ref_n = np.array([r["ref"] for r in noise_records])
    print(f"{'method':<24}{'HF noise':>10}{'vs ref':>9}{'predicted':>11}   (predicted = ref/sqrt(k))")
    print(f"{'reference frame':<24}{ref_n.mean():>10.3f}{1.0:>9.2f}{'-':>11}")
    for k in ks:
        for how in combines:
            name = f"A1_stack{k}_{how}"
            v = np.array([r[name] for r in noise_records])
            ratio = float(v.mean() / max(ref_n.mean(), 1e-9))
            noise_summary[name] = {"hf_noise": float(v.mean()), "ratio_to_ref": ratio,
                                   "predicted_ratio": 1.0 / np.sqrt(k)}
            print(f"{name:<24}{v.mean():>10.3f}{ratio:>9.2f}{1/np.sqrt(k):>11.2f}")

    print("\n  DENOISED OR JUST BLURRED? gradient energy in the GT boxes, noise-corrected")
    print("  E_obj = E_box - E_flat removes the noise/texture term, leaving the OBJECT's edges.")
    print("  pure denoising => E_obj ratio ~1.00 while noise falls. blur => E_obj falls too.")
    sharp_summary: dict[str, dict] = {}

    def _obj_energy(key: str) -> float:
        vals = [r[key][0] - r[key][1] for r in sharp_records if r[key][0] == r[key][0]]
        return float(np.mean(vals)) if vals else float("nan")

    ref_obj = _obj_energy("ref")
    ref_raw = float(np.mean([r["ref"][0] for r in sharp_records if r["ref"][0] == r["ref"][0]]))
    print(f"{'method':<24}{'E_box(raw)':>12}{'E_obj(corr)':>13}{'E_obj vs ref':>14}{'noise vs ref':>14}")
    print(f"{'reference frame':<24}{ref_raw:>12.1f}{ref_obj:>13.1f}{1.0:>14.2f}{1.0:>14.2f}")
    for k in ks:
        for how in combines:
            name = f"A1_stack{k}_{how}"
            raw = float(np.mean([r[name][0] for r in sharp_records if r[name][0] == r[name][0]]))
            obj = _obj_energy(name)
            ratio = obj / max(ref_obj, 1e-9)
            nratio = noise_summary[name]["ratio_to_ref"] ** 2   # noise ENERGY, comparable units
            sharp_summary[name] = {"edge_energy_raw": raw, "edge_energy_object": obj,
                                   "object_ratio_to_ref": ratio, "noise_energy_ratio": nratio}
            print(f"{name:<24}{raw:>12.1f}{obj:>13.1f}{ratio:>14.2f}{nratio:>14.2f}")

    print("\n  border cost (belt motion sweeps content in/out of the stack):")
    print(f"{'k':<5}{'avail':>8}{'no-neighbor px':>16}{'partial px':>13}{'mean depth':>12}")
    border_summary: dict[str, dict] = {}
    for k in ks:
        rs_ = [b for b in border_records if b["k"] == k]
        if not rs_:
            continue
        s = {"frac_no_neighbor": float(np.mean([b["frac_no_neighbor"] for b in rs_])),
             "frac_partial": float(np.mean([b["frac_partial"] for b in rs_])),
             "mean_depth": float(np.mean([b["mean_depth"] for b in rs_])),
             "mean_available": float(np.mean([b["n_available"] for b in rs_]))}
        border_summary[str(k)] = s
        print(f"{k:<5}{s['mean_available']:>8.1f}{s['frac_no_neighbor']:>16.1%}"
              f"{s['frac_partial']:>13.1%}{s['mean_depth']:>12.2f}")

    # ---- headline table --------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("[4] DETECTION — accept-all (recall ceiling) and conf>=0.25 (deployment)")
    print("=" * 110)
    methods = [m for m in P if m not in ("B0_evaljpg",)]
    results: dict[str, dict] = {}
    print(HEADER)
    for m in methods:
        a = score(P[m], gt)
        d25 = {s: (b[sc >= DEPLOY_CONF], sc[sc >= DEPLOY_CONF]) for s, (b, sc) in P[m].items()}
        c = score(d25, gt)
        results[m] = {"accept_all": a, "conf25": c}
        print(row(f"{m} accept-all", a))
    print()
    for m in methods:
        print(row(f"{m} conf>=0.25", results[m]["conf25"]))

    # ---- union coverage with the baseline (THE payoff metric) ------------------------------
    print("\n" + "=" * 110)
    print("[5] UNION COVERAGE WITH THE BASELINE — does the method find what B1_raw MISSES?")
    print("=" * 110)
    base_cov = coverage({s: b for s, (b, _) in P["B1_raw"].items()}, gt)
    base_r = cov_stats(base_cov)["recall"]
    print(f"{'method':<28}{'own R':>8}{'union':>8}{'gain':>8}{'new':>6}{'lost':>6}"
          f"   new-object classes")
    union: dict[str, dict] = {}
    for m in methods:
        if m == "B1_raw":
            continue
        cov = coverage({s: b for s, (b, _) in P[m].items()}, gt)
        ur = union_recall(base_cov, cov)
        new = lost = 0
        new_cls: dict[str, int] = {}
        for stem, g in gt.items():
            if not len(g):
                continue
            mb, mm = base_cov[stem], cov[stem]
            n_new = int((mm & ~mb).sum())
            new += n_new
            lost += int((mb & ~mm).sum())
            if n_new:
                new_cls[class_of(stem)] = new_cls.get(class_of(stem), 0) + n_new
        own = cov_stats(cov)["recall"]
        union[m] = {"own_recall": own, "union_recall": ur, "gain_over_baseline": ur - base_r,
                    "new_objects": new, "lost_objects": lost, "new_by_class": new_cls,
                    "per_class": cov_stats(cov)["per_class"]}
        print(f"{m:<28}{own:>8.3f}{ur:>8.3f}{ur - base_r:>+8.3f}{new:>6}{lost:>6}   "
              + (", ".join(f"{k}={v}" for k, v in sorted(new_cls.items())) or "-"))
    print(f"\n  baseline B1_raw accept-all coverage recall = {base_r:.3f}")
    print("  'new' = GT objects this method covers that the baseline does not. A method with a")
    print("  matching recall but new=0 is worthless: it re-finds the same objects.")

    # ---- verdict ---------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("[6] VERDICT")
    print("=" * 110)
    best = max((m for m in methods if m != "B1_raw"),
               key=lambda m: results[m]["accept_all"]["recall"])
    print(f"  baseline (raw+CLAHE) accept-all recall : "
          f"{results['B1_raw']['accept_all']['recall']:.3f}")
    print(f"  best temporal method                   : {best} "
          f"{results[best]['accept_all']['recall']:.3f} "
          f"({results[best]['accept_all']['recall'] - results['B1_raw']['accept_all']['recall']:+.3f})")
    if union:
        bu = max(union, key=lambda m: union[m]["gain_over_baseline"])
        print(f"  best union gain over baseline          : {bu} "
              f"{union[bu]['gain_over_baseline']:+.3f} ({union[bu]['new_objects']} new objects)")
    if ff_missing:
        print(f"  NOTE: {ff_missing} frame(s) had a run shorter than {args.ff_min_frames} frames, "
              "so A2 fell back to the unmodified frame for them.")
    print(f"  wall clock {elapsed:.0f}s for {len(imgs)} frames "
          f"({elapsed/max(len(imgs),1):.1f}s/frame, {len(P)} methods)")

    out_path = repo / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"frames": [p.stem for p in imgs], "n_frames": len(imgs), "n_gt": int(n_gt),
                 "weights": args.weights, "imgsz": IMGSZ, "conf_floor": args.conf,
                 "deploy_conf": DEPLOY_CONF, "device": args.device, "limit": args.limit,
                 "smoke": bool(args.limit), "stack_k": ks, "combine": combines,
                 "ff_modes": ff_modes, "ff_max_frames": args.ff_max_frames,
                 "ff_min_frames": args.ff_min_frames, "best_k": args.best_k,
                 "elapsed_seconds": elapsed, "ff_missing_frames": ff_missing},
        "published_reference": PUBLISHED,
        "sanity_gate": gate,
        "alignment_validation": align_records,
        "noise": {"reference_hf": float(ref_n.mean()), "methods": noise_summary},
        "object_edge_energy": {"reference_raw": ref_raw, "reference_object": ref_obj,
                               "methods": sharp_summary},
        "border_cost": border_summary,
        "detection": results,
        "union_with_baseline": {"baseline_recall": base_r, "methods": union},
    }
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
