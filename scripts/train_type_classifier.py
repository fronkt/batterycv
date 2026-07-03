"""Phase 3 — battery type classifier on the tracked crops (visual + trusted-OCR fusion).

Classifies each tracked battery crop into the 6 session types (li_ion_mobile, li_ion_laptop,
ni_cd_bulk, liso2, ni_mh_all, ni_cd_small) and reports a 4-way chemistry grouping
(li_ion / liso2 / ni_cd / ni_mh) from the same model — the grouping a sorting line acts on.

Design decisions (docs/ocr_findings.md is the why):
- **Split is grouped by run id**, never by crop: crops from one run share a conveyor session
  (same batteries, lighting, belt state), so a crop-level split would leak. Per class ~25% of
  runs go to test (min 1); every class has >=3 runs so all 6 appear in both splits.
- **OCR features follow the Phase-2b trust tier**: n_chars, brand presence, part-# presence,
  mark count (+ det_conf). chemistry/voltage/capacity are EXCLUDED — the 3B prior-fills them
  (chemistry is literally constant "Li-ion" and would only teach the model the prior).
- Visual model = yolo11n-cls fine-tune at 224. val==train inside ultralytics is only a fit
  monitor (finetune_detector.py precedent); the honest numbers come from ONE pass over the
  held-out test runs, reported for both best.pt and last.pt.
- Train-side class imbalance (455 vs 10 crops) is handled by duplicating minority-class train
  images toward the largest class, capped at x10 so a 7-image class isn't cloned 50x.
- Ablation on the same test runs: OCR-features-only logistic regression vs visual-only vs
  late fusion (LR on visual probs + OCR features). Caveat: the fusion LR is fit on train-set
  visual probs, which are near-one-hot (the net has fit them) — read the fusion column as
  "does OCR add anything", not as a deployable gain.

    python scripts/train_type_classifier.py                 # full: split -> train -> eval
    python scripts/train_type_classifier.py --skip-train    # re-eval existing weights
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths

CROP_RE = re.compile(r"^(?P<label>.+)__run(?P<run>\d+)__id(?P<tid>\d+)_c(?P<conf>[\d.]+)\.jpg$")

# 6-way session label -> 4-way chemistry (what the sorting line actually diverts on)
CHEM_GROUP = {
    "li_ion_mobile": "li_ion", "li_ion_laptop": "li_ion",
    "liso2": "liso2",
    "ni_cd_bulk": "ni_cd", "ni_cd_small": "ni_cd",
    "ni_mh_all": "ni_mh",
}

OCR_FEATURES = ["n_chars", "has_brand", "has_part", "n_marks", "det_conf"]


def parse_crops(crops_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(crops_dir.glob("*.jpg")):
        m = CROP_RE.match(p.name)
        if not m:
            print(f"  skipping unparseable crop name: {p.name}")
            continue
        rows.append({"path": p, "crop": p.name, "label": m["label"], "run": int(m["run"])})
    return rows


def split_by_run(rows: list[dict], test_frac: float, seed: int) -> tuple[set, set]:
    """Per class: shuffle its run ids, send ~test_frac of them (min 1) to test."""
    rng = np.random.default_rng(seed)
    runs_by_label = defaultdict(set)
    for r in rows:
        runs_by_label[r["label"]].add(r["run"])
    train_runs, test_runs = set(), set()
    for label in sorted(runs_by_label):
        runs = sorted(runs_by_label[label])
        rng.shuffle(runs)
        n_test = max(1, round(test_frac * len(runs)))
        test_runs.update((label, r) for r in runs[:n_test])
        train_runs.update((label, r) for r in runs[n_test:])
    return train_runs, test_runs


def ocr_feature_row(rec: dict | None) -> list[float]:
    if rec is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(rec.get("n_chars") or 0),
        1.0 if str(rec.get("manufacturer") or "").strip() else 0.0,
        1.0 if str(rec.get("model") or "").strip() else 0.0,
        float(len(rec.get("marks") or [])),
        float(rec.get("det_conf") or 0),
    ]


def report(name: str, y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = rep["accuracy"]
    mf1 = rep["macro avg"]["f1-score"]
    print(f"  {name:<28} acc {acc:.3f}  macro-F1 {mf1:.3f}")
    return {"accuracy": acc, "macro_f1": mf1, "per_class": {c: rep[c] for c in labels},
            "confusion": {"labels": labels, "matrix": cm.tolist()}}


def main() -> None:
    paths = load_paths()
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops", default=str(paths["work_dir"] / "crops_all"))
    ap.add_argument("--ocr", default=str(repo / "results/phase2_ocr/ocr.json"))
    ap.add_argument("--out", default=str(paths["work_dir"] / "type_clf"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.25, help="fraction of each class's RUNS held out")
    ap.add_argument("--oversample-cap", type=int, default=10,
                    help="max duplication factor when balancing train classes")
    ap.add_argument("--name", default="type_v1")
    ap.add_argument("--skip-train", action="store_true", help="reuse runs/classify/<name> weights")
    args = ap.parse_args()

    crops_dir, out = Path(args.crops), Path(args.out)
    rows = parse_crops(crops_dir)
    if not rows:
        sys.exit(f"no crops in {crops_dir} (run track.py first)")
    labels6 = sorted({r["label"] for r in rows})
    labels4 = sorted(set(CHEM_GROUP.values()))

    ocr = {rec["crop"]: rec for rec in json.loads(Path(args.ocr).read_text(encoding="utf-8"))}
    n_matched = sum(1 for r in rows if r["crop"] in ocr)
    print(f"{len(rows)} crops, {len(labels6)} classes; OCR records matched: {n_matched}/{len(rows)}")

    train_keys, test_keys = split_by_run(rows, args.test_frac, args.seed)
    train = [r for r in rows if (r["label"], r["run"]) in train_keys]
    test = [r for r in rows if (r["label"], r["run"]) in test_keys]
    tr_c, te_c = Counter(r["label"] for r in train), Counter(r["label"] for r in test)
    print(f"split by run: train {len(train)} crops / {len(train_keys)} runs, "
          f"test {len(test)} crops / {len(test_keys)} runs")
    for c in labels6:
        print(f"  {c:<16} train {tr_c[c]:3d}  test {te_c[c]:3d}")
    if 0 in tr_c.values() or 0 in te_c.values():
        sys.exit("a class is missing from train or test — adjust --test-frac/--seed")

    # ---- materialize the ultralytics classify dataset (train oversampled, val==train monitor)
    ds = out / "dataset"
    if ds.exists():
        shutil.rmtree(ds)
    biggest = max(tr_c.values())
    for r in train:
        n_copies = min(args.oversample_cap, max(1, round(biggest / tr_c[r["label"]])))
        for k in range(n_copies):
            dst = ds / "train" / r["label"] / (f"{k}__{r['crop']}" if k else r["crop"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r["path"], dst)
        val_dst = ds / "val" / r["label"] / r["crop"]  # single copy: val is only a fit monitor
        val_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["path"], val_dst)
    n_train_files = sum(1 for _ in (ds / "train").rglob("*.jpg"))
    print(f"dataset at {ds} ({n_train_files} train files after oversampling, cap x{args.oversample_cap})")

    from ultralytics import YOLO

    project = repo / "runs" / "classify"
    weights_dir = project / args.name / "weights"
    if not args.skip_train:
        model = YOLO("yolo11n-cls.pt")
        model.train(
            data=str(ds), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=args.workers, project=str(project), name=args.name,
            exist_ok=True, seed=args.seed, patience=args.epochs,  # fixed budget, no early stop
            cos_lr=True, fliplr=0.5, flipud=0.5, degrees=0.0, erasing=0.0,
        )
    if not (weights_dir / "best.pt").exists():
        sys.exit(f"no weights at {weights_dir} (train first / drop --skip-train)")

    # ---- ONE honest pass over the held-out runs, for both checkpoints
    def predict_probs(weights: Path, batch_rows: list[dict]) -> np.ndarray:
        m = YOLO(str(weights))
        idx = {v: k for k, v in m.names.items()}
        order = [idx[c] for c in labels6]  # model-index -> our sorted label order
        probs = np.zeros((len(batch_rows), len(labels6)))
        for i0 in range(0, len(batch_rows), 64):
            chunk = batch_rows[i0:i0 + 64]
            res = m.predict([str(r["path"]) for r in chunk], imgsz=args.imgsz,
                            device=args.device, verbose=False)
            for j, rr in enumerate(res):
                probs[i0 + j] = rr.probs.data.cpu().numpy()[order]
        return probs

    y_test = [r["label"] for r in test]
    metrics: dict = {"split": {"train_crops": len(train), "test_crops": len(test),
                               "train_runs": len(train_keys), "test_runs": len(test_keys),
                               "per_class": {c: {"train": tr_c[c], "test": te_c[c]} for c in labels6}},
                     "models": {}}
    print("\n=== held-out test (grouped by run) ===")
    probs_by_ckpt = {}
    for ckpt in ("best", "last"):
        probs = predict_probs(weights_dir / f"{ckpt}.pt", test)
        probs_by_ckpt[ckpt] = probs
        pred = [labels6[i] for i in probs.argmax(1)]
        metrics["models"][f"visual_{ckpt}_6way"] = report(f"visual {ckpt}.pt (6-way)", y_test, pred, labels6)
        y4, p4 = [CHEM_GROUP[c] for c in y_test], [CHEM_GROUP[c] for c in pred]
        metrics["models"][f"visual_{ckpt}_4way"] = report(f"visual {ckpt}.pt (4-way chem)", y4, p4, labels4)

    # ---- ablation: trusted-OCR-only LR, and late fusion (visual probs + OCR feats)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_ocr_tr = np.array([ocr_feature_row(ocr.get(r["crop"])) for r in train])
    X_ocr_te = np.array([ocr_feature_row(ocr.get(r["crop"])) for r in test])
    y_train = [r["label"] for r in train]

    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(X_ocr_tr, y_train)
    pred_ocr = list(lr.predict(X_ocr_te))
    metrics["models"]["ocr_only_6way"] = report("OCR-features-only LR (6-way)", y_test, pred_ocr, labels6)
    metrics["models"]["ocr_only_4way"] = report(
        "OCR-features-only LR (4-way)", [CHEM_GROUP[c] for c in y_test],
        [CHEM_GROUP[c] for c in pred_ocr], labels4)

    probs_tr = predict_probs(weights_dir / "best.pt", train)
    fus = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    fus.fit(np.hstack([probs_tr, X_ocr_tr]), y_train)
    pred_fus = list(fus.predict(np.hstack([probs_by_ckpt["best"], X_ocr_te])))
    metrics["models"]["fusion_6way"] = report("fusion best.pt+OCR LR (6-way)", y_test, pred_fus, labels6)
    metrics["models"]["fusion_4way"] = report(
        "fusion best.pt+OCR LR (4-way)", [CHEM_GROUP[c] for c in y_test],
        [CHEM_GROUP[c] for c in pred_fus], labels4)

    # ---- artifacts
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    import csv
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crop", "run", "true", "pred_visual_best", "prob_visual_best", "pred_ocr", "pred_fusion"])
        pv = probs_by_ckpt["best"]
        for i, r in enumerate(test):
            w.writerow([r["crop"], r["run"], r["label"], labels6[pv[i].argmax()],
                        f"{pv[i].max():.3f}", pred_ocr[i], pred_fus[i]])
    print(f"\nmetrics -> {out / 'metrics.json'}\npredictions -> {out / 'predictions.csv'}")
    print(f"weights -> {weights_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
