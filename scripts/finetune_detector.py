"""Fine-tune the YOLO-World-bootstrapped detector on the hand-labeled pool.

The zero-shot labelers (SAM, YOLO-World) cap at ~0.45 recall because they only box the bright
printed label on a dark battery body that blends into the dark belt (docs/recall_ceiling_findings.md).
The only lever left in software is whole-object human supervision on THIS imagery. This continues
training from the best zero-shot detector (battery_yw_s1280) on the label_pool — frames hand-fixed
via label_assisted.py, balanced across all 6 classes (so the easy classes stay represented and the
model doesn't forget them while it learns the dark/small cells).

Low lr0 so the strong init isn't wiped by a tiny set. The 72 hand-verified eval frames are NOT in
the pool (build_label_pool.py excludes them) and stay the honest held-out test — eval after with
    python scripts/eval_detection.py --weights runs/detect/<name>/weights/best.pt

    python scripts/finetune_detector.py                      # defaults: s1280 init, pool, 80 ep
    python scripts/finetune_detector.py --epochs 120 --lr0 0.0005 --freeze 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths


def main() -> None:
    paths = load_paths()
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default=str(repo / "runs/detect/battery_yw_s1280/weights/best.pt"),
                    help="weights to continue from (the best zero-shot detector)")
    ap.add_argument("--pool", default=str(paths["data_root"] / "label_pool"),
                    help="hand-labeled pool dir with images/ and labels/")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr0", type=float, default=0.001,
                    help="low — the init is already good; high lr0 would wipe it on a tiny set")
    ap.add_argument("--optimizer", default="AdamW",
                    help="explicit, NOT 'auto' — auto ignores lr0 and picks its own (2x too high here)")
    ap.add_argument("--freeze", type=int, default=0,
                    help="freeze first N layers (0 = full fine-tune; 10 = keep backbone stem)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-val", dest="val", action="store_false",
                    help="skip per-epoch validation (val==train here is just an overfit monitor; "
                         "halves CPU time — the honest test is eval_detection.py on the 72 frames)")
    ap.set_defaults(val=True)
    ap.add_argument("--name", default="battery_ft1")
    args = ap.parse_args()

    pool = Path(args.pool)
    img_dir = pool / "images"
    lbl_dir = pool / "labels"
    n_img = len(list(img_dir.glob("*.jpg")))
    n_lbl = len(list(lbl_dir.glob("*.txt")))
    if n_img == 0:
        sys.exit(f"no images in {img_dir} (build_label_pool.py + label_assisted.py first)")
    if n_lbl == 0:
        sys.exit(f"no labels in {lbl_dir} (label them with label_assisted.py first)")
    if not Path(args.init).exists():
        sys.exit(f"init weights not found: {args.init}")

    # data.yaml: train == val == the pool. The pool is the gold signal; with so few frames we
    # don't carve a separate val — the 72 hand eval frames (excluded from the pool) are the honest
    # test. val==train here only monitors fit; best.pt is re-checked externally on the eval set.
    data_yaml = pool / "data_finetune.yaml"
    data_yaml.write_text(
        f"path: {pool.as_posix()}\n"
        f"train: images\nval: images\nnames:\n  0: battery\n",
        encoding="utf-8",
    )
    print(f"fine-tune: {n_img} pool frames ({n_lbl} labeled) from {Path(args.init).name} "
          f"lr0={args.lr0} freeze={args.freeze} epochs={args.epochs} imgsz={args.imgsz}")

    from ultralytics import YOLO

    project = repo / "runs" / "detect"
    model = YOLO(args.init)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        exist_ok=True,
        seed=0,
        optimizer=args.optimizer,       # explicit so lr0 is honored (auto overrides it)
        lr0=args.lr0,
        cos_lr=True,                    # smooth decay of the already-low lr
        val=args.val,
        patience=args.epochs,           # tiny set: run full, judge on the held-out 72
        freeze=args.freeze,
        fliplr=0.5, flipud=0.5, degrees=0.0, mosaic=1.0,
    )
    print(f"\nfine-tune done -> {project / args.name / 'weights' / 'best.pt'}")
    print(f"honest eval:  python scripts/eval_detection.py "
          f"--weights runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
