"""Evaluate the detector against the hand-verified eval set (mAP / recall / precision).

Recall is the priority metric here — missing batteries is worse than an extra box, since
downstream sorting must not drop items. Saves overlays + PR / confusion plots via Ultralytics.

    python scripts/eval_detection.py --weights runs/detect/battery_yolo11/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = load_paths()
    eval_images = paths["eval_dir"] / "images"
    eval_labels = paths["eval_dir"] / "labels"
    if not any(eval_labels.glob("*.txt")):
        sys.exit(f"no verified labels in {eval_labels} — label the val set first "
                 "(see make_val_split.py)")

    data = Path(args.data) if args.data else paths["yolo_dataset"] / "data_eval.yaml"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(
        f"path: {paths['eval_dir'].as_posix()}\n"
        f"train: images\nval: images\nnames:\n  0: battery\n",
        encoding="utf-8",
    )

    # absolute project dir so Ultralytics doesn't nest it under its own runs_dir
    eval_project = Path(__file__).resolve().parents[1] / "runs" / "eval"
    model = YOLO(args.weights)
    m = model.val(data=str(data), split="val", device=args.device,
                  project=str(eval_project), name="battery_eval", exist_ok=True,
                  plots=True, save_json=True)
    box = m.box
    print("\n--- detection metrics (hand-verified set) ---")
    print(f"  precision : {box.mp:.3f}")
    print(f"  recall    : {box.mr:.3f}   <-- priority")
    print(f"  mAP@0.5   : {box.map50:.3f}")
    print(f"  mAP@0.5:.95: {box.map:.3f}")
    print(f"\noverlays + curves -> runs/eval/battery_eval")


if __name__ == "__main__":
    main()
