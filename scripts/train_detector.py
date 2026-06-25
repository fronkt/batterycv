"""Train a single-class (battery) YOLO11 detector on the SAM pseudo-labels.

Run on the GPU box. imgsz is large (batteries are small in 1280x1024 frames).

    python scripts/train_detector.py --model yolo11s.pt --epochs 80 --imgsz 1024 --batch 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--data", default=None, help="override path to data.yaml")
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = load_paths()
    data = Path(args.data) if args.data else paths["yolo_dataset"] / "data.yaml"
    if not data.exists():
        sys.exit(f"data.yaml not found: {data} (run pseudo_label_sam.py first)")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/detect",
        name="battery_yolo11",
        seed=0,
        patience=20,
        # conveyor frames are top-down: vertical/horizontal flips are valid; no big rotations
        fliplr=0.5, flipud=0.5, degrees=0.0, mosaic=1.0,
    )
    print("training done -> runs/detect/battery_yolo11/weights/best.pt")


if __name__ == "__main__":
    main()
