"""Assisted OpenCV labeler: the trained detector PRE-FILLS each frame's boxes; you just fix them.

For the fine-tune workflow. Instead of drawing every battery from scratch, the detector's
predictions load as editable boxes — delete the false positives (right-click), drag the few it
missed, nudge nothing else. ~5-10x faster than blank labeling for 150-300 frames.

Resumable: a frame with a saved label loads that (your confirmed work); an unlabeled frame is
pre-filled fresh from the detector. Images are expected already CLAHE-normalized (build by
build_label_pool.py) so they match training and what the detector sees.

    python scripts/label_assisted.py            # defaults to the label_pool + best s1280 weights

Controls:
  left-drag        draw a box (a battery the detector missed)
  right-click      delete the box under the cursor (a false positive)
  u                undo last drawn box        c   clear all boxes
  r                re-run detector on this frame (discard edits, re-pre-fill)
  n / SPACE / ->   save + next                p / <-   save + previous
  s                save in place              q / ESC  save + quit
A frame saved with zero boxes writes an empty .txt (valid 'no batteries' label).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths

WIN = "batterycv assisted labeler  [drag=add  R-click=delete  u=undo  r=re-predict  n=next  q=quit]"


def yolo_path(labels_dir: Path, img: Path) -> Path:
    return labels_dir / (img.stem + ".txt")


def load_boxes(p: Path, w: int, h: int):
    boxes = []
    if p.exists():
        for ln in p.read_text().splitlines():
            parts = ln.split()
            if len(parts) == 5:
                _, cx, cy, bw, bh = (float(x) for x in parts)
                boxes.append([int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                              int((cx + bw / 2) * w), int((cy + bh / 2) * h)])
    return boxes


def save_boxes(p: Path, boxes, w: int, h: int) -> None:
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
        bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
        if bw > 0.002 and bh > 0.002:
            lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {bw:.6f} {bh:.6f}")
    p.write_text("\n".join(lines), encoding="utf-8")


def predict_boxes(model, img, conf, imgsz):
    r = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
    if r.boxes is None or not len(r.boxes):
        return []
    return [[int(a), int(b), int(c), int(d)] for a, b, c, d in r.boxes.xyxy.cpu().numpy()]


def box_under(boxes, x, y):
    """Index of the smallest box containing (x,y), else nearest center within 40 px, else None."""
    contain = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        lo_x, hi_x = sorted((x1, x2)); lo_y, hi_y = sorted((y1, y2))
        if lo_x <= x <= hi_x and lo_y <= y <= hi_y:
            contain.append((abs(hi_x - lo_x) * abs(hi_y - lo_y), i))
    if contain:
        return min(contain)[1]
    best, bd = None, 40 ** 2
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        d = ((x1 + x2) / 2 - x) ** 2 + ((y1 + y2) / 2 - y) ** 2
        if d < bd:
            best, bd = i, d
    return best


def main() -> None:
    paths = load_paths()
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default=str(paths["data_root"] / "label_pool" / "images"))
    ap.add_argument("--labels", default=str(paths["data_root"] / "label_pool" / "labels"))
    ap.add_argument("--weights",
                    default=str(repo / "runs/detect/battery_yw_s1280/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    img_dir = Path(args.images)
    labels_dir = Path(args.labels)
    labels_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        sys.exit(f"no images in {img_dir} (run build_label_pool.py first)")
    if not Path(args.weights).exists():
        sys.exit(f"weights not found: {args.weights}")

    from ultralytics import YOLO
    print(f"loading detector {args.weights} ...")
    model = YOLO(args.weights)

    state = {"drawing": False, "p0": None, "cur": None, "del": None}
    boxes: list[list[int]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True; state["p0"] = (x, y); state["cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            x0, y0 = state["p0"]
            if abs(x - x0) > 3 and abs(y - y0) > 3:
                boxes.append([x0, y0, x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            j = box_under(boxes, x, y)
            if j is not None:
                boxes.pop(j)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)

    i = 0
    while 0 <= i < len(images):
        img_path = images[i]
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lp = yolo_path(labels_dir, img_path)
        prefilled = not lp.exists()
        boxes[:] = predict_boxes(model, img, args.conf, args.imgsz) if prefilled \
            else load_boxes(lp, w, h)

        while True:
            disp = img.copy()
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if state["drawing"] and state["p0"] and state["cur"]:
                cv2.rectangle(disp, state["p0"], state["cur"], (0, 200, 255), 1)
            tag = "PRE-FILLED (detector)" if prefilled else "saved"
            cv2.putText(disp, f"{i+1}/{len(images)}  {img_path.name}  boxes={len(boxes)}  [{tag}]",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WIN, disp)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("n"), ord(" "), 83):
                save_boxes(lp, boxes, w, h); i += 1; break
            if k in (ord("p"), 81):
                save_boxes(lp, boxes, w, h); i -= 1; break
            if k == ord("s"):
                save_boxes(lp, boxes, w, h); prefilled = False
            if k == ord("u") and boxes:
                boxes.pop()
            if k == ord("c"):
                boxes.clear()
            if k == ord("r"):                              # re-run detector, discard edits
                boxes[:] = predict_boxes(model, img, args.conf, args.imgsz); prefilled = True
            if k in (ord("q"), 27):
                save_boxes(lp, boxes, w, h)
                cv2.destroyAllWindows()
                done = len(list(labels_dir.glob("*.txt")))
                print(f"saved. {done}/{len(images)} frames labeled -> {labels_dir}")
                return
    cv2.destroyAllWindows()
    done = len(list(labels_dir.glob("*.txt")))
    print(f"done. {done}/{len(images)} frames labeled -> {labels_dir}")


if __name__ == "__main__":
    main()
