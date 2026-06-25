"""Minimal OpenCV labeler for the hand-verified eval set (no external tools needed).

Draw battery boxes with the mouse; writes YOLO-format txt (class 0 = battery) next to each
image's stem under <eval_dir>/labels/. Resumable: already-labeled frames load their boxes.

    python scripts/label_eval.py

Controls:
  left-drag        draw a box
  u                undo last box
  c                clear all boxes on this frame
  n / SPACE / ->   save this frame's labels and go to next
  p / <-           save and go to previous
  s                save without moving
  q / ESC          save current and quit
A frame with zero boxes still writes an empty .txt (a valid 'no batteries' label).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths

WIN = "batterycv labeler  [drag=box  u=undo  c=clear  n/space=next  p=prev  q=quit]"


def yolo_path(labels_dir: Path, img: Path) -> Path:
    return labels_dir / (img.stem + ".txt")


def load_boxes(p: Path, w: int, h: int):
    boxes = []
    if p.exists():
        for ln in p.read_text().splitlines():
            parts = ln.split()
            if len(parts) == 5:
                _, cx, cy, bw, bh = (float(x) for x in parts)
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                boxes.append([x1, y1, x2, y2])
    return boxes


def save_boxes(p: Path, boxes, w: int, h: int) -> None:
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
        cx = (x1 + x2) / 2 / w; cy = (y1 + y2) / 2 / h
        bw = abs(x2 - x1) / w; bh = abs(y2 - y1) / h
        if bw > 0.002 and bh > 0.002:
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    p.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = load_paths()
    img_dir = paths["eval_dir"] / "images"
    labels_dir = paths["eval_dir"] / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        sys.exit(f"no eval images in {img_dir} (run make_val_split.py first)")

    state = {"drawing": False, "p0": None, "cur": None}
    boxes: list[list[int]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True; state["p0"] = (x, y); state["cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            x0, y0 = state["p0"]
            boxes.append([x0, y0, x, y])

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)

    i = 0
    while 0 <= i < len(images):
        img_path = images[i]
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        boxes[:] = load_boxes(yolo_path(labels_dir, img_path), w, h)

        while True:
            disp = img.copy()
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if state["drawing"] and state["p0"] and state["cur"]:
                cv2.rectangle(disp, state["p0"], state["cur"], (0, 200, 255), 1)
            cv2.putText(disp, f"{i+1}/{len(images)}  {img_path.name}  boxes={len(boxes)}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WIN, disp)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("n"), ord(" "), 83):           # next
                save_boxes(yolo_path(labels_dir, img_path), boxes, w, h); i += 1; break
            if k in (ord("p"), 81):                        # prev
                save_boxes(yolo_path(labels_dir, img_path), boxes, w, h); i -= 1; break
            if k == ord("s"):                              # save in place
                save_boxes(yolo_path(labels_dir, img_path), boxes, w, h)
            if k == ord("u") and boxes:                    # undo
                boxes.pop()
            if k == ord("c"):                              # clear
                boxes.clear()
            if k in (ord("q"), 27):                        # quit
                save_boxes(yolo_path(labels_dir, img_path), boxes, w, h)
                cv2.destroyAllWindows()
                done = len(list(labels_dir.glob("*.txt")))
                print(f"saved. labeled {done}/{len(images)} frames -> {labels_dir}")
                return
    cv2.destroyAllWindows()
    done = len(list(labels_dir.glob("*.txt")))
    print(f"done. labeled {done}/{len(images)} frames -> {labels_dir}")


if __name__ == "__main__":
    main()
