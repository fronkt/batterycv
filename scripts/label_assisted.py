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


def stratify(images: list[Path]) -> list[Path]:
    """Round-robin over the class prefix (`ni_mh_all__cam_...jpg` -> `ni_mh_all`).

    Frames are named class-first, so plain sorted order exhausts one class before reaching the
    next: a 15-frame pilot would see only laptop cells, the class with the *best* recall, and
    would understate a correction driven by the worst ones. Interleaving makes any prefix of the
    pass representative. Resumability is unaffected — progress is keyed by filename, not index.
    """
    groups: dict[str, list[Path]] = {}
    for p in images:
        groups.setdefault(p.name.split("__")[0], []).append(p)
    out: list[Path] = []
    for k in range(max(len(v) for v in groups.values()) if groups else 0):
        out.extend(v[k] for v in groups.values() if k < len(v))
    return out


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


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])
    bx1, by1, bx2, by2 = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def unmatched(ref, boxes, thr=0.3):
    """Reference boxes with no editable box over `thr` IoU — i.e. objects about to be dropped."""
    return [r for r in ref if all(iou(r, b) < thr for b in boxes)]


def report(labels_dir: Path, images, seen: dict) -> None:
    """Say plainly how much of this pass was YOUR judgement vs the detector's.

    A frame saved byte-identical to the detector's pre-fill contributes nothing independent: a
    label set made only of those scores the detector at ~100% recall against itself, which looks
    like a triumph and means nothing. This prints the ratio while you can still act on it.
    """
    done = len(list(labels_dir.glob("*.txt")))
    touched = len(seen)
    untouched = sum(1 for v in seen.values() if v)
    print(f"\n{done}/{len(images)} frames labeled -> {labels_dir}")
    if touched:
        print(f"this session: {touched} frames saved, {untouched} of them EXACTLY as the "
              f"detector pre-filled them")
        if untouched >= 0.8 * touched:
            print("  ^ that is a circular label set. Recall measured against it is ~1.0 by\n"
                  "    construction. Re-run with --ref <old labels> so dropped objects are\n"
                  "    visible, and adopt or redraw them before trusting any number from it.")


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
    ap.add_argument("--order", choices=("name", "stratified"), default="name",
                    help="stratified: round-robin over the class prefix, so a partial pass "
                         "covers every class instead of exhausting one. Use for a pilot.")
    ap.add_argument("--ref", default=None,
                    help="A second label dir drawn as read-only RED reference (e.g. the old "
                         "labels when writing a v2). A red box with no green box over it is an "
                         "object you are about to drop -- the HUD counts them and 'a' adopts "
                         "them. Without this the tool cannot show you what is MISSING, only "
                         "what the detector found.")
    args = ap.parse_args()

    img_dir = Path(args.images)
    labels_dir = Path(args.labels)
    labels_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(img_dir.glob("*.jpg"))
    if args.order == "stratified":
        images = stratify(images)
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

    ref_dir = Path(args.ref) if args.ref else None
    seen: dict[str, bool] = {}          # stem -> was it saved exactly as pre-filled?

    i = 0
    while 0 <= i < len(images):
        img_path = images[i]
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lp = yolo_path(labels_dir, img_path)
        prefilled = not lp.exists()
        boxes[:] = predict_boxes(model, img, args.conf, args.imgsz) if prefilled \
            else load_boxes(lp, w, h)
        start = [list(b) for b in boxes]
        ref = load_boxes(yolo_path(ref_dir, img_path), w, h) if ref_dir else []

        while True:
            disp = img.copy()
            miss = unmatched(ref, boxes)
            for (x1, y1, x2, y2) in ref:                       # read-only reference
                dim = (0, 0, 255) if [x1, y1, x2, y2] in miss else (90, 90, 160)
                cv2.rectangle(disp, (x1, y1), (x2, y2), dim, 1, cv2.LINE_AA)
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if state["drawing"] and state["p0"] and state["cur"]:
                cv2.rectangle(disp, state["p0"], state["cur"], (0, 200, 255), 1)
            tag = "PRE-FILLED (detector)" if prefilled else "saved"
            cv2.putText(disp, f"{i+1}/{len(images)}  {img_path.name}  boxes={len(boxes)}  [{tag}]",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            if miss:
                cv2.putText(disp, f"{len(miss)} REFERENCE BOX(ES) UNCOVERED  -  'a' adopts, or "
                            f"draw/ignore deliberately", (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(WIN, disp)
            k = cv2.waitKey(20) & 0xFF
            if k == ord("a") and miss:
                boxes.extend([list(m) for m in miss])
            def commit():
                save_boxes(lp, boxes, w, h)
                seen[img_path.stem] = prefilled and boxes == start

            if k in (ord("n"), ord(" "), 83):
                commit(); i += 1; break
            if k in (ord("p"), 81):
                commit(); i -= 1; break
            if k == ord("s"):
                commit(); prefilled = False
            if k == ord("u") and boxes:
                boxes.pop()
            if k == ord("c"):
                boxes.clear()
            if k == ord("r"):                              # re-run detector, discard edits
                boxes[:] = predict_boxes(model, img, args.conf, args.imgsz); prefilled = True
            if k in (ord("q"), 27):
                commit()
                cv2.destroyAllWindows()
                report(labels_dir, images, seen)
                return
    cv2.destroyAllWindows()
    report(labels_dir, images, seen)


if __name__ == "__main__":
    main()
