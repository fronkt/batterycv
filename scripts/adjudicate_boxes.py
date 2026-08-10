"""Blind A/B over every disputed box: was the ~0.45 recall ceiling the detector, or the labels?

Relabeling cannot answer this. A label set built by accepting or rejecting detector pre-fills
scores that detector near 1.0 by construction -- every box in it came from the detector, so
"recall" degenerates into "how many of its own boxes did we keep". Two full relabel passes over
the 72 eval frames produced exactly that (recall 1.000 on all six classes, zero residual misses).

The question that actually needs answering is narrower and is a JUDGEMENT, not a drawing task:

    for each GT box the detector failed to match at IoU>=0.5,
    is the GT box the better box, or is the detector's box the better box?

So this shows you one disputed object at a time, zoomed, with both boxes drawn -- in CYAN and
MAGENTA, **randomly assigned per case**, so you cannot tell which is the detector's. You press one
key. That is the same design as the 34/36 blind audit in recall_ceiling_round2.md, extended from
a 36-panel sample to the whole disputed population, with you as the rater.

Cases where the detector produced no box at all are asked differently: is there a battery here at
all? Those are the only genuine detection failures, and the only evidence that could support a
lighting or hardware change.

    python scripts/adjudicate_boxes.py            # resumable; q saves and quits
    python scripts/adjudicate_boxes.py --report   # scores what you have judged so far, no GUI

Keys:  1 = the CYAN box is the better box      2 = the MAGENTA box is the better box
       3 / n = neither -- this is not a battery, or both boxes are wrong
       y = (single-box cases) yes, there is a battery here      n / 3 = no, there is not
       p = back one case      s = forward one case      g = jump to next unjudged
       u = undo the last verdict and return to that case        q = save + quit
Navigation covers every case, judged or not, so you can revisit and change an answer -- the
current answer is shown on screen and any key overwrites it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.evalutil import CLASSES, class_of, iou_mat, load_gt

WIN = "adjudicate  [1=cyan better  2=magenta better  3=neither  y/n  u=undo  s=skip  q=quit]"
CYAN, MAGENTA = (255, 255, 0), (255, 0, 255)
MATCH_IOU = 0.5
# Below this the detector's nearest box is around a DIFFERENT object, so "which box is better"
# has no answer -- it becomes the single-box question instead. 0.1 is the same threshold the
# miss taxonomy in analyze_misses.py uses to separate a near miss from a total miss.
NEAR_IOU = 0.1
PAD = 90
MIN_VIEW = 760


def build_cases(gt: dict, preds: dict, seed: int) -> list[dict]:
    """Every GT box unmatched at IoU>=MATCH_IOU, paired with the detector's best overlapping box.

    Colour assignment is randomised per case from a fixed seed, so the pass is blind but exactly
    reproducible -- a verdict file can be re-scored later without re-showing anything.
    """
    rng = random.Random(seed)
    cases = []
    for stem in sorted(gt):
        g, p = gt[stem], preds.get(stem, np.zeros((0, 4)))
        m = iou_mat(g, p)
        for j in range(len(g)):
            best_iou = float(m[j].max()) if m.size else 0.0
            if best_iou >= MATCH_IOU:
                continue
            det = p[int(m[j].argmax())].tolist() if best_iou >= NEAR_IOU else None
            cases.append({"id": f"{stem}#{j}", "stem": stem, "cls": class_of(stem),
                          "gt": g[j].tolist(), "det": det, "best_iou": best_iou,
                          "gt_is_cyan": rng.random() < 0.5})
    return cases


def render(img, case) -> np.ndarray:
    boxes = [case["gt"]] + ([case["det"]] if case["det"] else [])
    xs = [c for b in boxes for c in (b[0], b[2])]
    ys = [c for b in boxes for c in (b[1], b[3])]
    x1, y1 = max(0, int(min(xs)) - PAD), max(0, int(min(ys)) - PAD)
    x2, y2 = min(img.shape[1], int(max(xs)) + PAD), min(img.shape[0], int(max(ys)) + PAD)
    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return np.zeros((200, 400, 3), np.uint8)
    gt_c = CYAN if case["gt_is_cyan"] else MAGENTA
    det_c = MAGENTA if case["gt_is_cyan"] else CYAN
    cv2.rectangle(crop, (int(case["gt"][0]) - x1, int(case["gt"][1]) - y1),
                  (int(case["gt"][2]) - x1, int(case["gt"][3]) - y1), gt_c, 2)
    if case["det"]:
        cv2.rectangle(crop, (int(case["det"][0]) - x1, int(case["det"][1]) - y1),
                      (int(case["det"][2]) - x1, int(case["det"][3]) - y1), det_c, 2)
    s = max(1.0, MIN_VIEW / max(crop.shape[1], 1))
    return cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def score(cases: list[dict], verdicts: dict, n_gt_total: int, n_matched: int) -> dict:
    """Turn verdicts into the corrected recall, and say what each bucket means.

    'not a battery' cases leave the denominator -- a box that should never have been labelled is
    not a detection failure. Everything unjudged stays counted as a miss, so a partial pass can
    only ever UNDER-state the correction; it never flatters the detector.
    """
    out = {k: 0 for k in ("det_better", "gt_better", "not_a_battery", "real_miss",
                          "phantom_miss", "unjudged")}
    per_cls = {c: dict(recovered=0, disputed=0, dropped=0) for c in CLASSES}
    for c in cases:
        v = verdicts.get(c["id"])
        per_cls[c["cls"]]["disputed"] += 1
        if v is None:
            out["unjudged"] += 1
        elif v == "det_better":
            out["det_better"] += 1
            per_cls[c["cls"]]["recovered"] += 1
        elif v in ("not_a_battery", "phantom_miss"):
            out[v] += 1
            per_cls[c["cls"]]["dropped"] += 1
        else:
            out[v] = out.get(v, 0) + 1
    dropped = out["not_a_battery"] + out["phantom_miss"]
    denom = n_gt_total - dropped
    return {"buckets": out, "per_class": per_cls, "n_gt_total": n_gt_total,
            "n_matched": n_matched, "dropped_from_denominator": dropped,
            "recall_as_published": n_matched / n_gt_total if n_gt_total else float("nan"),
            "recall_corrected": (n_matched + out["det_better"]) / denom if denom else float("nan")}


def print_report(s: dict) -> None:
    b = s["buckets"]
    print(f"\n=== adjudicated {sum(b.values()) - b['unjudged']} of "
          f"{sum(b.values())} disputed boxes ===")
    print(f"  detector's box was the better box : {b['det_better']:>4}  -> recovered")
    print(f"  GT box was the better box         : {b['gt_better']:>4}  -> real localisation miss")
    print(f"  not a battery / both wrong        : {b['not_a_battery']:>4}  -> left denominator")
    print(f"  no detector box, battery IS there : {b['real_miss']:>4}  -> genuine detection miss")
    print(f"  no detector box, nothing there    : {b['phantom_miss']:>4}  -> left denominator")
    print(f"  not yet judged                    : {b['unjudged']:>4}  -> counted as missed")
    print(f"\n  recall as published : {s['recall_as_published']:.3f}"
          f"   ({s['n_matched']}/{s['n_gt_total']})")
    print(f"  recall corrected    : {s['recall_corrected']:.3f}"
          f"   ({s['n_matched'] + b['det_better']}/"
          f"{s['n_gt_total'] - s['dropped_from_denominator']})")
    if b["unjudged"]:
        print("  (unjudged boxes count as missed, so the corrected figure is a LOWER bound)")
    print(f"\n{'class':<16}{'disputed':>10}{'recovered':>11}{'dropped':>9}")
    for c, d in s["per_class"].items():
        if d["disputed"]:
            print(f"{c:<16}{d['disputed']:>10}{d['recovered']:>11}{d['dropped']:>9}")
    print(f"\n  'genuine detection miss' is the ONLY bucket that could justify a lighting or")
    print(f"  hardware change. Look at each one before making that argument.")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    paths = load_paths()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(paths["eval_dir"] / "labels"),
                    help="the GT under test -- the ORIGINAL labels, not a detector-prefilled v2")
    ap.add_argument("--weights", default=str(repo / "runs/detect/battery_ft1/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--report", action="store_true", help="score existing verdicts, no GUI")
    ap.add_argument("--crops", action="store_true",
                    help="write a padded crop of every case judged a genuine detection miss. "
                         "That bucket is the entire evidence base for a lighting/hardware ask, "
                         "so it should be looked at, not just counted.")
    ap.add_argument("--out", default=str(repo / "results/phase4/adjudication.json"))
    args = ap.parse_args()

    img_dir = paths["eval_dir"] / "images"
    gt = load_gt(args.labels)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    saved = json.loads(outp.read_text()) if outp.exists() else {}
    verdicts: dict[str, str] = saved.get("verdicts", {})

    from ultralytics import YOLO
    model = YOLO(args.weights)
    preds = {}
    for stem in sorted(gt):
        r = model.predict(str(img_dir / f"{stem}.jpg"), conf=args.conf, imgsz=args.imgsz,
                          verbose=False)[0]
        preds[stem] = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))

    cases = build_cases(gt, preds, args.seed)
    n_gt_total = sum(len(v) for v in gt.values())
    n_matched = n_gt_total - len(cases)
    print(f"{n_gt_total} GT boxes, {n_matched} already matched at IoU>={MATCH_IOU}, "
          f"{len(cases)} disputed")

    def persist():
        outp.write_text(json.dumps({
            "verdicts": verdicts, "seed": args.seed, "match_iou": MATCH_IOU,
            "score": score(cases, verdicts, n_gt_total, n_matched),
            "cases": [{k: v for k, v in c.items() if k != "gt_is_cyan"} for c in cases],
        }, indent=2), encoding="utf-8")

    if args.crops:
        cdir = outp.parent / "genuine_misses"
        cdir.mkdir(parents=True, exist_ok=True)
        n = 0
        for c in cases:
            if verdicts.get(c["id"]) != "real_miss":
                continue
            img = cv2.imread(str(img_dir / f"{c['stem']}.jpg"))
            x1, y1, x2, y2 = (int(t) for t in c["gt"])
            cx1, cy1 = max(0, x1 - PAD), max(0, y1 - PAD)
            crop = img[cy1:min(img.shape[0], y2 + PAD), cx1:min(img.shape[1], x2 + PAD)].copy()
            if crop.size:
                cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 255, 255), 2)
                cv2.imwrite(str(cdir / f"{n:02d}_{c['cls']}_{c['id'].split('#')[1]}.jpg"), crop)
                n += 1
        print(f"wrote {n} genuine-miss crops -> {cdir}")
        return

    if args.report:
        print_report(score(cases, verdicts, n_gt_total, n_matched))
        persist()
        print(f"\nwrote {outp}")
        return

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    # Walk EVERY case, not just the unjudged ones, so 'p' can reach a case you already answered
    # and change it. Start on the first unanswered one; a case that has a verdict shows it.
    order = cases
    history: list[str] = []
    i = next((n for n, c in enumerate(order) if c["id"] not in verdicts), 0)
    while 0 <= i < len(order):
        case = order[i]
        img = cv2.imread(str(img_dir / f"{case['stem']}.jpg"))
        view = render(img, case)
        single = case["det"] is None
        head = (f"{i+1}/{len(order)}  {case['cls']}   ONE BOX ONLY (detector found nothing "
                f"here)  -  is there a battery?   y=yes   n/3=no") if single else \
               (f"{i+1}/{len(order)}  {case['cls']}   IoU {case['best_iou']:.2f}  -  which box "
                f"is better?   1=cyan   2=magenta   3/n=neither")
        nav = f"p=back  s=forward  g=next unjudged  u=undo  q=quit   [{len(verdicts)}/{len(order)} judged]"
        hint = ""
        while True:
            disp = view.copy()
            prior = verdicts.get(case["id"])
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 58), (0, 0, 0), -1)
            cv2.putText(disp, head, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(disp, nav, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 170, 170), 1,
                        cv2.LINE_AA)
            if prior:
                cv2.rectangle(disp, (0, 58), (disp.shape[1], 86), (0, 0, 0), -1)
                cv2.putText(disp, f"already answered: {prior}  -  press a key to change it",
                            (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            if hint:
                y0 = 86 if prior else 58
                cv2.rectangle(disp, (0, y0), (disp.shape[1], y0 + 28), (0, 0, 0), -1)
                cv2.putText(disp, hint, (8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 165, 255), 1, cv2.LINE_AA)
            cv2.imshow(WIN, disp)
            k = cv2.waitKey(20) & 0xFF
            v = None
            if single and k == ord("y"):
                v = "real_miss"
            elif single and k in (ord("n"), ord("3")):
                v = "phantom_miss"
            elif not single and k == ord("1"):
                v = "det_better" if not case["gt_is_cyan"] else "gt_better"
            elif not single and k == ord("2"):
                v = "det_better" if case["gt_is_cyan"] else "gt_better"
            elif not single and k in (ord("3"), ord("n")):
                v = "not_a_battery"
            elif k in (ord("y"), ord("1"), ord("2")):
                # A key that is valid on the OTHER kind of case. Say so instead of ignoring it --
                # a dead keypress reads as a frozen window.
                hint = ("this case has only one box -- press y or n"
                        if single else "this case has two boxes -- press 1, 2 or 3")
            elif k in (ord("s"), 83):                       # forward, leaving the answer alone
                i = min(i + 1, len(order) - 1)
                break
            elif k in (ord("p"), 81):                       # back, leaving the answer alone
                i = max(i - 1, 0)
                break
            elif k == ord("g"):                            # skip ahead to the next unanswered
                nxt = next((n for n in range(i + 1, len(order))
                            if order[n]["id"] not in verdicts), None)
                if nxt is None:
                    hint = "no unjudged case after this one"
                else:
                    i = nxt
                    break
            elif k == ord("u") and history:                # undo: erase the answer AND go to it
                last = history.pop()
                verdicts.pop(last, None)
                persist()
                i = next((n for n, c in enumerate(order) if c["id"] == last), max(i - 1, 0))
                break
            elif k in (ord("q"), 27):
                persist()
                cv2.destroyAllWindows()
                print_report(score(cases, verdicts, n_gt_total, n_matched))
                print(f"\nwrote {outp}")
                return
            if v:
                verdicts[case["id"]] = v
                history.append(case["id"])
                persist()
                i += 1
                break
    cv2.destroyAllWindows()
    persist()
    print_report(score(cases, verdicts, n_gt_total, n_matched))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
