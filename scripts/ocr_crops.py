"""OCR the per-battery crops from track.py — Phase-2 step.

Reads each best-conf battery crop (one per tracked battery, exported by track.py) and writes
per-battery text records that feed the Phase-3 type classifier (text-only vs text+image).

Two engines:
  - qwen  (default): Qwen2.5-VL, an open vision-language model run locally. It reads the dark,
    low-contrast, rotated labels holistically and returns STRUCTURED fields (manufacturer,
    chemistry, model/part #, voltage, capacity, certification marks) plus a free-text
    transcription. This is the engine the imagery needs.
  - easyocr (baseline): classical OCR. Documented to fail on this imagery — the text on the dark
    battery bodies is too small / low-contrast / dense (the same wall detection hit). Kept for
    comparison and as an offline-light fallback.

The crops are already CLAHE-normalized by track.py.

Outputs under <crops_dir>/../ocr/:
  - ocr.json    per-crop records (fields + raw_text for qwen; boxed strings for easyocr)
  - ocr.csv     flat summary: crop, track_id, label, manufacturer, chemistry, model, V, cap, marks
  - vis/*.jpg   crop with the extraction drawn alongside (sanity-check the reads)
  - prints a summary table

    python scripts/ocr_crops.py                                  # qwen, latest track run's crops
    python scripts/ocr_crops.py --engine easyocr
    python scripts/ocr_crops.py --crops <dir> --device cuda
    python scripts/ocr_crops.py --image one_crop.jpg --model Qwen/Qwen2.5-VL-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.preprocess import read_bgr

# crop filenames from track.py: "<label>__run<id>__id<tid>_c<conf>.jpg"
CROP_RE = re.compile(r"^(?P<label>.+?)__run(?P<run>\d+)__id(?P<tid>\d+)_c(?P<conf>[\d.]+)\.jpg$")

FIELDS = ["manufacturer", "chemistry", "model", "voltage", "capacity"]


def parse_crop_name(name: str) -> dict:
    m = CROP_RE.match(name)
    if not m:
        return {"label": "", "run_id": -1, "track_id": -1, "det_conf": 0.0}
    return {"label": m["label"], "run_id": int(m["run"]),
            "track_id": int(m["tid"]), "det_conf": float(m["conf"])}


def latest_crops_dir(paths) -> Path | None:
    """Most-recently-modified <work_dir>/track/*/crops (track.py's default output)."""
    cand = sorted((paths["work_dir"] / "track").glob("*/crops"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0] if cand else None


def upscale_to(bgr: np.ndarray, longest: int) -> np.ndarray:
    """Resize so the longest side is `longest` px (only upscales) — helps small label text."""
    h, w = bgr.shape[:2]
    s = longest / max(h, w)
    if s <= 1.0:
        return bgr
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)


# ----------------------------------------------------------------------------- Qwen2.5-VL engine
QWEN_PROMPT = (
    "This is a cropped photo of ONE battery on a recycling conveyor. The image may be dark, "
    "rotated, or low-contrast. Read the printed label. Transcribe only what you can actually "
    "read — do NOT guess or invent. Reply with ONLY a JSON object, no other text, with keys:\n"
    '  "manufacturer": brand if legible (e.g. "HP","DELL","Samsung","LG","Panasonic") else ""\n'
    '  "chemistry": cell chemistry if printed (e.g. "Li-ion","Li-polymer","Ni-MH","Ni-Cd") else ""\n'
    '  "model": model or part number from the label/barcode sticker, else ""\n'
    '  "voltage": rated voltage with unit (e.g. "11.55V"), else ""\n'
    '  "capacity": rated capacity (e.g. "41Wh" or "3600mAh"), else ""\n'
    '  "marks": list of certification/handling marks present (e.g. "CE","WEEE","UL","RoHS","recycle")\n'
    '  "raw_text": a short transcription of any other distinct legible words/numbers '
    '(space-separated; do NOT repeat the same word)\n'
)


def _extract_json(text: str) -> dict:
    """Pull the first {...} block out of the model reply and parse it; tolerate junk."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(text[a:b + 1])
        except json.JSONDecodeError:
            pass
    return {"raw_text": text}


_JUNK = {"", "unknown", "n/a", "na", "none", "null", "not visible", "not legible",
         "not specified", "not available", "illegible", "unclear", "no text"}


def _clean_val(v: str) -> str:
    """Small VLMs write 'unknown'/'N/A' for fields they can't read — treat those as empty."""
    v = str(v).strip().strip('"').strip()
    return "" if v.lower() in _JUNK else v


# distinctive fragments of our own prompt that the model sometimes echoes back
_PROMPT_MARKERS = ("recycling conveyor", "do not guess", "transcribe only", "low-contrast",
                   "json object", "manufacturer:", "the image may be", "one battery",
                   "cropped photo", "photo of", "printed label")
_KEYS = set(FIELDS) | {"marks", "raw_text"}        # JSON key-names to strip from raw_text


def _clean_raw(s: str) -> str:
    """Clean the raw_text catch-all: cut prompt-echo, drop JSON scaffolding / junk / field
    labels, collapse runaway repeats, cap length. Keeps the genuine reads (brands, part #s)."""
    low = s.lower()
    for mk in _PROMPT_MARKERS:                      # cut an echoed prompt and everything after
        i = low.find(mk)
        if i != -1:
            s, low = s[:i], low[:i]
    out = []
    for t in (t for t in re.split(r"[\s,]+", s) if t):
        clean = t.strip('"{}[]:,')
        tl = clean.lower()
        if not tl or tl in _JUNK or tl in _KEYS:
            continue
        if not out or out[-1].lower() != tl:
            out.append(clean)
    return " ".join(out[:40])


def parse_qwen_output(out: str) -> dict:
    """Pull structured fields from the reply, tolerating truncated / repetition-spammed JSON."""
    data = _extract_json(out)
    fields = {k: str(data.get(k, "") or "").strip() for k in FIELDS}
    marks = data.get("marks", []) or []
    raw = str(data.get("raw_text", "") or "").strip()
    # regex salvage for anything a truncated/unparseable JSON dropped
    for k in FIELDS:
        if not fields[k]:
            m = re.search(rf'"{k}"\s*:\s*"([^"]*)"', out)
            if m:
                fields[k] = m.group(1).strip()
    if not marks:
        mm = re.search(r'"marks"\s*:\s*\[([^\]]*)\]', out)
        if mm:
            marks = mm.group(1).split(",")
    if isinstance(marks, str):
        marks = re.split(r"[,/]", marks)
    if not raw:
        rm = re.search(r'"raw_text"\s*:\s*"([^"]*)"', out)
        if rm:
            raw = rm.group(1).strip()
    fields = {k: _clean_val(v) for k, v in fields.items()}
    marks = [m for m in (_clean_val(x) for x in marks) if m]
    return {"fields": fields, "marks": marks, "raw": _clean_raw(raw)}


class QwenEngine:
    name = "qwen"

    def __init__(self, model_id: str, device: str, max_new_tokens: int, longest: int):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.longest = longest
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        print(f"loading {model_id} on {device} ({dtype}) — first run downloads weights ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, device_map=device)
        self.model.eval()

    def read(self, bgr: np.ndarray) -> dict:
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        rgb = cv2.cvtColor(upscale_to(bgr, self.longest), cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil}, {"type": "text", "text": QWEN_PROMPT}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        vi = process_vision_info(messages)
        image_inputs, video_inputs = vi[0], vi[1]
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, repetition_penalty=1.05)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        parsed = parse_qwen_output(out)
        fields, marks, raw = parsed["fields"], parsed["marks"], parsed["raw"]
        joined = " ".join([v for v in fields.values() if v] + marks + ([raw] if raw else []))
        return {"engine": self.name, **fields, "marks": marks, "raw_text": raw,
                "joined": joined, "n_chars": sum(c.isalnum() for c in joined)}

    def draw(self, bgr: np.ndarray, rec: dict) -> np.ndarray:
        h, w = bgr.shape[:2]
        panel_w = max(360, w)
        panel = np.full((h, panel_w, 3), 30, np.uint8)
        lines = [f"#{rec['track_id']}  ({rec['label']})"]
        lines += [f"{k}: {rec[k]}" for k in FIELDS if rec.get(k)]
        if rec.get("marks"):
            lines.append("marks: " + ", ".join(rec["marks"]))
        if rec.get("raw_text"):
            words = rec["raw_text"].split()
            for i in range(0, len(words), 6):
                lines.append(("raw: " if i == 0 else "     ") + " ".join(words[i:i + 6]))
        y = 24
        for ln in lines:
            cv2.putText(panel, ln[:60], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 255, 200), 1, cv2.LINE_AA)
            y += 20
        return np.hstack([bgr, panel])


# --------------------------------------------------------------------------------- EasyOCR engine
def _meaningful(t: str) -> bool:
    t = t.strip()
    return len(t) >= 2 and any(c.isalnum() for c in t)


def label_rois(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the bright white sticker patches (serial/part/barcode labels)."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = max(200, float(np.percentile(gray, 98)))
    _, m = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    area = h * w
    out = []
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        if a < 0.0015 * area or a > 0.30 * area or bw < 14 or bh < 14:
            continue
        pad = 6
        out.append((int(max(0, x - pad)), int(max(0, y - pad)),
                    int(min(w, x + bw + pad)), int(min(h, y + bh + pad))))
    return out


class EasyOcrEngine:
    name = "easyocr"

    def __init__(self, langs, gpu, upscale, label_upscale, min_conf):
        import easyocr
        self.reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
        self.upscale, self.label_upscale, self.min_conf = upscale, label_upscale, min_conf

    def _ocr(self, img, up):
        big = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up != 1 else img
        out = []
        for box, text, conf in self.reader.readtext(big, detail=1, paragraph=False):
            if conf < self.min_conf or not _meaningful(text):
                continue
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            out.append(((int(min(xs) / up), int(min(ys) / up),
                         int(max(xs) / up), int(max(ys) / up)), text.strip(), float(conf)))
        return out

    def read(self, bgr):
        found = [{"text": t, "conf": c, "box": list(xy), "pass": "full"}
                 for xy, t, c in self._ocr(bgr, self.upscale)]
        for (rx1, ry1, rx2, ry2) in label_rois(bgr):
            roi = bgr[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                continue
            for (x1, y1, x2, y2), t, c in self._ocr(roi, self.label_upscale):
                found.append({"text": t, "conf": c,
                              "box": [x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1], "pass": "label"})
        best: dict[str, dict] = {}
        for it in found:
            k = it["text"].lower()
            if it["conf"] > best.get(k, {"conf": -1})["conf"]:
                best[k] = it
        items = sorted(best.values(), key=lambda it: (round(it["box"][1] / 20), it["box"][0]))
        joined = " ".join(it["text"] for it in items)
        return {"engine": self.name, "n_texts": len(items),
                "n_chars": sum(c.isalnum() for c in joined),
                "joined": joined, "texts": items,
                **{k: "" for k in FIELDS}, "marks": [], "raw_text": joined}

    def draw(self, bgr, rec):
        disp = bgr.copy()
        for it in rec["texts"]:
            x1, y1, x2, y2 = it["box"]
            col = (255, 255, 0) if it["pass"] == "label" else (0, 230, 0)
            cv2.rectangle(disp, (x1, y1), (x2, y2), col, 1)
            cv2.putText(disp, it["text"], (x1, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        return disp


def main() -> None:
    paths = load_paths()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["qwen", "easyocr"], default="qwen")
    ap.add_argument("--crops", default=None, help="crops dir (default: latest track run)")
    ap.add_argument("--image", default=None, help="OCR a single crop instead of a dir")
    ap.add_argument("--out", default=None, help="output dir (default <crops>/../ocr)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (qwen)")
    # 2B runs on a laptop CPU in ~80s/crop and reads brand/chemistry/marks well; pass a bigger
    # model (Qwen/Qwen2.5-VL-3B/7B-Instruct) with --device cuda for stronger fine-detail reads.
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct", help="qwen model id")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--longest", type=int, default=768, help="upscale crop's long side to this")
    # easyocr-only
    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument("--label-upscale", type=float, default=3.0)
    ap.add_argument("--min-conf", type=float, default=0.30)
    ap.add_argument("--langs", default="en")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    if args.image:
        crops = [Path(args.image)]
        out = Path(args.out) if args.out else Path(args.image).parent / "ocr"
    else:
        cdir = Path(args.crops) if args.crops else latest_crops_dir(paths)
        if cdir is None or not cdir.exists():
            sys.exit("no crops dir found — run track.py first, or pass --crops")
        crops = sorted(cdir.glob("*.jpg"))
        if not crops:
            sys.exit(f"no .jpg crops under {cdir}")
        out = Path(args.out) if args.out else cdir.parent / "ocr"
    vis_dir = out / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"OCR {len(crops)} crop(s) with {args.engine} -> {out}")

    if args.engine == "qwen":
        engine = QwenEngine(args.model, args.device, args.max_new_tokens, args.longest)
    else:
        engine = EasyOcrEngine(args.langs.split(","), args.gpu,
                               args.upscale, args.label_upscale, args.min_conf)

    records = []
    for p in crops:
        bgr = read_bgr(str(p))                          # already CLAHE-normalized by track.py
        rec = {**parse_crop_name(p.name), "crop": p.name, **engine.read(bgr)}
        records.append(rec)
        cv2.imwrite(str(vis_dir / p.name), engine.draw(bgr, rec))
        summary = (rec["joined"][:70] if args.engine == "easyocr"
                   else f"{rec['manufacturer']}|{rec['chemistry']}|{rec['model']}|"
                        f"{rec['voltage']}|{rec['capacity']}|{','.join(rec['marks'])}")
        print(f"  id{rec['track_id']:>3}: {rec['n_chars']:>3} chars  {summary}")

    (out / "ocr.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    hdr = "crop,track_id,label,manufacturer,chemistry,model,voltage,capacity,marks,n_chars"
    rows = [hdr] + [
        f'{r["crop"]},{r["track_id"]},{r["label"]},"{r["manufacturer"]}","{r["chemistry"]}",'
        f'"{r["model"]}","{r["voltage"]}","{r["capacity"]}","{";".join(r["marks"])}",{r["n_chars"]}'
        for r in records]
    (out / "ocr.csv").write_text("\n".join(rows), encoding="utf-8")

    n_with = sum(1 for r in records if r["n_chars"] > 0)
    print(f"\n--- OCR summary ({args.engine}) ---")
    print(f"  crops          : {len(records)}")
    print(f"  with text      : {n_with}")
    print(f"  total chars    : {sum(r['n_chars'] for r in records)}")
    print(f"  json / csv / vis: {out}")


if __name__ == "__main__":
    main()
