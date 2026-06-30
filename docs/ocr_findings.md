# Phase 2 — reading the battery labels (OCR on tracked crops)

Goal: turn each tracked battery crop (one best-conf image per battery, exported by `track.py`)
into structured per-battery metadata — manufacturer, chemistry, model/part #, voltage, capacity,
certification marks — that feeds the Phase-3 type classifier. `scripts/ocr_crops.py` does this
with a pluggable engine (`--engine qwen|easyocr`). Demo set: the 6 batteries tracked over the
longest laptop run (`work/track/li_ion_laptop_run4/crops`).

## Finding 1 — classical OCR fails on this imagery (the same wall as detection)
EasyOCR, run two ways per crop (full-frame + bright sticker-ROI passes, with upscaling and
inversion), returned **4 garbage fragments across all 6 crops** (`Ihov`, `ce`, `IS`, a stray `0`).
The ROI finder correctly locates the regulatory text block, the cert symbols, and the white
barcode stickers — so this is a *recognition* failure, not localization. The text on the dark
battery bodies (~70/255) is too small / low-contrast / dense for classical OCR, exactly the
structural limit detection hit. Kept as `--engine easyocr` for the record.

## Finding 2 — an open VLM (Qwen2-VL) reads the structured signal
Switched the engine to **Qwen2-VL-2B-Instruct** (open, local, free — user's call). It reads the
label holistically and returns structured JSON; a regex-salvage parser tolerates the small model's
truncated/looping output, and `repetition_penalty=1.05` curbs the worst loops. Run on the 6 crops,
laptop CPU, ~80 s/crop:

| id | manufacturer | chemistry | voltage | capacity | marks | raw highlight |
|----|--------------|-----------|---------|----------|-------|---------------|
| 1  | —            | Li-ion    | 11.55V  | 41Wh     | CE WEEE UL RoHS | (11.55V/41Wh = real HP spec) |
| 3  | **DELL**     | Li-ion    | 9.0V*   | 8.0Ah*   | CE WEEE UL RoHS | DELL sticker confirmed in crop |
| 8  | —            | —         | —       | —        | CE WEEE | (side-on, dense — hard) |
| 11 | —            | —         | —       | —        | recycle | **HP 727897-001** (real part #) |
| 13 | —            | —         | —       | —        | UL FR-1681 CE WEEE RoHS recycle | |
| 17 | —            | —         | —       | —        | CE WEEE UL RoHS | |

\* likely hallucinated (odd values for a laptop pack).

## Honest assessment
- **Reliable:** certification / handling **marks** (all 6 crops), **chemistry** (Li-ion where
  legible), **brand** (DELL read correctly, confirmed against the visible sticker), and on the
  clearest sticker a **genuine part number** (`HP 727897-001`). None of this is reachable by
  classical OCR.
- **Lower-confidence:** fine specs (voltage/capacity) — some are real (id1 11.55V/41Wh matches HP),
  some are hallucinated (id3 9.0V/8.0Ah). A 2B model on dark imagery invents plausible numbers when
  it can't read them; the prompt says "do not guess" but it isn't fully obeyed. Treat specs as
  hints, not ground truth.
- **Still unread:** the dense multilingual regulatory micro-print — below the limit for a 2B VLM at
  this resolution.

Net: enough structured signal (brand / chemistry / marks, sometimes a part #) to drive a coarse
Phase-3 type classifier, and a categorical win over classical OCR. Precise spec extraction would
need a bigger model and/or better imagery.

## Engine / infra notes
- `ocr_crops.py` outputs per crop: `ocr.json` (full records → Phase 3), `ocr.csv` (flat fields),
  `vis/*.jpg` (crop + extracted fields panel). Engine is behind a class so a bigger model or a
  different backend drops in.
- **Bigger model = better fidelity.** Pass `--model Qwen/Qwen2.5-VL-3B-Instruct` (or 7B) with
  `--device cuda` for stronger reads; 2B is the CPU-friendly floor that already clears EasyOCR.
- **HF download throttling** was the real friction, not the code: unauthenticated large-file pulls
  stalled on both the laptop and the GPU box (Xet backend hangs on Windows; `hf-mirror.com` is
  firewalled from the datacenter box; `hf_transfer` wouldn't install). Resolved with a plain-HTTPS
  **resume-on-retry loop** (`HF_HUB_DISABLE_XET=1`), which carried the 4.4 GB download through the
  drops. An HF token would remove the throttle entirely.

## Next
- Scale to **all** tracked crops (run `track.py` per run, then `ocr_crops.py --crops`), ideally on
  GPU with a 3B/7B model for the fine fields.
- Feed `ocr.json` (presence of text + brand/chemistry/marks) into **Phase 3** (text-only vs
  text+image type classification).
