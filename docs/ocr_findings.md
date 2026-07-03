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

## Scaled run — all 698 crops, Qwen2.5-VL-3B on GPU (2026-07-02)

Pipeline: `track.py` over **all 103 runs** locally (CPU, ~35 min, 0 failures) → 698 best-conf
crops pooled → `ocr_crops.py --engine qwen --model <local Qwen2.5-VL-3B> --device cuda` on a
rented RTX 3090 → **698/698 records in 50 min (~4.3 s/crop)**. Results archived at
`results/phase2_ocr/ocr.{json,csv}` (+ two evidence panels); full vis set in
`batterycv-data/work/ocr_all/` locally.

### Coverage vs. truth — the 3B scales the hallucination, it doesn't fix it
Raw field coverage looks spectacular (chemistry/voltage/capacity ~98% non-empty) and is **not
real**. Value distributions expose systematic prior-filling:

- **chemistry**: "Li-ion" on **100 % of non-empty rows — including all 49 LiSO2 and all 98
  Ni-Cd/Ni-MH crops** (0 % correct outside the Li-ion classes). It is a prior, not a read.
- **voltage**: 94 % of claims are two default values (3.7 V ×465, 11.55 V ×194).
- **capacity**: 2600 mAh ×377 (54 %), 41 Wh ×137.
- Visual spot-checks confirm (panels in `results/phase2_ocr/`): a LiSO2 crop where the read
  model # "38A" **matches visible print** while the claimed Li-ion/3.7V/2600mAh appears nowhere;
  an IOTA Ni-Cd emergency pack with a fully legible dense label the 3B barely transcribed —
  on such crops the *model*, not the imagery, is the limit.

### Trust tier for Phase 3
1. **Trustworthy:** `model`/part # (107 crops, 15 %) and `manufacturer` (99 crops, 14 % — Samsung,
   Dell, Huawei, DEWALT, Canon, Panasonic, HP…). Hard to fake consistently; spot-checks match print.
2. **Partially trustworthy:** `marks` (99 % non-empty; CE ×601 suggests some prior-filling too).
3. **Do NOT use:** `chemistry`, `voltage`, `capacity` — prior-dominated. In particular, OCR
   chemistry is **constant** ("Li-ion") and carries zero class signal.

### Phase-3 implication + upgrade paths
Type classification cannot lean on OCR chemistry/specs. Usable OCR-side features: text
presence/density (`n_chars`), brand/part-# presence, mark count — combined with visual features.
Paths to better fine fields, in rough order of value: (a) **7B model** (needs a box with ≥25 GB
disk; the 16 GB shared boxes can't hold it), (b) transcribe-then-parse prompting (ask for a
verbatim transcript first, parse fields from it — harder to prior-fill), (c) reject any
voltage/capacity/chemistry not substringed in the raw transcript.

### Infra notes (this run)
- The rented box's egress to HF was flaky (TLS handshake timeouts); a snapshot_download attempt
  even **exited 0 without the weight shards**. Fix: download on the laptop (model was already
  cached), then ship the snapshot dir with a **resumable byte-offset loop** (`dd
  iflag=skip_bytes,count_bytes skip=$(remote size) | ssh 'cat >>'` in 256 MB rounds) and gate on
  **sha256 match**, not exit codes. Point `--model` at the local dir — no network at load time.
- Tracking locally + shipping only crops (19 MB) beats staging 0.8 GB of frames onto a
  disk-tight box that is also running other workloads.

## Next
- **Phase 3 is unblocked**: 698 per-battery records with label/run/track ids at
  `results/phase2_ocr/ocr.json`. Feature set per the trust tier above.
- Optional fidelity pass: 7B + transcript-constrained prompting on the 206 crops with a
  brand/part-# read, on a box with real disk.
