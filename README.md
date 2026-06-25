# batterycv

Computer-vision battery sorting for Prof. Bin Chen's DOE-funded recycling project
(Purdue Fort Wayne · UHV Technologies · Penn State). Conveyor-mounted camera → detect & track
batteries → read text → classify type → evaluate.

**Phase 1 (current): detection + tracking.** OCR and type classification are scaffolded but
deferred (see `tasks/todo.md`).

## Data
- Source: `OneDrive_1_3-7-2025.zip` (~10.7 GB), Basler acA1300-200uc, **1280×1024 color BMP**.
- 6 folders = weak, session-level **type labels**.
- ⚠️ The zip nests an exact duplicate of the laptop folder inside the mobile folder (366 files).
  `extract_data.py` auto-quarantines it. **True dataset = 2,493 distinct frames:**

| label | chemistry | form | frames |
|---|---|---|---|
| li_ion_mobile | Li-ion | mobile | 1099 |
| li_ion_laptop | Li-ion | laptop | 366 |
| ni_cd_bulk | Ni-Cd | bulk | 346 |
| liso2 | LiSO2 | cell | 273 |
| ni_cd_small | Ni-Cd | small | 248 |
| ni_mh_all | Ni-MH | mixed | 161 |

Frames are dark/low-contrast with many batteries scattered per frame → illumination is
normalized (CLAHE) before detection. No box/text ground truth exists; a small hand-verified
set is used for honest detection metrics.

## Layout
```
scripts/   extract_data, build_manifest, eda, make_val_split,
           pseudo_label_sam, train_detector, track, eval_detection
batterycv/ config, io (manifest/timestamps/runs), preprocess, detect_classical, viz
configs/   paths.yaml (all paths), yolo_battery.yaml
vast/      setup.sh (RTX 5090 / cu128 GPU box)
tasks/     todo.md, lessons.md
```
Data, weights, and `runs/` live outside git (see `.gitignore`); paths in `configs/paths.yaml`.

## Pipeline
```bash
# local (CPU): data prep
python scripts/extract_data.py
python scripts/build_manifest.py
python scripts/eda.py
python scripts/make_val_split.py --per-class 12      # generate 72 eval frames
python scripts/label_eval.py                         # draw boxes -> eval/labels/ (built-in cv2 labeler)

# GPU box (Vast.ai): bash vast/setup.sh ; stage data ; then
python scripts/pseudo_label_sam.py --ckpt checkpoints/sam_vit_h_4b8939.pth
python scripts/train_detector.py --model yolo11s.pt --epochs 80 --imgsz 1024 --device 0

# local: evaluate + track
python scripts/eval_detection.py --weights runs/detect/battery_yolo11/weights/best.pt
python scripts/track.py --weights runs/detect/battery_yolo11/weights/best.pt --run 0
```

## Setup
```bash
python -m venv .venv && .venv/Scripts/activate    # Windows
pip install -r requirements.txt
```
