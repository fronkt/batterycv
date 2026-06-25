# Lessons / gotchas — batterycv

## Data quality
- **Nested duplicate folder in the delivered zip.** `OneDrive_1_3-7-2025.zip` nests an exact
  copy of `Li_ion_laptop_battery_03_03_25/` *inside* `Li_ion_mobile_battery_03_03_25/`
  (366 files, identical MD5). Naive top-level counting gives mobile=1465 / total=2859, but the
  **true distinct dataset is mobile=1099 / total=2493**. `extract_data.py` auto-quarantines it.
  Pattern to apply: after any extraction, **recursive-count and check for unexpected subfolders**
  before trusting per-class counts; never label by naive top-level grouping.

## Imagery
- Frames are **very dark / low-contrast**; ~2.6x brightness (or CLAHE) is needed before the
  batteries and their text become legible. Always normalize illumination before detection/OCR.
- **Many batteries per frame** (10+), rotated, often clipped at edges. Belt is textured/striped
  (can fool naive edge/threshold detectors → prefer SAM or a trained detector).
- Labels are **session-level (folder)**, not per-object. Every battery in a folder shares that
  folder's type — usable as a weak label, but detection/OCR have **no ground truth** (hence the
  small hand-verified eval set).
