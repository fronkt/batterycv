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

## VLM OCR (Phase 2/2b)
- **Coverage % is not accuracy.** Qwen2.5-VL-3B returned chemistry/voltage/capacity on ~98 % of
  698 crops — and the value distributions showed it was prior-filling (chemistry literally
  constant "Li-ion", incl. every LiSO2/NiCd/NiMH cell). Always check the **distribution of
  values against known class truth** before treating a field as signal; "do not guess" prompts
  are not obeyed at 2B–3B scale.
- Trust in order: verbatim-looking strings (part #, brand) > symbols/marks > any field with a
  plausible-default answer (V, mAh, chemistry).

## Tracking (Phase 1b / demos)
- **IoU trackers silently drop objects smaller than the per-frame motion.** The belt moves
  ~156 px/frame; any battery whose box is ≲160 px (ni_cd_small, ni_mh_all) moves its own
  length between frames → consecutive-frame IoU ≈ 0 → ByteTrack tracks never confirm → no
  ID, no crop, no box in the video (run 92: 41 raw detections → 1 unique id). The detector
  was fine — the association was. Fix: **BoT-SORT with global motion compensation** (the
  textured belt gives GMC a strong translation estimate) → run 92: 1 → 7 batteries.
- The failure was invisible in aggregate metrics (96.6% agreement!) because the dropped
  batteries never became rows. **Sanity-check COUNTS per class against the footage, not
  just accuracy of what survived** — absence doesn't show up in accuracy.
- Corollary: dense classes (li_ion_mobile, same ~167 px boxes) didn't collapse — neighbors
  provided accidental IoU matches → inflated/ID-switched tracks instead. Counts from IoU
  tracking on small fast objects are soft in both directions.

## Remote-box transfers / downloads
- **Never trust a downloader's exit code — verify artifacts.** On a box with flaky HF egress,
  `snapshot_download` exited 0 with zero weight shards on disk; a completion marker keyed on
  exit code fired spuriously. Gate on file count + byte sizes + sha256 vs the source.
- Long single-stream transfers die on cheap-box links. Resumable pattern without rsync on
  Windows: `dd iflag=skip_bytes,count_bytes skip=$(remote byte size) count=256M | ssh 'cat >> f'`
  in a loop until sizes match, then sha256 both ends.
- Prefer shipping small derived artifacts (698 crops = 19 MB) over staging raw data onto
  disk-tight shared boxes; run the heavy model where the disk is.
