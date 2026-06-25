"""Extract the battery-image zip into the working data dir and verify counts.

The zip ships the 6 class folders at top level (folder name == weak type label).
We extract to <data_root>/raw/<class>/*.bmp and assert the expected per-class counts
so a corrupt/partial download is caught immediately.

Usage:
    python scripts/extract_data.py
    python scripts/extract_data.py --zip <path> --out <data_root>/raw
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

DEFAULT_ZIP = Path(r"C:/Users/frank/Downloads/OneDrive_1_3-7-2025.zip")
DEFAULT_OUT = Path(r"C:/Users/frank/batterycv-data/raw")

# Distinct per-class frame counts AFTER removing a packaging duplicate (see note below).
# The zip nests an exact copy of the laptop folder inside the mobile folder:
#   Li_ion_mobile_battery_03_03_25/Li_ion_laptop_battery_03_03_25/  (366 files, identical MD5)
# Left in place, those 366 laptop frames would be mislabeled "mobile". We quarantine them,
# so the genuine mobile count is 1099 (not the naive 1465) and the true total is 2493.
EXPECTED = {
    "Li_ion_mobile_battery_03_03_25": 1099,
    "Li_ion_laptop_battery_03_03_25": 366,
    "Ni_Cd_bulk_battery_03_03_25": 346,
    "LiSO2_battery_03_03_25": 273,
    "Ni_Cd_small_battery_03_03_25": 248,
    "Ni_MH_all_battery_03_03_25": 161,
}
TOTAL = sum(EXPECTED.values())  # 2493

# Known mislabeled duplicate to move out of the labeled tree (reversible: re-extract from zip).
NESTED_DUPE = "Li_ion_mobile_battery_03_03_25/Li_ion_laptop_battery_03_03_25"


def extract(zip_path: Path, out_dir: Path) -> None:
    if not zip_path.exists():
        sys.exit(f"ERROR: zip not found: {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".bmp")]
        print(f"zip contains {len(members)} .bmp entries; extracting to {out_dir} ...")
        done = 0
        for m in members:
            zf.extract(m, out_dir)
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(members)}")
    print(f"extracted {done} files")


def quarantine_dupe(out_dir: Path) -> None:
    """Move the misplaced nested laptop copy out of the labeled tree, into ../quarantine."""
    nested = out_dir / NESTED_DUPE
    if nested.is_dir():
        dest = out_dir.parent / "quarantine" / "nested_laptop_dupe_under_mobile"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"quarantine target already exists, skipping move: {dest}")
        else:
            nested.rename(dest)
            print(f"quarantined nested duplicate -> {dest}")


def verify(out_dir: Path) -> bool:
    ok = True
    total = 0
    print("\n--- verification ---")
    for cls, exp in EXPECTED.items():
        # recursive count so a stray subfolder is caught rather than silently miscounted
        n = len(list((out_dir / cls).rglob("*.bmp"))) if (out_dir / cls).is_dir() else 0
        total += n
        flag = "ok " if n == exp else "FAIL"
        if n != exp:
            ok = False
        print(f"  {flag} {cls:<34} {n}/{exp}")
    print(f"  total {total}/{TOTAL}")
    if not ok:
        print("WARNING: counts do not match expected — extraction may be incomplete.")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify-only", action="store_true", help="skip extraction, just check counts")
    args = ap.parse_args()

    if not args.verify_only:
        extract(args.zip, args.out)
    quarantine_dupe(args.out)
    ok = verify(args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
