"""Re-encode the raw BMP frames as JPEGs to shrink transfer to the GPU box ~15x.

The conveyor frames are uncompressed 1280x1024 BMPs (~3.8 MB each, ~9 GB total). JPEG q95 is
visually lossless for detection and cuts the dataset to ~0.6 GB, so it tar-pipes over SSH in
minutes. The eval set is already JPEG, so training-on-JPEG is *consistent* with eval. The class
folder names and filename stems are preserved, so build_manifest (which now globs jpg/bmp) and
the {label}__{stem} contract are unchanged.

    python scripts/bmp_to_jpg.py                 # raw_dir -> <data_root>/stage/raw
    python scripts/bmp_to_jpg.py --quality 92 --workers 12
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batterycv.config import load_paths
from batterycv.io import CLASSES


def convert(args_tuple) -> int:
    src, dst, quality = args_tuple
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  WARN unreadable: {src}")
        return 0
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="output root (default <data_root>/stage/raw)")
    args = ap.parse_args()

    paths = load_paths()
    raw = paths["raw_dir"]
    out_root = Path(args.out) if args.out else paths["data_root"] / "stage" / "raw"

    jobs = []
    for folder in CLASSES:
        src_dir = raw / folder
        if not src_dir.is_dir():
            continue
        dst_dir = out_root / folder
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in src_dir.glob("*.bmp"):
            jobs.append((p, dst_dir / f"{p.stem}.jpg", args.quality))

    print(f"converting {len(jobs)} BMP -> JPG q{args.quality} -> {out_root}")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ok in ex.map(convert, jobs):
            done += ok
            if done % 250 == 0:
                print(f"  {done}/{len(jobs)}")
    print(f"done: {done}/{len(jobs)} frames -> {out_root}")


if __name__ == "__main__":
    main()
