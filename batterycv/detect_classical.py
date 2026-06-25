"""Classical, label-free battery detector — FALLBACK / baseline only.

SAM (pseudo_label_sam.py) is the primary box generator; this exists so we can bootstrap and
sanity-check without a GPU. It keys on local texture/edge energy (batteries carry edges & text,
the belt is comparatively smooth at large scale) plus colour saturation (labels).
Tune via the keyword args; evaluate against the hand-verified set before trusting it.
"""
from __future__ import annotations

import cv2
import numpy as np

from .preprocess import normalize_illumination


def detect_batteries(
    bgr: np.ndarray,
    min_area_frac: float = 5e-4,
    max_area_frac: float = 0.15,
    aspect_range: tuple[float, float] = (0.15, 6.0),
    morph_kernel: int = 9,
) -> list[tuple[int, int, int, int]]:
    """Return candidate battery boxes as (x1, y1, x2, y2) in pixel coords."""
    h, w = bgr.shape[:2]
    img = normalize_illumination(bgr)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    # texture/edge energy via morphological gradient on a blurred image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

    # combine edge energy with colour saturation (labels), then Otsu threshold
    energy = cv2.addWeighted(grad, 0.7, sat, 0.3, 0)
    _, mask = cv2.threshold(energy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w
    boxes: list[tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if not (min_area_frac * img_area <= area <= max_area_frac * img_area):
            continue
        aspect = bw / max(bh, 1)
        if not (aspect_range[0] <= aspect <= aspect_range[1]):
            continue
        boxes.append((x, y, x + bw, y + bh))
    return boxes
