"""Illumination normalization for the dark, unevenly-lit conveyor frames.

The raw BMPs are very dark with a bright center / dark corners. CLAHE on the L channel
recovers contrast (and battery text legibility) without blowing out the bright region.
"""
from __future__ import annotations

import cv2
import numpy as np


def normalize_illumination(bgr: np.ndarray, clip: float = 2.5, grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel of LAB. Returns a BGR uint8 image."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def brightness_stats(bgr: np.ndarray) -> dict[str, float]:
    """Mean / std / p5 / p95 of grayscale intensity (for EDA)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "p5": float(np.percentile(g, 5)),
        "p95": float(np.percentile(g, 95)),
    }


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img
