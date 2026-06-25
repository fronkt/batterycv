"""Lightweight drawing helpers for boxes / track IDs."""
from __future__ import annotations

import cv2
import numpy as np


def draw_boxes(
    bgr: np.ndarray,
    boxes,
    labels=None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw xyxy boxes (optionally with text labels) on a copy of the image."""
    out = bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        if labels is not None:
            txt = str(labels[i])
            cv2.putText(out, txt, (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out
