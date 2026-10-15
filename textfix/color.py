"""Màu: CIELAB (L 0-100, a / b quanh 0), để đo khác biệt màu bằng ΔE (khoảng cách Euclid trong Lab)."""

from __future__ import annotations

import numpy as np


def lab(rgb: np.ndarray) -> np.ndarray:
    """Ảnh RGB uint8 (h, w, 3) -> Lab float32."""
    import cv2
    a = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a[..., 0] *= 100 / 255
    a[..., 1:] -= 128
    return a


def hex_lab(hx: str) -> np.ndarray:
    """"#rrggbb" -> Lab (3,)."""
    return lab(np.uint8([[[int(hx[i:i + 2], 16) for i in (1, 3, 5)]]]))[0, 0]
