"""Mảnh nét Hi-SAM (thành phần liên thông) và việc chia mảnh cho các dòng."""

from __future__ import annotations

import numpy as np

MIN_PX = 6    # mảnh nét nhỏ hơn 6 điểm: hạt nhiễu
OWN = 0.5     # mảnh thuộc vùng chứa nhiều điểm của nó nhất, khi vùng đó chứa >= nửa mảnh


class Ink:
    """Nhãn liên thông, khung, diện tích và toạ độ mọi điểm nét (đo theo hướng bất kỳ mà không quét lại ảnh)."""

    def __init__(self, ink: np.ndarray):
        import cv2
        self.mask = np.asarray(ink, bool)
        self.n, self.cc, self.stats, _ = cv2.connectedComponentsWithStats(self.mask.astype(np.uint8), 8)
        ys, xs = np.nonzero(self.mask)
        self.ys, self.xs, self.lab = ys.astype(np.float64), xs.astype(np.float64), self.cc[ys, xs]
        self.area = np.bincount(self.lab, minlength=self.n)
        self.ok = self.area >= MIN_PX
        self.ok[0] = False

    def points(self, ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, y, nhãn) các điểm nét của các mảnh `ids`."""
        sel = np.isin(self.lab, list(ids))
        return self.xs[sel], self.ys[sel], self.lab[sel]

    def mask_of(self, ids) -> np.ndarray:
        return np.isin(self.cc, list(ids))


def own(ink: Ink, regions: list[np.ndarray]) -> list[list[int]]:
    """Mỗi mảnh nét về MỘT vùng (vùng chứa nhiều điểm của nó nhất, khi >= OWN mảnh): hai vùng chồng nhau không chia đôi một
    chữ. -> id các mảnh của từng vùng."""
    if not regions:
        return []
    hits = np.stack([np.bincount(ink.cc[r], minlength=ink.n) for r in regions])
    best = hits.argmax(0)
    good = ink.ok & (hits.max(0) >= OWN * ink.area)
    return [[int(i) for i in np.nonzero(good & (best == k))[0]] for k in range(len(regions))]
