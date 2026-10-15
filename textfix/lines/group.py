"""Các mảnh nét của một dòng trong hệ trục của chính dòng đó (trục chính = hướng đọc theo hình dáng đám nét)."""

from __future__ import annotations

import numpy as np

MAIN = 0.35     # mảnh CHÍNH: cao >= 0,35 cao chữ (dấu thanh, chấm, gạch dưới thì không)
DIRECTED = 3    # >= 3 mảnh chính mới có hướng đọc đo được; ít hơn (một hai chữ cái, "-50") thì nét không nói hướng


def frame(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hệ trục của đám điểm: tâm, e1 (trục chính, x không âm), e2 (pháp tuyến "xuống")."""
    P = np.c_[xs, ys]
    c = P.mean(0)
    sub = P[:: max(1, len(P) // 4000)] - c
    _, _, vt = np.linalg.svd(sub, full_matrices=False)
    e1 = vt[0] if vt[0][0] > 1e-9 or (abs(vt[0][0]) <= 1e-9 and vt[0][1] > 0) else -vt[0]
    return c, e1, np.array([-e1[1], e1[0]])


class Group:
    """Toạ độ (u dọc trục chính, v vuông góc) mọi điểm nét, khoảng (u0, u1, v0, v1) từng mảnh, cao chữ h (p75 cao mảnh)."""

    def __init__(self, ink, ids):
        xs, ys, lab = ink.points(ids)
        self.c, self.e1, self.e2 = frame(xs, ys)
        P = np.c_[xs, ys] - self.c
        self.u, self.v, self.lab = P @ self.e1, P @ self.e2, lab
        self.ext = {}
        for i in set(ids):
            s = lab == i
            if s.any():
                self.ext[i] = (float(self.u[s].min()), float(self.u[s].max()), float(self.v[s].min()), float(self.v[s].max()))
        self.ids = [i for i in ids if i in self.ext]
        hs = [e[3] - e[2] + 1 for e in self.ext.values()]
        self.h = float(np.percentile(hs, 75)) if hs else 1.0

    def main(self) -> list[int]:
        return [i for i in self.ids if self.ext[i][3] - self.ext[i][2] + 1 >= MAIN * self.h]
