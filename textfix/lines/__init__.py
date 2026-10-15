"""HÌNH HỌC VÀ SỐ ĐO DÒNG từ NÉT Hi-SAM (bước 4a).

Vùng dòng do OCR chia (slots.lines: bộ dò + các bước tách / gộp). Ở đây mỗi dòng nhận các MẢNH NÉT của nó (một mảnh thuộc
đúng một dòng) và mọi đại lượng hình học đo trên chính các mảnh đó:

    G = measure_lines(draft, ink, regions, ocr_angles, boxes)
    G[k] = {"shape", "angle", "ink_angle", "ids", "ornament" (khung mảnh trang trí), "x0", "h_cap", "h_x", "base", "color", "stroke", "stroke_contrast"}
    R = rows(...)     # hàng nhiều cỡ giữa các dòng (runs.rows)

  hướng   angle.orient: nét + OCR (hai nguồn), xem angle.py;
  số đo   measure: cao chữ hoa / thường, đường chân, màu lõi nét, độ dày và tương phản nét;
  đầu dòng lead.lead (dòng ngang): mảnh trang trí đứng trước chữ (không thuộc chữ: ids bỏ chúng) + mép trái mới của khung;
Dòng không có mảnh nét nào (Hi-SAM không thấy chữ): chỉ có hướng theo OCR, số đo None.
"""

from __future__ import annotations

import numpy as np

from .angle import SNAP, VERTICAL, direction, orient, wrap
from .group import Group
from .ink import Ink, own
from .lead import lead
from .measure import CAP_EM, measure
from .runs import rows

__all__ = ["CAP_EM", "SNAP", "VERTICAL", "Ink", "measure_lines", "rows", "wrap"]

NONE = {"ornament": [], "x0": None, "h_cap": None, "h_x": None, "base": None, "color": None, "stroke": None, "stroke_contrast": None}


def measure_lines(draft: np.ndarray, ink: Ink, regions: list[np.ndarray], ocr_angles: list[float],
                  boxes: list[list[float]]) -> list[dict]:
    out = []
    for ids, ao, b in zip(own(ink, regions), ocr_angles, boxes):
        tall = b[3] - b[1] > b[2] - b[0]
        if not ids:
            shape, ang = orient(None, ao, tall)
            out.append({"shape": shape, "angle": ang, "ink_angle": None, "ids": [], **NONE})
            continue
        g = Group(ink, ids)
        ia = direction(g)
        shape, ang = orient(ia, ao, tall)
        orn, x0 = lead(g, ink, b[0]) if shape == "straight" else ([], None)
        ids = [i for i in ids if i not in orn]
        out.append({"shape": shape, "angle": ang, "ink_angle": None if ia is None else round(ia, 2), "ids": ids,
                    "ornament": [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h in ink.stats[orn, :4]],
                    "x0": x0, **measure(draft, ink, ids)})
    return out

