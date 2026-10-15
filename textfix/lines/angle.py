"""Hướng của một dòng: ngang / nghiêng / dọc, từ hai nguồn đo độc lập.

  nét   góc trục chính của đám nét, chỉnh lại bằng phép NẮN THẲNG theo mép chữ (_deskew) -- chỉ khi dòng có >= DIRECTED mảnh chính;
  OCR   góc khung bộ dò của dòng.
Mỗi nguồn sai theo một kiểu riêng: nét đo lệch ở từ vài chữ cái (ít mép để nắn), OCR đo lệch ở từ ngắn, từ có chân thò
("200g", "Đặt ngay" tới 5 độ) và chữ cỡ lớn ("MC" tới 8 độ). Nên dòng chỉ NGHIÊNG khi cả hai cùng thấy nghiêng (>= SNAP, cùng
chiều); góc lấy theo nét.
Dòng không đủ mảnh để nét nói hướng ("-50", "Liên") chỉ còn OCR. Dòng DỌC: nét dốc >= VERTICAL và khung cao hơn rộng.
Đo trên bench/lines_gt (kiểu đường gán mù 15/10), chữ poster, OCR chạy lại trên nháp: sai kiểu 8 -> 5 so với 4 ngưỡng góc của
OCR (TILT_*); chữ dọc trên bao bì 7 -> 0. Đòi hai nguồn lệch nhau <= 5 / 8 độ thì tệ hơn (10 / 6): OCR đo góc thô.
"""

from __future__ import annotations

import math

import numpy as np

from .group import DIRECTED, Group

SNAP = 2.0       # độ: nghiêng < 2 độ là ngang (mắt không thấy; trùng ngưỡng xoay của bộ dựng, render.ROT_MIN)
VERTICAL = 60.0  # độ: hướng đọc dốc hơn 60 độ là chữ dọc
DESKEW = 15.0    # độ: tìm góc nắn trong +-15 độ quanh trục chính


def wrap(a: float) -> float:
    """Góc của một đường (không có chiều) về [-90, 90)."""
    return (float(a) + 90.0) % 180.0 - 90.0


def direction(g: Group) -> float | None:
    """Góc hướng đọc theo nét (độ, chiều kim đồng hồ trên ảnh), None khi dòng có < DIRECTED mảnh chính."""
    if len(g.main()) < DIRECTED:
        return None
    return wrap(math.degrees(math.atan2(g.e1[1], g.e1[0])) + _deskew(g))


def _deskew(g: Group) -> float:
    """Góc lệch so với trục chính làm MÉP CHỮ sắc nhất: hình chiếu nét lên pháp tuyến, cực đại tổng bình phương đạo hàm của nó
    (mép trên / mép đáy thân mọi chữ thẳng hàng tạo bậc nhảy lớn). Chân thò (g, p, y) và dấu thanh không có mép chung nên
    không kéo lệch: "200g", "Đặt ngay" ra 0 độ (đo theo khối hình chiếu -- tổng bình phương -- ra 5 độ). Thô 0,5 độ, mịn 0,1."""
    step = max(1, len(g.u) // 20000)
    u, v = g.u[::step], g.v[::step]

    def sharp(d):
        t = math.radians(d)
        w = v * math.cos(t) - u * math.sin(t)
        return float((np.diff(np.bincount((w - w.min()).astype(int)).astype(np.float64)) ** 2).sum())
    best = max(np.arange(-DESKEW, DESKEW + 1e-6, 0.5), key=sharp)
    return float(max(np.arange(best - 0.5, best + 0.51, 0.1), key=sharp))


def orient(ink_angle: float | None, ocr_angle: float, tall: bool) -> tuple[str, float]:
    """-> (kiểu: straight / tilt / vertical, góc dựng). tall: khung dòng cao hơn rộng. Chiều đọc của dòng dọc (+90 / -90)
    do bộ đọc chọn sau."""
    o = wrap(ocr_angle)
    if ink_angle is None:
        return ("tilt", o) if SNAP <= abs(o) < VERTICAL else ("straight", 0.0)
    a = wrap(ink_angle)
    if abs(a) >= VERTICAL:
        return ("vertical", a) if tall else ("straight", 0.0)
    if abs(a) >= SNAP and abs(o) >= SNAP and a * o > 0:
        return "tilt", round(a, 2)
    return "straight", 0.0
