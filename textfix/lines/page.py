"""BƯỚC 4a -- CÁC DÒNG CHỮ CỦA NHÁP: mỗi dòng thành một ô L# đúng toạ độ (OCR + số đo nét + chữ cong + cụm nét "orphan").

  M = build(draft, ink)        # {"L": [dòng], "size": [W, H], "by": {id: dòng}, "geo_log": [...]}

Các phần: detect.py (dòng OCR + số đo nét), split.py (tách mục trong dòng), curve.py (chữ cong), orphans.py (nét OCR
không thấy), order.py (đánh số), view.py (ảnh đánh dấu, JSON, ghi chú chữ khó), printed.py (chữ in trên sản phẩm).
VLM không đưa toạ độ: chỉ trỏ id dòng.
"""

from __future__ import annotations

import numpy as np

from ..color import lab as _lab  # noqa: E402
from .curve import _geo_lines
from .detect import lines, shape_of
from .order import _renumber
from .orphans import _orphans
from .view import _typo_notes

# ---------------------------------------------------------------------------------------------------- đo


def build(draft: np.ndarray, ink: np.ndarray, product: bool = False) -> dict:
    """Ô của nháp: L# = dòng chữ. ink = nét chữ của Hi-SAM (hisam.strokes), nguồn duy nhất cho biết chữ ở đâu ở mức điểm ảnh
    (14/10: không còn bản xoá FLUX, không còn ô vỏ S / ô chi tiết I -- vỏ và icon là điểm ảnh của nháp).
      1. dòng OCR nguyên khung + số đo nét;
      2. hình học (lines/curve.py): chữ chạy theo cung -> một ô L có đường chân; vùng chữ của bộ dò mà OCR không trả dòng nào -> ô L mới;
      3. CỤM NÉT CHƯA DÒNG NÀO NHẬN (Hi-SAM thấy chữ, OCR không): thành ô L "orphan" để VLM quyết giữ / viết lại / xoá
         (người dùng chốt 14/10). VLM không nhắc tới thì GIỮ nguyên điểm ảnh nháp (compose.assign.resolve)."""
    H, W = draft.shape[:2]
    ink = np.asarray(ink, bool)
    L = lines(draft, ink)
    A = _lab(draft)
    for l in L:
        # KHUNG DÒNG = KHUNG OCR, nguyên bản (12/10: bỏ hẳn bước nắn khung theo nét mực -- trên nền ảnh chụp nét mực đo thiếu,
        # khung bị cắt hẹp / đẩy mép trái vào mỗi dòng một kiểu: 11-25% dòng, danh sách thò thụt, chữ co).
        l["ocr_box"] = list(l["box"])
    I: list = []
    glog, masks = [], {}
    L, I, glog, masks = _geo_lines(L, I, draft, A, ink, [])
    L, olog = _orphans(L, draft, ink, masks)
    L, glog = _renumber(L, [], [], glog + olog)
    for l in L:
        l["shape"] = shape_of(l)
    _rows(L)
    _typo_notes(L, glog)
    return {"L": L, "size": [W, H], "by": {m["id"]: m for m in L}, "geo_log": glog}


def _rows(L: list) -> None:
    """HÀNG NHIỀU CỠ (lines.rows): các dòng ngang chung đường chân, sát nhau, lệch cỡ ("Giảm" | "30%" | "tất") -> l["row"] = "R#"."""
    from .runs import rows
    ink = [l.get("ink") or {} for l in L]
    R = rows([l["box"] for l in L], [k.get("h_cap") or 0.0 for k in ink],
             [k.get("base") if l["shape"] == "straight" and k.get("h_cap") else None for l, k in zip(L, ink)])
    for l, r in zip(L, R):
        l.pop("row", None)
        if r is not None:
            l["row"] = f"R{r + 1}"
