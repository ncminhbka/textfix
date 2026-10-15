"""MÀU và hiệu ứng: màu đo trên nét nháp đối chiếu màu VLM (hai trong ba nguồn), lọc CSS hiệu ứng VLM đưa.
"""

from __future__ import annotations

import re

import numpy as np

COLOR_AGREE = 25.0  # hai nguồn màu đồng ý: lệch <= 25 ΔE


EFFECT_KEYS = re.compile(r"^\s*(-webkit-)?(text-shadow|text-stroke(-width|-color)?|background(-image|-clip|-color)?|"
                         r"text-fill-color|filter|paint-order)\s*:", re.I)


def _rgb(hx) -> tuple | None:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(hx or ""))
    return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)) if m else None


def _sanitize(css: str) -> tuple[str, str]:
    """Hiệu ứng VLM ghi cho một vai (CSS) -> (phần đặt ở khung, phần đặt ở span ôm chữ). Chỉ nhận bóng / viền / dải màu; bỏ url()."""
    keep = [x.strip() for x in str(css or "").split(";") if EFFECT_KEYS.match(x) and "url(" not in x.lower() and "expression" not in x.lower()]
    grad = [x for x in keep if re.match(r"\s*(-webkit-)?(background|text-fill-color)", x, re.I)]
    clip = any("clip" in x.lower() and "text" in x.lower() for x in grad)
    # CHỮ TÔ DẢI MÀU (background-clip:text) phải ở span ôm chữ: ở khung thì dải màu chỉ phủ trong khung, nét tràn ra trong suốt
    # (09/10 dev3 g14, probe b21a "An khang thịnh vượng" mất hai đầu)
    inner = ";".join(grad) if clip else ""
    outer = ";".join(x for x in keep if not (clip and x in grad))
    return outer, inner


def pick_color(glyph: str | None, ink: str | None, vlm: str | None) -> tuple[str | None, str]:
    """MÀU CHỮ từ ba nguồn, mỗi nguồn sai một kiểu (dev2 15/10, 58 dòng ba nguồn lệch > 25 ΔE, soát 14):
      glyph  gom cụm màu trong khung (slots / glyph) -- sai 3/14 (chữ trên nền cùng tông);
      ink    lõi nét Hi-SAM (lines) -- sai 7/14 (chữ mảnh lẫn màu nền ở mép, chữ neon ra màu quầng);
      vlm    màu VLM ghi cho cả VAI -- sai 6/14 (một vai trên hai nền khác màu).
    Hai nguồn đồng ý (<= COLOR_AGREE) -> màu đo của cặp đó (glyph trước); không cặp nào -> glyph, rồi ink, rồi vlm. Đúng 13/14.
    -> (màu, lý do)."""
    from ..color import hex_lab

    def near(a, b):
        try:
            return a and b and float(np.linalg.norm(hex_lab(a) - hex_lab(b))) <= COLOR_AGREE
        except Exception:
            return False
    if glyph and (near(glyph, ink) or near(glyph, vlm)):
        return glyph, "đo khung + " + ("nét" if near(glyph, ink) else "VLM") + " đồng ý"
    if ink and near(ink, vlm):
        return ink, "nét + VLM đồng ý"
    for c, why in ((glyph, "chỉ còn đo khung"), (ink, "chỉ còn đo nét"), (vlm, "chỉ còn VLM")):
        if c:
            return c, why + (" (ba nguồn lệch nhau)" if sum(bool(x) for x in (glyph, ink, vlm)) > 1 else "")
    return None, "không nguồn nào"
