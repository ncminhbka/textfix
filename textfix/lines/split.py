"""Tách một dòng OCR thẳng thành các MỤC riêng ở khoảng trống không nét (OCR gộp "địa chỉ [icon] website").
"""

from __future__ import annotations

import numpy as np

from ..color import lab as _lab

GAP_SPLIT = 1.0      # trong một dòng OCR, khoảng KHÔNG nét rộng >= 1 cao dòng = hai mục riêng (khoảng cách từ ~0.3 cao dòng;


                     # OCR gộp "địa chỉ [icon] website" ở chân trang thành một dòng)
LEADER_H = 0.3       # hàng chấm nối: cột có nét cao <= 0.3 cao dòng là chấm / gạch nối, coi như khoảng trống


GAP_MIN = 0.4        # khoảng không nét quanh chỗ OCR chèn dấu cách kép: >= 0.4 cao dòng (khoảng cách từ thường ~0.3)


ICON_W = 1.6         # khối kẹp giữa hai khoảng không ở chỗ cắt, rộng <= 1.6 cao dòng = icon giữa hai mục


INK_DE = 25.0        # điểm nét = khác màu nền khung dòng (trung vị viền khung) >= 25 ΔE


def _split_gaps(img: np.ndarray, l: dict) -> list[dict]:
    """Tách dòng OCR (thẳng) thành các MỤC riêng:
    (a) khoảng không nét rộng >= GAP_SPLIT cao dòng;
    (b) chỗ OCR chèn >= 2 dấu cách (nó đánh dấu khoảng cách lớn): cắt tại khoảng không nét >= GAP_MIN cao dòng gần vị trí đó
        nhất; nếu giữa hai khoảng không quanh đó là một khối hẹp (<= ICON_W cao dòng: icon giữa hai mục, "địa chỉ [điện thoại]
        số") thì khối đó tách riêng, không thuộc dòng nào (lớp phủ tách nó thành chi tiết I).
    Chữ chia theo dấu cách gần chỗ cắt nhất (OCR không cho toạ độ từng chữ: chỉ là gợi ý cho designer)."""
    import re
    if abs(l.get("angle") or 0) >= 2:
        return [l]
    x0, y0, x1, y1 = (int(round(v)) for v in l["box"])
    h = y1 - y0
    if h < 4 or x1 - x0 < 3 * h:
        return [l]
    lab = _lab(np.ascontiguousarray(img[max(0, y0):y1, max(0, x0):x1]))
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    ink2 = np.linalg.norm(lab - np.median(border, 0), axis=2) >= INK_DE
    ink = ink2.any(0)
    # HÀNG CHẤM NỐI "tên ........ giá" (13/10 accept1 c03: 4 hàng OCR tách tên / giá, hàng thứ 5 chấm nối liền nên thành một
    # dòng, VLM viết tên, giá mất): OCR đọc ra >= 3 dấu chấm / gạch liền nhau -> các cột chỉ có nét THẤP (<= 0.3 cao dòng: chấm,
    # gạch dưới) là khoảng trống, không phải chữ
    if re.search(r"[.·•_…]{3,}|(?:[.·•] ?){4,}", l["text"]):
        ext = ink2.sum(0)
        ink = ink & (ext > LEADER_H * h)
    cols = np.flatnonzero(ink)
    if len(cols) < 2:
        return [l]
    W = x1 - x0
    gaps = [(int(a) + 1, int(b)) for a, b in zip(cols[:-1], cols[1:]) if b - a - 1 >= GAP_MIN * h]   # [đầu, cuối) khoảng không nét
    text = l["text"]
    cuts = []   # (cột kết thúc mục trái, cột bắt đầu mục phải, vị trí chữ để chia)
    for a, b in gaps:
        if b - a >= GAP_SPLIT * h:
            cuts.append((a, b, None))
    for m in re.finditer(r" {2,}", text):
        want = (m.start() + m.end()) / 2 / max(1, len(text)) * W
        near = sorted(gaps, key=lambda g: abs((g[0] + g[1]) / 2 - want))
        if not near or abs((near[0][0] + near[0][1]) / 2 - want) > 2 * h:
            continue
        g = near[0]
        # khối hẹp kẹp giữa hai khoảng không liền nhau quanh chỗ cắt = icon tách riêng
        side = [q for q in gaps if q != g and abs((q[0] + q[1]) / 2 - want) <= 2 * h]
        pair = None
        for q in side:
            lo, hi = (g, q) if q[0] > g[0] else (q, g)
            if 0 < hi[0] - lo[1] <= ICON_W * h:
                pair = (lo[0], hi[1])
        cuts.append((pair[0], pair[1], m) if pair else (g[0], g[1], m))
    if not cuts:
        return [l]
    cuts = sorted({(a, b): (a, b, m) for a, b, m in cuts}.values(), key=lambda c: c[0])
    spans, start = [], int(cols[0])
    for a, b, _ in cuts:
        if a > start:
            spans.append((start, a))
        start = b
    spans.append((start, int(cols[-1]) + 1))
    # chia chữ: chỗ cắt có dấu cách kép dùng đúng vị trí đó, chỗ khác dùng dấu cách gần tỉ lệ vị trí cắt nhất
    parts, pos = [], 0
    for k, (a, b, m) in enumerate(cuts):
        if m is not None:
            cut, nxt = m.start(), m.end()
        else:
            want = int(round(len(text) * (a + b) / 2 / W))
            sp = [i for i, ch in enumerate(text) if ch == " " and i > pos]
            cut = min(sp, key=lambda i: abs(i - want)) if sp else max(pos + 1, want)
            nxt = cut
        parts.append(text[pos:cut])
        pos = max(pos, nxt)
    parts.append(text[pos:])
    if len(parts) != len(spans):
        return [l]
    out = []
    for (a, b), part in zip(spans, parts):
        part = part.strip().strip(".·•_… ").strip()
        if not part or b - a < 2:
            continue
        bx = [float(x0 + a), float(y0), float(x0 + b), float(y1)]
        out.append({**l, "text": part, "box": bx, "poly": [[bx[0], bx[1]], [bx[2], bx[1]], [bx[2], bx[3]], [bx[0], bx[3]]]})
    return out or [l]
