"""Cụm nét Hi-SAM chưa dòng nào nhận (Hi-SAM thấy chữ, OCR không) -> dòng "orphan" để VLM quyết.
"""

from __future__ import annotations

import numpy as np

from .detect import CAP_EM

ORPHAN_MIN = 0.012   # cụm nét chưa dòng nào nhận: cao >= 1,2% cạnh ngắn ảnh ...


ORPHAN_PX = 80       # ... và >= 80 điểm nét (nhỏ hơn là hạt nhiễu, để nguyên)


ORPHAN_MAX = 8       # nhiều nhất 8 cụm to nhất mỗi poster thành dòng (chữ li ti trên bao bì, biển hiệu trong ảnh nền: để nguyên)


def _orphans(L: list, draft: np.ndarray, ink: np.ndarray, masks: dict) -> tuple[list, list[str]]:
    """Mảng nét Hi-SAM không chạm vùng của dòng nào -> gom các mảng gần nhau (đóng hình thái 0,8 cao nét) -> ô L "orphan". Chữ đọc bằng bộ đọc
    của OCR (đọc không ra: để rỗng -- vẫn là ô, VLM nhìn ảnh mà quyết)."""
    import cv2

    from ..erase.mask import line_region
    from . import ocr
    H, W = ink.shape
    zone = np.zeros((H, W), bool)
    for l in L:
        zone |= masks[l["id"]] if l["id"] in masks else line_region(l, H, W, 0.3)
    n, cc, st, _c = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    taken = np.unique(cc[zone & ink])
    rest = ink & ~np.isin(cc, taken[taken > 0])
    hs = [int(st[i, cv2.CC_STAT_HEIGHT]) for i in np.unique(cc[rest]) if i > 0 and st[i, cv2.CC_STAT_HEIGHT] >= 4]
    if not hs:
        return L, []
    hm = float(np.median(hs))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(round(0.8 * hm)) + 1, 2 * int(round(0.8 * hm)) + 1))   # mọi hướng: chữ chạy vòng
    nb, bc, bst, _c = cv2.connectedComponentsWithStats(cv2.morphologyEx(rest.astype(np.uint8), cv2.MORPH_CLOSE, k), 8)
    found = []
    for i in range(1, nb):
        sel = rest & (bc == i)
        px = int(sel.sum())
        ys, xs = np.nonzero(sel)
        if px < ORPHAN_PX or (ys.max() + 1 - ys.min()) < ORPHAN_MIN * min(W, H):
            continue
        found.append((px, sel, [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]))
    found.sort(key=lambda t: -t[0])
    log = []
    if len(found) > ORPHAN_MAX:
        log.append(f"cụm nét chưa dòng nào nhận: {len(found)} cụm, chỉ {ORPHAN_MAX} cụm to nhất thành ô, còn lại để nguyên")
    for px, sel, box in found[:ORPHAN_MAX]:
        x0, y0, x1, y1 = box
        text = ocr.read_region(draft, (((x0 + x1) / 2, (y0 + y1) / 2), (x1 - x0, y1 - y0), 0.0))
        ch = [int(st[j, cv2.CC_STAT_HEIGHT]) for j in np.unique(cc[sel]) if j > 0 and st[j, cv2.CC_STAT_HEIGHT] >= 4]
        gh = float(np.median(ch)) if ch else (y1 - y0)
        caps = bool(text) and sum(c.isupper() or c.isdigit() for c in text if c.isalnum()) >= 0.8 * max(1, sum(c.isalnum() for c in text))
        col = np.median(draft[sel], axis=0)
        new = {"id": f"L{1 + max([int(l['id'][1:]) for l in L] or [0])}", "box": box, "ocr_box": list(box),
               "poly": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "ocr": text, "base": None,
               "size": float(min(gh, y1 - y0) / (CAP_EM if caps or not text else 0.62)), "color": "#%02x%02x%02x" % tuple(int(v) for v in col),
               "angle": 0.0, "caps": caps, "stroke": None, "stroke_contrast": None, "effect": None,
               "missed": True, "orphan": True}
        L = L + [new]
        log.append(f"cụm nét chưa dòng nào nhận: {[round(v) for v in box]} ({px} điểm nét) đọc {text!r} -> {new['id']} (VLM quyết giữ / viết / xoá)")
    return L, log
