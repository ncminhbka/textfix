"""DÒNG CHỮ của nháp và việc của VLM: code đo mỗi dòng chữ thành một Ô L# đúng toạ độ; VLM chỉ quyết viết gì vào từng ô; code dựng
đúng khung từng ô (render.py). Đường đang chạy là "chỉ dòng chữ" (lonly.py); tệp này còn giữ lời dặn và bộ mô tả của đường ô
vỏ / ô chi tiết cũ (SYSTEM, describe, review) -- chưa cắt, xem docs/ROADMAP.md.

  M = build(draft, ink)                           # L# dòng chữ: OCR nguyên khung + số đo nét + chữ cong + cụm nét "orphan"
  P = plan(draft, M, prompt, texts, call)         # VLM: {style, ops: [{slots, kind, html, box_style, client}], missing}
  P, log = validate(P, M, texts)                 # + chặn chữ: dấu theo câu khách, chữ bịa, dòng lặp

VLM không đưa toạ độ: chỉ trỏ id ô. Mỗi ô thuộc đúng một lệnh (lệnh chữ được gộp nhiều dòng xếp chồng thành một khối).
"""

from __future__ import annotations

import io
import json
import os
import re

import numpy as np

CAP_EM = 0.72        # cao chữ hoa / em (cũng ~ cao chữ số, nét lên b d h k l t)
TALL_MIN = 0.15      # dòng có >= 15% ký tự cao -> ước cỡ theo cao nét cao (glyph h_tall)
X_EM = 0.53          # cao chữ thường (x-height) / em
INK_IN_SHELL = 25.0  # trong vỏ: nét chữ = khác màu lòng vỏ >= 25 ΔE (cả lòng vỏ đều thuộc lớp phủ)
GAP_SPLIT = 1.0      # trong một dòng OCR, khoảng KHÔNG nét rộng >= 1 cao dòng = hai mục riêng (khoảng cách từ ~0.3 cao dòng;
                     # OCR gộp "địa chỉ [icon] website" ở chân trang thành một dòng)
LEADER_H = 0.3       # hàng chấm nối: cột có nét cao <= 0.3 cao dòng là chấm / gạch nối, coi như khoảng trống
GAP_MIN = 0.4        # khoảng không nét quanh chỗ OCR chèn dấu cách kép: >= 0.4 cao dòng (khoảng cách từ thường ~0.3)
ICON_W = 1.6         # khối kẹp giữa hai khoảng không ở chỗ cắt, rộng <= 1.6 cao dòng = icon giữa hai mục
INK_DE = 25.0        # điểm nét = khác màu nền khung dòng (trung vị viền khung) >= 25 ΔE
ROW_GAP = 0.5        # chi tiết giống nhau (cao lệch <= 25%) cùng hàng (tâm lệch <= 25% cao), cách nhau <= 0.5 cao = MỘT chi tiết
                     # (hàng sao, hàng chấm: designer dựng cả hàng bằng một thẻ)
DETAIL_MIN = 0.5     # chi tiết I: cạnh dài >= 0.5 cao dòng chữ trung vị của poster (icon designer đặt không nhỏ hơn nửa chữ thân;
                     # nhỏ hơn = vụn: mép vật FLUX vẽ lại, mẩu hoạ tiết)
SHELL_INK_DE = 8.0   # VỎ GIẢ: màu lòng "vỏ" gần trùng màu mực của dòng nằm trong nó (ΔE < 8) = nét chữ to bị coi là vỏ (vỏ thật
                     # luôn khác màu chữ: 130 poster 12/10, 2 ô <= 0.4, ô kế 16.5) -> không tạo ô S
KEEP_INK_MAX = 0.75  # GIỮ VỎ NHÁP: trong khung các dòng của vỏ, điểm "nét" (khác màu lòng) chiếm > 75% = lòng không phẳng màu
                     # (ảnh / hoạ tiết / màu lòng đo sai) -> không lấp nét được, vỏ đó vẽ lại bằng CSS
KEEP_HALO = 2        # nới mặt nạ nét chữ cũ (px): viền khử răng cưa


def _lab(x):
    import cv2
    a = cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float32)
    a[..., 0] *= 100 / 255
    a[..., 1:] -= 128
    return a


def _png(im) -> bytes:
    from PIL import Image
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _cov(a, b) -> float:
    """phần diện tích b nằm trong a"""
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h) / max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))


# ---------------------------------------------------------------------------------------------------- đo
def _tall(c: str) -> bool:
    import unicodedata
    return c.isupper() or c.isdigit() or unicodedata.normalize("NFD", c)[0] in "bdfhklt"


TILT_MIN = 4.0     # độ: dòng đo ra nghiêng dưới 4 độ là nhiễu đo của OCR -> thẳng (13/10: 57 / 1505 dòng ở 2-4 độ, nháp vẽ thẳng)
TILT_SURE = 10.0   # ... từ 10 độ trở lên là nghiêng thật (nhãn dán, dải xéo); ở giữa chỉ tin khi dòng đủ dài:
TILT_PAIR = 2.5    # ... VÀ có dòng khác trong poster nghiêng cùng góc (lệch <= 2,5 độ): nghiêng thật đi theo cụm (nhãn hai dòng);
                   # một dòng dài nghiêng lẻ 4-10 độ là OCR gộp hai mẩu lệch hàng (accept2 q02 "Hương dịu nhẹ" đo 8,4 độ)
TILT_LONG = 3.0    # cạnh dài / cạnh ngắn của đa giác >= 3 (từ ngắn "MC", "200g", "Glow Up" đo sai tới 8 độ)
VERT_TALL = 2.2    # dòng viết DỌC: đa giác OCR cao >= 2,2 lần rộng, đọc được >= 3 ký tự


def _line_angle(l: dict) -> float:
    """Góc DỰNG của một dòng OCR: bỏ nhiễu đo (người dùng 13/10: chữ ngả nghiêng dù nháp viết thẳng)."""
    a = float(l.get("angle") or 0.0)
    if abs(a) >= TILT_SURE:
        return a
    if abs(a) < TILT_MIN:
        return 0.0
    pl = l.get("poly")
    if pl and len(pl) == 4:
        e = sorted(float(np.hypot(pl[i][0] - pl[(i + 1) % 4][0], pl[i][1] - pl[(i + 1) % 4][1])) for i in range(4))
        reach = e[-1] / max(1.0, e[0])
    else:
        reach = (l["box"][2] - l["box"][0]) / max(1.0, l["box"][3] - l["box"][1])
    return a if reach >= TILT_LONG else 0.0


def _vertical(img: np.ndarray, l: dict) -> dict:
    """Dòng viết DỌC (chữ chạy dọc mép poster): OCR trả đa giác cao - hẹp với góc 0 và chữ rác (đọc sai chiều) -> mã coi là dòng
    ngang trong ô hẹp, chữ dựng ngang bé tí (accept2 "Ngày hội siêu sale"). Đọc lại theo hai chiều +90 / -90, lấy chiều tin hơn:
    ghi góc, chữ đọc được và đa giác theo chiều đọc. Không phải dòng dọc: trả nguyên."""
    from .ocr import MIN_REC, read_rect
    b = l["box"]
    w, h = b[2] - b[0], b[3] - b[1]
    if h < VERT_TALL * w or abs(float(l.get("angle") or 0.0)) >= TILT_SURE or sum(ch.isalnum() for ch in l["text"]) < 3:
        return l
    c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    best = max(((*read_rect(img, c, h, w, a), a) for a in (90.0, -90.0)), key=lambda t: t[1])
    if best[1] < MIN_REC or not any(ch.isalnum() for ch in best[0]):
        return l
    x0, y0, x1, y1 = b
    poly = [[x1, y0], [x1, y1], [x0, y1], [x0, y0]] if best[2] > 0 else [[x0, y1], [x0, y0], [x1, y0], [x1, y1]]
    return {**l, "text": best[0], "angle": best[2], "poly": poly, "vertical": True}


def lines(img: np.ndarray) -> list[dict]:
    """Dòng OCR + số đo nét: cao thân chữ -> cỡ chữ (px), màu, góc nghiêng."""
    import cv2
    from .glyph import measure
    from .ocr import read_lines
    H, W = img.shape[:2]
    # "dòng" không có chữ cái / chữ số (hàng sao "★★★★★", gạch, chấm) không phải chữ: để lớp phủ tách thành chi tiết I
    raw = [l for l in read_lines(img) if any(ch.isalnum() for ch in l["text"])]
    raw = [_vertical(img, q) for l in raw for q in _split_gaps(img, l)]
    raw_ang = [float(l.get("angle") or 0.0) for l in raw]   # góc OCR đo, trước khi bỏ nhiễu (ghi lại cho debug)
    angs = [_line_angle(l) for l in raw]
    for k, l in enumerate(raw):   # góc dựng chốt ở đây: mọi nơi dùng l["angle"] thấy cùng một góc
        a = angs[k]
        if TILT_MIN <= abs(a) < TILT_SURE and not any(j != k and abs(b - a) <= TILT_PAIR for j, b in enumerate(angs)):
            a = 0.0
        l["angle"] = a
    L = []
    for i, l in enumerate(raw, 1):
        ang = float(l["angle"])
        if abs(ang) >= 2:   # dòng NGHIÊNG: xoay thẳng ảnh quanh dòng rồi đo (khung thẳng của dòng nghiêng làm cao thân chữ phồng)
            P_ = np.asarray(l["poly"], np.float32)
            c = P_.mean(0)
            R = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0)
            # dòng dọc sát mép ảnh: xoay quanh tâm dòng thì dòng chìa ra ngoài khung -> dời tâm dòng về giữa ảnh
            R[0, 2] += W / 2 - float(c[0])
            R[1, 2] += H / 2 - float(c[1])
            img_r = cv2.warpAffine(img, R, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            q = np.hstack([P_, np.ones((4, 1), np.float32)]) @ R.T
            m = measure(img_r, [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())], l["text"], [])
        else:
            m = measure(img, l["box"], l["text"], [o["box"] for o in raw if o is not l])
        lt = [c for c in l["text"] if c.isalnum()]
        caps = bool(lt) and sum(c.isupper() or c.isdigit() for c in lt) >= 0.8 * len(lt)
        h = m["h"] if m else 0.7 * min(l["box"][3] - l["box"][1], l["box"][2] - l["box"][0] if l.get("vertical") else 1e9)
        size = h / (CAP_EM if caps else X_EM)
        # dòng có ký tự CAO (chữ hoa / số / nét lên): cỡ theo cao nét cao nhất -- mặt cắt mật độ đo lẫn cao chữ hoa với cao chữ
        # thường khi dòng nhiều chữ số ("Hotline 0866 777 888" ước 1.37 lần cỡ thật -> chữ dựng tràn ngang, co < 0.7: too_small)
        if not caps and m and m.get("h_tall") and sum(_tall(c) for c in lt) >= TALL_MIN * max(1, len(lt)):
            size = m["h_tall"] / CAP_EM
        L.append({"id": f"L{i}", "box": [float(v) for v in l["box"]], "poly": l["poly"], "ocr": l["text"],
                  "base": float(m["base"]) if m and abs(ang) < 2 else None,
                  "size": float(size), "color": m["color"] if m else None, "angle": ang,
                  "caps": caps, "stroke": round(float(m["stroke"]), 1) if m else None,
                  "stroke_contrast": round(float(m["contrast"]), 2) if m else None,
                  "effect": {k: v for k, v in (m.get("effect") or {}).items() if k in ("dx", "dy", "color")} if m and m.get("effect") else None,
                  "angle_ocr": round(raw_ang[i - 1], 1), **({"vertical": True} if l.get("vertical") else {})})
    return L


def _rows(I: list[dict]) -> list[dict]:
    """Gộp chi tiết giống nhau xếp thành hàng (ROW_GAP) thành một chi tiết."""
    I = sorted(I, key=lambda c: c["box"][0])
    out = []
    for c in I:
        x0, y0, x1, y1 = c["box"]
        h = y1 - y0
        prev = next((o for o in reversed(out) if o.get("shell") == c.get("shell")
                     and abs((o["_h"]) - h) <= 0.25 * max(o["_h"], h)
                     and abs((o["box"][1] + o["box"][3]) / 2 - (y0 + y1) / 2) <= 0.25 * max(o["_h"], h)
                     and 0 <= x0 - o["box"][2] <= ROW_GAP * max(o["_h"], h)), None)
        if prev is None:
            out.append({**c, "box": list(c["box"]), "_h": h})
        else:
            b = prev["box"]
            prev["box"] = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]
    for o in out:
        o.pop("_h")
    return out


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


def _anchor(l: dict) -> tuple[float, float]:
    """Một điểm nằm TRÊN dòng chữ: tâm khung; dòng cong: giữa đường chân, nhích lên nửa cao chữ (tâm khung thẳng của một vòm chữ
    rơi vào thứ nằm dưới vòm, không phải vào chữ)."""
    pt = l.get("path")
    if pt and len(pt) >= 2:
        k = len(pt) // 2
        (ax, ay), (bx, by) = pt[k - 1], pt[k]
        ln = float(np.hypot(bx - ax, by - ay)) or 1.0
        up = 0.35 * float(l.get("size") or 0.0)
        return ((ax + bx) / 2 + (by - ay) / ln * up, (ay + by) / 2 - (bx - ax) / ln * up)
    b = l["box"]
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


LEAD_GAP = (0.6, 4.0)   # khung OCR thừa bên trái nét chữ đầu tiên một khoảng 0,6-4 cao nét = khung ôm cả icon đứng trước dòng


def _trim_lead(L: list, ink: np.ndarray) -> list[str]:
    """KHUNG DÒNG BẮT ĐẦU TỪ NÉT CHỮ ĐẦU TIÊN khi khung OCR ôm cả icon đứng trước dòng (accept2 q02: OCR lấy luôn chiếc lá).
    Icon là của nháp, không gỡ (lonly.erase_mask), nên chữ mới không được dựng đè lên nó: mép trái khung dời tới nét đầu. Chỉ
    dời khi Hi-SAM thấy rõ chữ ở dòng đó (>= 3 mảng nét) -- thấy ít thì không biết chữ bắt đầu ở đâu, giữ khung OCR."""
    import cv2
    H, W = ink.shape
    log = []
    for l in L:
        if l.get("path") or l.get("orphan") or l.get("vertical") or abs(l.get("angle") or 0) >= 2:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in l["box"])
        if y1 - y0 < 8 or x1 - x0 < 16:
            continue
        sub = ink[max(0, y0):min(H, y1), max(0, x0):min(W, x1)].astype(np.uint8)
        n, _cc, st, _c = cv2.connectedComponentsWithStats(sub, 8)
        comps = [st[i] for i in range(1, n) if st[i][4] >= 6 and st[i][3] >= 4]
        if len(comps) < 3:
            continue
        hi = float(np.median([c[3] for c in comps]))
        first = min(int(c[0]) for c in comps)
        if LEAD_GAP[0] * hi <= first <= LEAD_GAP[1] * hi:
            nx = float(max(0, x0) + first - max(1.0, 0.1 * hi))
            b = l["box"]
            l["box"] = [nx, b[1], b[2], b[3]]
            l["ocr_box"] = list(l["box"])
            l["poly"] = [[nx, b[1]], [b[2], b[1]], [b[2], b[3]], [nx, b[3]]]
            l["lead_icon"] = True
            log.append(f"{l['id']}: khung OCR ôm cả thứ đứng trước chữ ({first} px) -> mép trái khung dời tới nét chữ đầu tiên")
    return log


DOT_MAX = 0.5    # chấm tròn đầu dòng: mảng nét đầu tiên của dòng, cạnh <= 0,5 cao dòng, gần vuông (cạnh ngắn >= 0,5 cạnh dài) ...
DOT_GAP = 0.2    # ... cách mảng nét kế tiếp >= 0,2 cao dòng theo chiều ngang ...
DOT_MID = (0.2, 0.8)   # ... tâm nằm ở khoảng giữa chiều cao dòng (dấu nháy ở trên, dấu chấm câu ở dưới thì không)


def _lead_dots(L: list, ink: np.ndarray) -> list[str]:
    """CHẤM TRÒN ĐẦU DÒNG LÀ KÝ TỰ CỦA DÒNG (14/10): OCR gộp "•" vào khung dòng và Hi-SAM coi nó là nét chữ, nên nó bị gỡ cùng
    chữ. Ghi l["dot"] = True để dòng chữ mới mở đầu bằng "• " (lonly.marks_rule). Icon, tick, hình nhỏ trước dòng KHÔNG phải nét
    chữ: không dò, không gỡ, không vẽ lại -- TextFix chỉ sửa chữ (người dùng chốt 14/10)."""
    import cv2
    H, W = ink.shape
    log = []
    for l in L:
        if l.get("path") or l.get("orphan") or l.get("vertical") or abs(l.get("angle") or 0) >= 2:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in l["box"])
        h = y1 - y0
        if h < 8 or x1 - x0 < 2 * h:
            continue
        sub = ink[max(0, y0):min(H, y1), max(0, x0):min(W, x1)].astype(np.uint8)
        n, _cc, st, _c = cv2.connectedComponentsWithStats(sub, 8)
        comps = sorted((st[i] for i in range(1, n) if st[i][4] >= 6), key=lambda t: t[0])
        if len(comps) < 3:
            continue
        bx, by, bw, bh = (int(v) for v in comps[0][:4])
        nxt = min(int(c[0]) for c in comps[1:])
        mid = (by + bh / 2) / max(1, sub.shape[0])
        # ... VÀ chữ OCR đọc không mở đầu bằng chữ / số (accept2 r04: " Swittch", "• Ket", ": Pin"). Chỉ theo hình thì nhầm:
        # h04 dòng "HOI" của tiêu đề có một mẩu nét nhỏ đứng trước -> poster ra "NGÀY • HỘI"
        o1 = (l.get("ocr") or "")[:1]
        if max(bw, bh) <= DOT_MAX * h and min(bw, bh) >= 0.5 * max(bw, bh) and nxt - (bx + bw) >= DOT_GAP * h \
                and DOT_MID[0] <= mid <= DOT_MID[1] and (not o1 or not o1.isalnum()):
            l["dot"] = True
            log.append(f"{l['id']}: chấm tròn đầu dòng (ký tự của dòng, gỡ cùng chữ)")
    return log


def _xy_order(boxes: list, texts: list | None = None) -> list[int]:
    """Thứ tự ĐỌC THEO KHỐI của các khung: cắt đệ quy tại khe trống RỘNG NHẤT (ngang hay dọc). Khe giữa hai cột chữ rộng hơn khe
    giữa các dòng trong một cột nên cột trái được xếp hết rồi mới tới cột phải; tiêu đề chạy ngang cả poster tách ra trước nhờ
    khe phía dưới nó. Không có khe nào: trên xuống, trái sang.
    BẢNG (texts = chữ OCR từng khung): một bên khe dọc là CỘT GIÁ TRỊ (>= 2 ô, đa số ô toàn chữ số: giá, giờ, số lượng) và mỗi ô
    đứng cùng hàng với một ô bên kia -> đó là các HÀNG "tên ... giá", xếp theo hàng (tên rồi giá của nó), không theo cột
    (dev2 2963: bảng giá 8 hàng, VLM ghép từng cặp tên - giá)."""
    def valuey(i):
        t = [ch for ch in (texts[i] if texts else "") if ch.isalnum()]
        return bool(t) and sum(ch.isdigit() for ch in t) >= 0.5 * len(t)

    def table(a, b):   # b là cột giá trị của a (hoặc ngược lại)?
        for v, o in ((b, a), (a, b)):
            if len(v) >= 2 and sum(valuey(i) for i in v) >= 0.6 * len(v) and all(
                    any(abs((boxes[i][1] + boxes[i][3]) - (boxes[j][1] + boxes[j][3])) / 2 <= 0.5 * (boxes[i][3] - boxes[i][1]) for j in o)
                    for i in v):
                return True
        return False

    def rows(idx):   # theo hàng: các ô có tâm lệch <= nửa cao là một hàng, trong hàng trái sang
        out, left = [], sorted(idx, key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
        while left:
            i0 = left[0]
            h, cy = boxes[i0][3] - boxes[i0][1], (boxes[i0][1] + boxes[i0][3]) / 2
            row = [i for i in left if abs((boxes[i][1] + boxes[i][3]) / 2 - cy) <= 0.5 * h]
            out += sorted(row, key=lambda i: boxes[i][0])
            left = [i for i in left if i not in row]
        return out

    def gaps(idx, lo, hi):   # các khe trống trên một trục -> [(bề rộng khe, toạ độ cắt)]
        iv = sorted((boxes[i][lo], boxes[i][hi]) for i in idx)
        out, end = [], iv[0][1]
        for a, b in iv[1:]:
            if a > end:
                out.append((a - end, (a + end) / 2))
            end = max(end, b)
        return out

    def rec(idx):
        if len(idx) <= 1:
            return list(idx)
        gy, gx = gaps(idx, 1, 3), gaps(idx, 0, 2)
        by, bx = max(gy, default=(0, 0)), max(gx, default=(0, 0))
        if by[0] <= 0 and bx[0] <= 0:
            return sorted(idx, key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0]))
        lo, hi, cut = (1, 3, by[1]) if by[0] >= bx[0] else (0, 2, bx[1])
        a = [i for i in idx if (boxes[i][lo] + boxes[i][hi]) / 2 < cut]
        b = [i for i in idx if i not in a]
        if not a or not b:
            return sorted(idx, key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0]))
        if lo == 0 and table(a, b):
            return rows(idx)
        return rec(a) + rec(b)
    return rec(list(range(len(boxes))))


def _renumber(L: list, S: list, I: list, logs: list[str]) -> tuple[list, list[str]]:
    """ĐÁNH SỐ Ô L THEO KHỐI (13/10 run 2638; 12/10 run 320d): OCR trả dòng theo từng hàng ngang qua cả poster, hai cột chữ đứng
    cạnh nhau bị đánh số xen kẽ (L5 trái, L6 phải, L7 trái...) và VLM gộp theo số liền nhau -- lời dặn "đừng gộp theo số" không
    giữ được nó. Đánh lại để số liền nhau = cùng một khối; id trong vỏ, quan hệ của chi tiết và log đổi theo. -> (L, logs)."""
    order = _xy_order([l["box"] for l in L], [l.get("ocr") or "" for l in L])
    new = {L[i]["id"]: f"L{k}" for k, i in enumerate(order, 1)}
    if all(a == b for a, b in new.items()):
        return L, logs
    sub = lambda t: re.sub(r"(?<![A-Za-z0-9])L\d+(?![0-9])", lambda m: new.get(m.group(0), m.group(0)), t)   # noqa: E731
    out = [L[i] for i in order]
    for l in out:
        l["id"] = new[l["id"]]
    for s_ in S:
        s_["lines"] = sorted((new.get(i, i) for i in s_.get("lines") or []), key=lambda i: int(i[1:]))
    for c in I:
        if c.get("near"):
            c["near"] = sub(c["near"])
    return out, [sub(t) for t in logs]


ORPHAN_MIN = 0.012   # cụm nét chưa dòng nào nhận: cao >= 1,2% cạnh ngắn ảnh ...
ORPHAN_PX = 80       # ... và >= 80 điểm nét (nhỏ hơn là hạt nhiễu, để nguyên)
ORPHAN_MAX = 8       # nhiều nhất 8 cụm to nhất mỗi poster thành dòng (chữ li ti trên bao bì, biển hiệu trong ảnh nền: để nguyên)


def build(draft: np.ndarray, ink: np.ndarray, product: bool = False) -> dict:
    """Ô của nháp: L# = dòng chữ. ink = nét chữ của Hi-SAM (hisam.strokes), nguồn duy nhất cho biết chữ ở đâu ở mức điểm ảnh
    (14/10: không còn bản xoá FLUX, không còn ô vỏ S / ô chi tiết I -- vỏ và icon là điểm ảnh của nháp).
      1. dòng OCR nguyên khung + số đo nét;
      2. hình học (geo.py): chữ chạy theo cung -> một ô L có đường chân; vùng chữ của bộ dò mà OCR không trả dòng nào -> ô L mới;
      3. CỤM NÉT CHƯA DÒNG NÀO NHẬN (Hi-SAM thấy chữ, OCR không): thành ô L "orphan" để VLM quyết giữ / viết lại / xoá
         (người dùng chốt 14/10). VLM không nhắc tới thì GIỮ nguyên điểm ảnh nháp (lonly.fill_gaps)."""
    H, W = draft.shape[:2]
    ink = np.asarray(ink, bool)
    L = lines(draft)
    A = _lab(draft)
    for l in L:
        # KHUNG DÒNG = KHUNG OCR, nguyên bản (12/10: bỏ hẳn bước nắn khung theo nét mực -- trên nền ảnh chụp nét mực đo thiếu,
        # khung bị cắt hẹp / đẩy mép trái vào mỗi dòng một kiểu: 11-25% dòng, danh sách thò thụt, chữ co).
        l["ocr_box"] = list(l["box"])
        l["shell"] = None
    I: list = []
    from . import geo
    glog, glog2, masks = [], [], {}
    if geo.ENABLED:
        L, I, glog, masks = _geo_lines(L, I, draft, A, ink, [])
        if geo.RUNS:
            L, glog2 = geo.runs(L)
    L, olog = _orphans(L, draft, ink, masks)
    mlog = _lead_dots(L, ink) + _trim_lead(L, ink)
    L, glog = _renumber(L, [], [], glog + glog2 + olog + mlog)
    _typo_notes(L, glog)
    return {"L": L, "S": [], "I": [], "size": [W, H], "by": {m["id"]: m for m in L}, "mask": ink, "shells": {}, "geo_log": glog}


GEO_NAME = {"arc": "cung tròn", "wave": "lượn sóng", "grow": "to dần / nhỏ dần"}


def _typo_notes(L: list, logs: list[str]) -> None:
    """CHỮ KHÓ, ghi cho debug (người dùng 14/10: "debug chưa in các ca typography khó được phát hiện và xử lý thế nào"): mỗi dòng
    l["typo"] = [{"kind", "how"}] -- loại chữ khó và bằng chứng mã đã dùng để nhận ra nó. Cách dựng ghi ở plan["render"]."""
    for l in L:
        notes = []
        lg = [x for x in logs if re.search(rf"(?<![A-Za-z0-9]){l['id']}(?![0-9])", x)]
        if l.get("path"):
            how = []
            if l.get("sag") is not None:
                how.append(f"vùng chữ của bộ dò võng {l['sag']:.2f} cao chữ")
            if l.get("merged"):
                how.append(f"gộp {len(l['merged'])} mẩu OCR {l['merged']}")
            c = l.get("curve") or {}
            if c.get("r"):
                how.append(f"khớp đường tròn bán kính {c['r']:.0f} px, chữ {'trên' if c.get('up') else 'dưới'} vòm")
            notes.append({"kind": f"chữ cong ({GEO_NAME.get(l.get('geo'), l.get('geo') or '?')})", "how": "; ".join(how) or "phần hình học (geo.py)"})
        if l.get("vertical"):
            bw, bh = l["box"][2] - l["box"][0], l["box"][3] - l["box"][1]
            notes.append({"kind": "chữ dọc", "how": f"khung OCR cao gấp {bh / max(1.0, bw):.1f} lần rộng; đọc lại theo {l['angle']:+.0f}° ra {l.get('ocr')!r}"})
        a, ao = float(l.get("angle") or 0.0), l.get("angle_ocr")
        if not l.get("vertical") and not l.get("path") and abs(a) >= 2:
            rule = "từ 10° trở lên là nghiêng thật" if abs(a) >= TILT_SURE else \
                "4-10°: giữ vì có dòng khác nghiêng cùng góc (lệch <= 2,5°) và dòng đủ dài"
            notes.append({"kind": f"nghiêng {a:+.0f}°", "how": f"OCR đo {ao if ao is not None else a:+.1f}°; {rule}"})
        elif ao is not None and abs(ao) >= 2 and abs(a) < 2 and not l.get("vertical") and not l.get("path"):
            why = "dưới 4° là nhiễu đo của OCR" if abs(ao) < TILT_MIN else "4-10° ở dòng ngắn / không có dòng nào nghiêng cùng góc"
            notes.append({"kind": "dựng thẳng", "how": f"OCR đo {ao:+.1f}°, {why}"})
        if l.get("missed") and not l.get("orphan"):
            notes.append({"kind": "dòng OCR sót", "how": "; ".join(x for x in lg if "OCR sót" in x)[:300] or "vùng chữ của bộ dò không có dòng OCR, bộ đọc đọc ra chữ"})
        if l.get("orphan"):
            notes.append({"kind": "cụm nét OCR không đọc", "how": "; ".join(x for x in lg if "cụm nét" in x)[:300] or "nét Hi-SAM ngoài mọi dòng OCR"})
        if l.get("dot"):
            notes.append({"kind": "chấm đầu dòng", "how": "mảng nét đầu dòng nhỏ, tròn, cách chữ; OCR đọc ký tự đầu không phải chữ"})
        if l.get("lead_icon"):
            notes.append({"kind": "khung dời khỏi icon", "how": "; ".join(x for x in lg if "khung OCR ôm" in x)[:300]})
        if l.get("printed"):
            notes.append({"kind": "chữ in trên sản phẩm", "how": "chữ OCR khớp chữ đọc được trên ảnh sản phẩm khách tải"})
        if notes:
            l["typo"] = notes
        else:
            l.pop("typo", None)


def _orphans(L: list, draft: np.ndarray, ink: np.ndarray, masks: dict) -> tuple[list, list[str]]:
    """Mảng nét Hi-SAM không chạm vùng của dòng nào -> gom các mảng gần nhau (đóng hình thái 0,8 cao nét) -> ô L "orphan". Chữ đọc bằng bộ đọc
    của OCR (đọc không ra: để rỗng -- vẫn là ô, VLM nhìn ảnh mà quyết)."""
    import cv2
    from . import lonly, ocr
    H, W = ink.shape
    zone = np.zeros((H, W), bool)
    for l in L:
        zone |= masks[l["id"]] if l["id"] in masks else lonly._region(l, H, W, 0.3)
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
               "angle": 0.0, "caps": caps, "stroke": None, "stroke_contrast": None, "effect": None, "shell": None,
               "missed": True, "orphan": True}
        L = L + [new]
        log.append(f"cụm nét chưa dòng nào nhận: {[round(v) for v in box]} ({px} điểm nét) đọc {text!r} -> {new['id']} (VLM quyết giữ / viết / xoá)")
    return L, log


def _geo_lines(L: list, I: list, draft: np.ndarray, A: np.ndarray, ink0: np.ndarray, shells: list,
               missed: bool = True) -> tuple[list, list, list[str], dict]:
    """CHỮ HÌNH DẠNG KHÓ từ VÙNG CHỮ của bộ dò (ocr.regions: bản đồ của chính RapidOCR trước khi bị ép khung 4 điểm). Đo trên 1307
    vùng của 162 nháp (12/10): mọi vùng hiển thị võng >= 0.3 cao dòng đều là chữ cong thật (dòng thẳng <= 0.24).
      (a) vùng CONG -> MỘT ô L có đường chân, gộp các mẩu OCR và chi tiết nằm trên nó;
      (b) vùng KHÔNG có dòng OCR nào, có nét chữ Hi-SAM (>= 10% vùng) và bộ đọc đọc ra chữ -> ô L mới
          (dòng to bị OCR bỏ ở khâu ép khung: "FLASH SALE", dòng giữa của nhãn dán), thay các ô chi tiết vụn nằm trong nó.
          KHÔNG làm khi khách tải ảnh sản phẩm (missed=False): chữ in trên sản phẩm cũng bị bước xoá gỡ và cũng "không có dòng
          OCR" (12/10: con dấu bánh pía "DAU", chữ dọc trên lon) -- các ô chi tiết + "keep" đang lo đúng phần đó.
    Dòng thẳng đã có ô OCR: không đụng. -> (L, I, log, masks {id ô L tạo ra ở đây: vùng chữ của nó (bool cỡ ảnh)}): build đo
    lại vỏ / chi tiết với các vùng này là CHỮ (xem _layers)."""
    import cv2
    from . import geo, ocr
    H, W = A.shape[:2]
    ink = np.asarray(ink0, bool)   # nét mực = nét chữ của Hi-SAM (14/10; trước: chỗ bước xoá FLUX đã gỡ)
    found, log, masks = [], [], {}
    cen_in = lambda b, m: bool(m[int(min(H - 1, max(0, (b[1] + b[3]) / 2))), int(min(W - 1, max(0, (b[0] + b[2]) / 2)))])   # noqa: E731
    regs = ocr.regions(draft)
    log.append(f"bộ dò: {len(regs)} vùng chữ" if regs else "bộ dò: KHÔNG lấy được bản đồ vùng chữ (phiên bản RapidOCR?) -- phần chữ khó không chạy")
    for R in regs:
        g = geo.region_line(R["mask"], R["pad"])
        if g is None or g["h"] < geo.BAND_MIN_H * min(W, H):   # chỉ chữ hiển thị (cao dòng >= 3.5% cạnh ngắn)
            continue
        k = max(1, int(round(R["pad"])))
        near = cv2.dilate(R["mask"].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
        ys, xs = np.nonzero(near)
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        # cao chữ thật: bề rộng dải nét mực NGANG đường tâm (bỏ 5% hai mép: dấu thanh, đuôi chữ); không có nét (chữ còn trên nền
        # hoặc nằm trong vỏ không đo được): 0.6 cao dòng
        cen = g["center"]
        t = np.gradient(cen, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)
        nrm = np.c_[-t[:, 1], t[:, 0]]
        iy, ix = np.nonzero(ink & near)
        ext = 0.6 * g["h"]
        if len(ix) >= 50:
            P_ = np.c_[ix, iy].astype(np.float64)
            d2 = ((P_[:, None, :] - cen[None, :, :]) ** 2).sum(2)
            j = d2.argmin(1)
            v = ((P_ - cen[j]) * nrm[j]).sum(1)
            lo, hi = np.percentile(v, [5, 95])
            ext = float(np.clip(hi - lo, 0.35 * g["h"], 1.0 * g["h"]))
            cen = cen + nrm * ((lo + hi) / 2)   # tâm dải nét (lõi vùng của bộ dò có thể lệch khỏi tâm chữ)
        on = [l for l in L if cen_in(l["box"], near)]
        # HAI DÒNG NGHIÊNG XẾP CHỒNG dính một vùng (13/10 accept1 g06: "GIẢM" / "30%" trên ruy băng chéo) cũng "võng" như một vòm
        # -> vùng tách được thành các dòng bằng khe không nét thì không phải vòm; đi tiếp nhánh dòng OCR sót
        # ... và vùng NGẮN (dài < ARC_SPAN cao dòng, vài chữ cái) không đủ để nói là cong: đường tâm đi theo hình chữ
        curved = g["sag"] >= geo.ARC_SAG and g["span"] >= geo.ARC_SPAN * g["h"]
        stack = _stacked(near, ink, g["tilt"], ext) if curved else [0]
        if curved and len(stack) == 1:
            found.append({"box": box, "geo": "arc", "path": geo._baseline(cen, np.full(len(cen), ext / 2)), "scale": None, "h": ext,
                          "mask": near, "parts": geo.ARC_BLIND, "sag": round(g["sag"], 2), **geo.circle_of(cen, ext)})
            continue
        if not missed or on or g["span"] < 1.5 * g["h"]:
            continue
        # (b) dòng bị sót: phải là LỚP PHỦ bị gỡ (nét mực chiếm >= 10% vùng; chữ in trên vật / hoạ tiết ảnh còn nguyên trên nền)
        if float((ink & near).sum()) < 0.1 * float(near.sum()):
            continue
        ang = g["tilt"] if abs(g["tilt"]) >= geo.BAND_TILT else 0.0
        for part, pext in _stacked(near, ink, g["tilt"], ext):
            pys, pxs = np.nonzero(part)
            pbox = [float(pxs.min()), float(pys.min()), float(pxs.max() + 1), float(pys.max() + 1)]
            rect = cv2.minAreaRect(np.c_[pxs, pys].astype(np.float32))
            text = ocr.read_region(draft, rect)
            if sum(ch.isalnum() for ch in text) < 2:   # hàng sao, hoạ tiết: bộ đọc không ra chữ -> để các ô chi tiết lo như cũ
                continue
            poly = [[float(a), float(b)] for a, b in cv2.boxPoints(rect)]
            caps = sum(ch.isupper() or ch.isdigit() for ch in text if ch.isalnum()) >= 0.8 * sum(ch.isalnum() for ch in text)
            col = np.median(draft[ink & part], axis=0) if (ink & part).any() else np.array([20, 20, 20])
            new = {"id": f"L{1 + max([int(l['id'][1:]) for l in L] or [0])}", "box": pbox, "ocr_box": list(pbox), "poly": poly, "ocr": text,
                   "base": None, "size": float(pext / (CAP_EM if caps else X_EM)), "color": "#%02x%02x%02x" % tuple(int(x) for x in col),
                   "angle": float(ang), "caps": bool(caps), "stroke": None, "stroke_contrast": None, "effect": None, "shell": None,
                   "missed": True}
            gone = [c for c in I if cen_in(c["box"], part)]
            I = [c for c in I if c not in gone]
            L = L + [new]
            masks[new["id"]] = part
            log.append(f"dòng OCR sót: vùng chữ {[round(x) for x in pbox]} đọc {text!r} + {[c['id'] for c in gone]} -> {new['id']}")
    L, I, lg = geo.apply_arcs(L, I, found, CAP_EM)
    for l in L:
        mk = l.pop("_mask", None)
        if mk is None:
            continue
        # vùng chữ của ô cung = vùng bộ dò + các mẩu đã gộp + dải dọc đường chân (cung nối dài ra ngoài vùng bộ dò)
        band = np.zeros((H, W), np.uint8)
        sz = float(l["size"])
        for (ax, ay), (bx, by) in zip(l["path"][:-1], l["path"][1:]):
            ln = float(np.hypot(bx - ax, by - ay)) or 1.0
            nx, ny = (by - ay) / ln, -(bx - ax) / ln   # pháp tuyến "lên" của hướng đọc
            cv2.fillConvexPoly(band, np.int32([[ax - nx * 0.25 * sz, ay - ny * 0.25 * sz], [bx - nx * 0.25 * sz, by - ny * 0.25 * sz],
                                               [bx + nx * 1.1 * sz, by + ny * 1.1 * sz], [ax + nx * 1.1 * sz, ay + ny * 1.1 * sz]]), 1)
        mk = mk | band.astype(bool)
        masks[l["id"]] = mk
        # MÀU MỰC đo trên chính vùng chữ (số đo màu của mẩu OCR đầu cung lấy trên khung thẳng cắt xiên chữ; cung OCR không đọc
        # được thì không có màu): các điểm bị gỡ đổi màu mạnh nhất = lõi nét, không lẫn quầng
        sel = ink & mk
        if int(sel.sum()) >= 50:
            core = sel & (cv2.erode(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)   # lõi nét, không lẫn viền khử răng cưa
            core = core if int(core.sum()) >= 30 else sel
            l["color"] = "#%02x%02x%02x" % tuple(int(v) for v in np.median(draft[core], axis=0))
    return L, I, log + lg, masks


STACK_VALLEY = 0.2   # hai dòng xếp chồng dính một vùng: giữa vùng có một dải gần như không nét (<= 20% mật độ nét của mỗi dòng)


def _stacked(near: np.ndarray, ink: np.ndarray, tilt: float, ext: float) -> list[tuple[np.ndarray, float]]:
    """Một vùng chữ của bộ dò có thể DÍNH hai dòng xếp chồng sát nhau ("FLASH" trên "SALE", 12/10 dev3 g07: một ô cỡ 244 px).
    Chiếu nét mực lên trục vuông góc hướng đọc: có một KHE (dải gần như không nét) ở khoảng giữa, hai bên đều có nét -> tách tại
    khe thành hai vùng (đệ quy cho ba dòng). -> [(vùng, cao chữ)] trên xuống; không có khe: [(near, ext)]."""
    iy, ix = np.nonzero(ink & near)
    if len(ix) < 200:
        return [(near, ext)]
    t = np.radians(tilt)
    v = -ix * np.sin(t) + iy * np.cos(t)   # toạ độ vuông góc hướng đọc
    v0, v1 = np.percentile(v, [1, 99])
    hgt = v1 - v0
    if hgt < 20:
        return [(near, ext)]
    n = max(8, int(hgt / 3))
    hist = np.histogram(v, bins=n, range=(v0, v1))[0].astype(np.float64)
    hist = np.convolve(hist, np.ones(3) / 3, mode="same")
    best = None
    for k in range(int(0.25 * n), int(0.75 * n) + 1):
        up, dn = hist[:k].max(), hist[k + 1:].max() if k + 1 < n else 0.0
        if hist[k] <= STACK_VALLEY * min(up, dn) and (best is None or hist[k] < hist[best]):
            best = k
    if best is None:
        return [(near, ext)]
    cut = v0 + (best + 0.5) * hgt / n
    yy, xx = np.nonzero(near)
    side = (-xx * np.sin(t) + yy * np.cos(t)) < cut
    out = []
    for sel in (side, ~side):
        m = np.zeros_like(near)
        m[yy[sel], xx[sel]] = True
        pv = v[(v < cut) if sel is side else (v >= cut)]
        if m.sum() < 30 or len(pv) < 50:
            return [(near, ext)]
        lo, hi = np.percentile(pv, [5, 95])
        out += _stacked(m, ink, tilt, float(hi - lo))
    return out


def attach_geo(M: dict, draft: np.ndarray, ink: np.ndarray, product: bool = False) -> list[str]:
    """M đọc từ plan.json: chạy phần đo chữ theo cung trên các ô ĐÃ LƯU (không OCR lại) -> đổi M["L"], M["by"] tại chỗ, trả log.
    Dùng cho đo / dựng lại tại máy."""
    for l in M["L"]:
        l.setdefault("poly", [[l["box"][0], l["box"][1]], [l["box"][2], l["box"][1]], [l["box"][2], l["box"][3]], [l["box"][0], l["box"][3]]])
    L, I, log, masks = _geo_lines(M["L"], [], draft, _lab(draft), ink, [])
    M["L"], M["I"] = L, []
    M["by"] = {m["id"]: m for m in L}
    return log


def mark_printed(M: dict, product_ocr: list | None) -> list[str]:
    """CHỮ IN TRÊN SẢN PHẨM khách tải (13/10, người dùng: PHẢI GIỮ, không thay bằng bất cứ gì): dòng của nháp có chữ OCR gần
    trùng một dòng chữ đọc được trên ẢNH SẢN PHẨM (FLUX chép sản phẩm vào nháp rất sát) -> l["printed"] = True. Dòng đó luôn
    là "keep" (validate), VLM được báo trong bảng mô tả. -> id các dòng."""
    words = [{"text": w} for w in product_ocr or [] if w and w.strip()]
    out = []
    for l in M["L"] if words else []:
        if _client_like(l.get("ocr") or "", words) is not None:
            l["printed"] = True
            out.append(l["id"])
    return out


def _hex_lab(hx: str) -> np.ndarray:
    return _lab(np.uint8([[[int(hx[i:i + 2], 16) for i in (1, 3, 5)]]]))[0, 0]


def public(M: dict) -> dict:
    """Ô để ghi JSON."""
    return {"L": [{k: m[k] for k in ("id", "box", "ocr_box", "ocr", "size", "color", "angle", "shell", "curve", "runs", "geo", "path", "scale", "missed", "orphan", "printed", "dot", "lead_icon", "vertical", "angle_ocr", "sag", "merged", "typo", "poly") if k in m}
                  for m in M["L"]],
            "S": M["S"], "I": M["I"]}


def marked_image(img: np.ndarray, M: dict):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    s = max(12, img.shape[1] // 75)
    try:
        f = ImageFont.truetype("arial.ttf", s)
    except OSError:
        try:
            f = ImageFont.truetype("DejaVuSans.ttf", s)
        except OSError:
            f = ImageFont.load_default()
    items = [(m, (0, 140, 255), 4, "tl") for m in M["S"]] + [(m, (230, 20, 20), 2, "tl") for m in M["L"]] + \
            [(m, (255, 150, 0), 2, "br") for m in M["I"]]
    for m, col, wd, corner in items:
        if m.get("path"):   # ô chữ cong: đường chân thay cho khung (khung thẳng của một vòm trùm cả thứ nằm dưới vòm)
            pt = [tuple(q) for q in m["path"]]
            d.line(pt, fill=col, width=4)
            tw, th = d.textbbox((0, 0), m["id"], font=f)[2:]
            tx, ty = pt[0][0], max(0, pt[0][1] - th - 4)
            d.rectangle([tx, ty, tx + tw + 2, ty + th + 2], fill=col)
            d.text((tx + 1, ty), m["id"], fill=(255, 255, 255), font=f)
            continue
        x0, y0, x1, y1 = m["box"]
        x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)   # ảnh debug không được làm hỏng phiên bản
        d.rectangle([x0, y0, x1, y1], outline=col, width=wd)
        tw, th = d.textbbox((0, 0), m["id"], font=f)[2:]
        tx, ty = (x0, max(0, y0 - th - 2)) if corner == "tl" else (max(0, x1 - tw - 2), min(img.shape[0] - th - 2, y1 + 1))
        d.rectangle([tx, ty, tx + tw + 2, ty + th + 2], fill=col)
        d.text((tx + 1, ty), m["id"], fill=(255, 255, 255), font=f)
    return im


# ---------------------------------------------------------------------------------------------------- VLM
SYSTEM = """You are a senior graphic designer finishing a poster. An image model drew the DRAFT from the PROMPT below. Its layout \
and pictures are good; its overlay layer -- text, cards, pills, bands, buttons, icons, avatars, stars -- is rough and misspelt. \
An eraser removed that whole layer (what is left is the BASE) and code measured what was removed as SLOTS. You decide what each \
slot is and how it looks; the renderer draws your decision exactly inside the slot. Final poster = BASE + your slots, every \
slot where the model drew it, at the size it drew.

THE RULES THAT MATTER MOST
1. The LOOK comes from the draft, the WORDS come from the client texts. Never take spelling from the image or the OCR \
(the model draws TRƯỞNG for TRƯƠNG); copy client words character for character with every Vietnamese diacritic, and never \
swap a word for a synonym (nghìn stays "nghìn").
2. FIT. The room is what the model drew, never what the client wrote. Each line lists how many characters the model wrote \
there: that is the most the line holds (real letters are wider than the model's thin ones); fewer is fine. A client text \
longer than its room is REWRITTEN SHORT in the client's own words until it fits at the drawn size: keep the key words, drop \
the rest. Example: "Hỗ trợ nấu cơm chín đều, giữ độ mềm dẻo và giúp việc chuẩn bị bữa ăn hằng ngày trở nên thuận tiện" drawn \
as 2 lines of about 22 characters -> "Cơm chín đều, mềm dẻo<br>mỗi bữa ăn". Never the full sentence at a smaller size, never \
more lines than the model drew, and the same inside a pill / button / badge ("Tặng buổi tư vấn đầu cho khách mới").
3. FACTS -- names, prices, numbers, dates, phones, addresses, emails, links -- are never changed, abbreviated or invented: \
exact, with their unit (35.000đ, 30 phút, 50%, 15-25 triệu; "Quận 3", never "Q.3"), or left out. A fact the model drew \
that is in no client text is never written: skip that slot.
4. Every slot id appears in exactly ONE op; never leave a slot out ("skip" drops it). Always return ops: an empty plan is a \
blank poster.
5. Each client text is written ONCE. The model often repeats a line (a date, a price, a badge label): write it in the slot \
that fits best -- the biggest one for a headline -- and skip the copies.
6. The rows of one list look exactly alike and keep the client's order. Items the model drew no row for are left out and \
listed in "missing", never crammed into another row. A clean, well-fitted poster beats a complete one.

IMAGES: the message names each one. DRAFT = what the model drew. SLOTS = the draft with every slot marked (blue S#, red L#, \
orange I#); ZOOM = enlarged parts of it. BASE (when given) = the erased poster your slots are drawn on: what is really behind \
each slot (shells the eraser kept are still there; a skipped slot shows this background).

SLOTS, as listed after the images:
- S# SHELL: a card / pill / badge / band / button / panel; box, fill color, corner radius, the L# / I# inside it.
- I# DETAIL: a small non-text element (icon, avatar, star row, check, logo, ornament); box, color, its shell, the nearest line.
- L# TEXT LINE: the garbled OCR letters, drawn text size in px, number of characters drawn (its room), color, its shell. \
L# numbers follow the reading order block by block: a column is numbered top to bottom before the column beside it starts.

ONE OP PER SLOT
- S#: "shape" -- box_style = CSS matching the draft (start from the measured fill and radius; add border / shadow / gradient / \
transparency the draft shows), html "". "keep" -- the draft's own pixels are put back (old letters of the L# lines inside are \
wiped; you still write those lines): for a shape a CSS box cannot draw (ring, seal, starburst, ribbon, tilted or irregular \
sticker) drawn cleanly in ONE flat color, for a shell that is part of the picture (printed on a product or object), and when \
the S# box is clearly bigger than the shell you see. Not for glossy, gradient or see-through shells. "skip" -- junk only: a \
shell holding a line you write is never skipped (the text would sit bare).
- I#: "icon" -- html = ONE helper tag without a size (it fills the slot): <i-icon name="<lucide name>" color="#hex"></i-icon>; \
a person -> name "circle-user"; a star row -> <i-stars n="5" fill="#f5b301" empty="#d9d9d9"></i-stars>. "text" -- the detail is \
really a big character or number OCR missed (a step number in a circle). "keep" -- anything that is NOT lettering and that the \
draft drew well: a frame line, divider, ornament, logo mark, a graphic printed on a product, and any piece of the PHOTO (the \
eraser redrew it a little differently; "keep" gives the draft's back). Unsure between skip and keep for a non-letter -> "keep". \
"skip" -- garbled letter fragments, duplicates, a bullet / icon whose row you dropped.
- L#: "text" -- html = the text; one op may take several CONSECUTIVE lines of ONE block stacked one under the other \
(["L5","L6"], <br> between the lines), never lines standing side by side (only two pieces of one drawn line that OCR cut \
apart go together). "keep" -- text printed on a product, package, screen, dial or object in the photo: never rewrite or add \
words there. "skip" -- junk / duplicates.
- "keep" never applies to POSTER lettering however decorated (a headline on a ribbon, 3D or metallic letters, badge text): \
the draft's letters are misspelt, so it is always rewritten as a text op.
- CURVED LINE slot (the list says so; SLOTS shows its curve): one line of lettering along an arch, a smile or a ring. ONE \
text op with that single slot, one line, no <br>: the renderer lays your letters along the measured curve. Never in the same \
op as another slot; a ring with text on top and at the bottom is two slots, two ops.
- A line "drawn at N degrees" is tilted and the renderer rotates it itself: never write transform rotate / skew anywhere. \
Tilted lines stacked in one sticker / ribbon / tag are ONE op.
- "OCR missed this line": real lettering of the draft (usually a big decorated headline). Look at the DRAFT to see which \
client text it is and write it there.
- "row mark in front": a bullet / check / small icon sits at the left INSIDE that line's slot. Start the row's html with \
the mark and a space -- "• " for a dot, <i-icon name="check" color="#hex"></i-icon> for a check or real icon (drawn at text \
size) -- the same mark on every row of the list. Never a ring or circle icon for a dot. A row whose mark the BASE still \
shows gets only the text.

TEXT
- Write each line at the size measured for its slot, with the model's line breaks; fine print and footers too, never \
smaller "to be safe". Set "client": "T<i>" on every op writing (part of) a client text.
- One client text drawn over several stacked lines (MENU / TRÀ SỮA) is one op with all those slots. Read the garbled OCR to \
find them: a short line under a line that ends its sentence is that text's second line.
- Client texts marked OURS are filler we wrote (headline, tagline, button, section label): shorten, reword or drop them \
freely; they carry no facts.
- Text the model invented (in no client text): write what it was meant to say from its role and the PROMPT, a natural line of \
about the same length in the poster's language. Never copy garbled OCR. Never invent a fact.
- Star ratings are <i-stars n="..."></i-stars>, never star characters, and never "5 sao" / "5.0" next to the stars.
- html: inline spans only (color / weight / italic / size for emphasis), inline CSS, no <img>, URLs or scripts; sizes in poster \
px. The slot is a flex box centered both ways; left-aligned text needs box_style "justify-content:flex-start; text-align:left".

LISTS AND GROUPS
- One item per drawn row, in the client's order, each starting with a capital letter; item / price rows keep their pairing. \
More rows than items: fill top to bottom and skip the LAST extra rows with their marks, never a hole in the middle. More \
items than rows: one item per row, the rest go to "missing".
- GROUPS -- you declare, the renderer obeys. Ops that belong together carry the same "group" name ("benefits", "menu-names", \
"menu-prices", "badges", "headline"), the same "group_kind" and "align":
  "list" = separate items, one per row: drawn on ONE margin at ONE size, font, weight and color.
  "flow" = ONE sentence the draft broke over several lines that you did not merge into one op: all lines drawn alike.
  "align" = "left" | "center" | "right", as the draft aligned them.
  Write the ops of a group identically yourself (same font key, weight, size, color, case, letter-spacing; only the words \
change). Every list and every multi-line sentence that is not one merged op MUST be a group. Lines the draft drew at clearly \
different sizes (a small "GIẢM" beside a huge "50%", a big headline over a small tagline) are NOT a group: "group": null.
- Icons in front of list rows are ONE set: every row you write keeps exactly one, same style and color.

SHELLS
- A line inside an S# shell, or on a shell the BASE still shows, never gets its own background, border or shadow (no shell \
inside a shell). A list row or body line never gets a background bar: use a text-shadow for contrast.
- Lost shell: the DRAFT shows a button / pill / band behind a line, the BASE does not and no S# covers it -> recreate it in \
that text op's box_style (background, border-radius, a little padding, the draft's colors). A call-to-action is never bare text.
- A badge / seal the BASE still shows is never left empty: when its drawn lines match no client text, write ONE short phrase \
from the client texts there (the key number or benefit) and skip the other lines.
- Badge / sticker with a big middle line: the key word or number big in the big slot, the rest small in the small slots \
(Giảm / 50%); leftover slots are skipped, not filled with made-up words.

STYLE -- the poster should look like the draft with clean text, not like another design
- One type system: style.display_font for headlines, style.text_font for the rest (keys from the catalog); <i-font key="..." \
weight="700">...</i-font> only where a line needs another font. Pick fonts whose family matches the draft's letterforms \
(condensed, rounded, geometric, serif, script, display) and whose letter width is within about 0.05em of the width listed for \
the line (caps width for all-caps lines): a wider font gets shrunk, a narrower one looks lost.
- Match each line's weight, case, italic and tracking: a heavy headline stays heavy (800-900), a script stays a script, \
slanted stays italic, a wide headline is never set condensed. The measured weight / strokes / shadow are hints; the image \
wins. Effects with inline CSS, as strong as the draft: -webkit-text-stroke or text-shadow for outlines, text-shadow for \
shadows and glows, background-clip:text with a linear-gradient for gradient or metallic letters.
- Colors: put the draft's 3-5 main colors in style.palette and take every text, shell and icon color from it or from the \
measured colors; no new hues. Same role, same color and font.
- Contrast: every text must read clearly on what is really behind it (check the BASE); on a busy or mid-tone background add \
a subtle text-shadow.
- Hierarchy and alignment as the draft shows them: headline > price / offer / call-to-action > body > fine print; a centered \
block stays centered, a left column stays left.

OUTPUT: one JSON object
{"style": {"display_font": "<key>", "text_font": "<key>", "palette": ["#hex"], "notes": "..."},
 "ops": [{"id": "o1", "slots": ["L1"], "kind": "text|keep|skip|shape|icon", "client": "T0" | null, "html": "...",
          "box_style": "...", "group": "name" | null, "group_kind": "list" | "flow" | null,
          "align": "left" | "center" | "right" | null, "why": "a few words"}],
 "missing": ["T5"], "notes": "..."}

FONT CATALOG (key: family, class): {fonts}

LUCIDE ICONS (common ones by theme; any other Lucide name also works -- the renderer maps an unknown name to the nearest \
icon): {icons}"""

# Lời dặn designer bản đầy đủ trước 13/10 (dài gấp đôi, kèm cả 1544 tên icon): TEXTFIX_PROMPT=v1 để so trên máy chủ.
SYSTEM_V1 = """You are a senior graphic designer finishing a poster. An image model was given the PROMPT below and drew the DRAFT. \
Its layout, composition and pictures are good; its overlay layer is broken: text is garbled, icons / avatars / stars are \
malformed, cards / pills / bands are rough. An eraser removed the WHOLE overlay layer: every text, card, pill, band, button, \
icon, avatar and star -- only the picture remains (the BASE). Code measured exactly what was removed and split it into SLOTS. \
You decide what each slot is and how it should look; the renderer draws your decision exactly inside that slot. The final \
poster = the BASE + your slots. Keep the model's layout: every slot stays where the model drew it, at its size.

AESTHETICS COME FIRST. For layout and look, follow the draft. A clean, consistent, well-fitted poster beats a complete one. \
But the WORDS come from the client texts, never from the image.

IMAGES: the message names each image. DRAFT = what the model drew. SLOTS = the draft with every slot marked (blue S#, red \
L#, orange I#); ZOOM images are enlarged parts of it. BASE (when given) = the erased poster the renderer draws your slots on: \
what will really be behind each slot (shells the eraser kept are still there; skipped slots leave this background).

SLOTS:
- S# SHELL: a card / pill / badge / band / button / panel the model drew; measured box, fill color, corner radius, and which \
L# / I# lie inside it.
- I# DETAIL: a small non-text element (icon, avatar, star row, check mark, bullet, logo, emoji...); measured box and color, \
the shell it is in, the nearest line.
- L# TEXT LINE: what OCR read (garbled), the drawn text size in px, how many characters the model wrote there (its CAPACITY), \
the measured color, the shell it is in.

FOR EACH SLOT, ONE op (a text op may take several consecutive lines of one text block):
- S#: kind "shape", slots ["S1"], box_style = CSS for the shell matching the draft (start from the measured fill and radius; \
add border / shadow / gradient / transparency if the draft shows it), html "". Or kind "skip" if the shell is junk, or kind \
"keep" if it is part of the picture (a graphic printed on a product, package or object: its original pixels are restored).
  A shell whose SHAPE a CSS box cannot draw (a ring or double ring, a seal, a starburst / scalloped badge, a ribbon, a tilted or irregular sticker) and that the draft drew cleanly in ONE flat color: kind "keep" -- the renderer puts the draft's own shell back and wipes the old letters of the L# lines inside; you still write those lines as text ops. Not for glossy, gradient or see-through shells, and not when the shell carries lettering outside its L# lines (curved text along its edge): those are "shape". Also "keep" when the S# box is clearly bigger than the shell you see in the draft (the slot swallowed a \
frame line or a band next to the pill): a "shape" would paint the whole box, "keep" restores exactly what the draft had. \
A shell that holds a line you write is never "skip": the text would sit bare on the background.
- I#: kind "icon", slots ["I3"], html = ONE helper tag without a size (it fills the slot): <i-icon name="<lucide name>" \
color="#hex"></i-icon>; a person avatar -> <i-icon name="circle-user" color="#hex"></i-icon>; a star row -> <i-stars n="5" \
fill="#f5b301" empty="#d9d9d9"></i-stars>. Pick the icon from what the model drew and what the nearby text means. Or kind \
"text" (html = the text) if the detail is really a big character or number OCR missed (a step number "1" in a circle), or kind "skip" for junk or duplicates, or kind "keep" for a detail that belongs to the picture (a logo or graphic printed on a product, \
package or object, a sign in the scene: its original pixels are restored).
  "keep" is also right for anything that is NOT lettering and that the draft drew well: a frame line, a divider, an ornament, a sparkle, a logo mark, and any piece of the PHOTO itself (the eraser redrew part of the picture a little differently: "skip" would leave the eraser's version, "keep" gives the draft's back). Unsure between "skip" and "keep" for something that is not lettering -> "keep". "skip" is for garbled letter fragments, duplicates and bullets / icons whose row is dropped; "icon" is for a bullet / check / icon / avatar / star row (redrawn clean).
- L#: kind "text", slots ["L5"] or ["L5","L6"], html = the text. Or kind "keep" for text printed on a product, package, screen \
or object in the photo (its original pixels are restored, never redraw such text), or kind "skip" for junk / duplicates.
- A CURVED LINE slot (the list says so; SLOTS shows its curve instead of a box) is one line of lettering that follows an arch, \
a smile or a ring. Write it as ONE text op with that single slot, one line, no <br>, no transform / rotate of your own: the \
renderer lays your letters along the measured curve. Style it like any headline (font, weight, color, outline). A ring with \
text on top and at the bottom is two slots, each its own op. Never put a CURVED LINE slot in the same op as any other slot: \
if the list shows more slots than the phrase needs around the curve, write the whole phrase on the curve and "skip" the others.
- A line listed "drawn at N degrees" is tilted: the renderer rotates the op to the measured angle itself. Never write \
transform: rotate / skew anywhere (box_style or html) -- it would be turned twice. Tilted lines stacked in one sticker / \
ribbon / tag are ONE op (<br> between them).
- A slot marked "OCR missed this line" is real lettering of the draft (usually a big decorated headline): look at the DRAFT to \
see which client text it is and write that text there; never skip it because its OCR letters look wrong.
- "keep" never applies to POSTER lettering, however decorated it is \
(a headline on a ribbon or banner, curved, 3D or metallic letters, badge and sticker text): the draft's letters are often \
misspelt, so it is always rewritten as a text op with the client's spelling.
- A product's screen, display, dial, label or packaging text is part of the picture: "keep" it (or "skip"), never write \
words or numbers there.
- Every slot id appears in exactly ONE op -- never leave a slot out (use "skip" to drop it).

TEXT
- Decide what the model meant to write on each line and write it at the size it drew, keeping its line breaks (<br> \
between the lines of a multi-line op).
- FIT -- the room is what the model drew, never what the client wrote. Each line lists how many characters the model wrote \
there: that is the most the line holds (the model's letters are thin; real letters are wider), fewer is fine. A client text \
longer than its room is REWRITTEN SHORT, in the client's own words, until it fits at the drawn size: keep the key words, \
drop the rest. A 20-word description drawn as two short lines becomes one short phrase ("Hỗ trợ nấu cơm chín đều, giữ độ \
mềm dẻo và giúp việc chuẩn bị bữa ăn hằng ngày trở nên thuận tiện" in a 2 x 22-character slot -> "Cơm chín đều, \
mềm dẻo<br>mỗi bữa ăn"). Never the full sentence at a smaller size, never extra lines the model did not draw.
- Facts (names, prices, numbers, dates, phones, addresses, emails, links) are never changed, abbreviated or invented -- keep \
them exact (Quận 3 stays "Quận 3", never "Q.3") or leave them out; the unit belongs to the number (15-25 triệu, 35.000đ, \
30 phút, 50%): never drop it. Never swap a client word for a synonym (nghìn stays "nghìn", never "ngàn"). Set \
"client": "T<i>" on ops writing (part of) a client text; list client texts the model did not draw, or that you dropped, in \
"missing".
- Client texts marked OURS are filler copy we wrote (a headline, tagline, button or section label the client did not give): \
you may shorten, reword or drop them freely so the poster fits and looks right; they never carry facts. All other client \
texts follow the rules above and below.
- SPELLING: copy client words character for character, with every Vietnamese diacritic. The draft's letters are often \
wrong (TRƯỞNG or TRƯỜNG drawn for TRƯƠNG, Thứ Bẩy for Thứ Bảy): never take spelling from the image or the OCR.
- ONCE: each client text is written once. The image model often repeats a line (the same date, price or badge label twice): \
write it in the slot that fits it best and skip the copies. Never write a word that is in no client text inside an op that \
claims a client text.
- One client text drawn over several stacked lines (MENU / TRÀ SỮA): one op with all those slots, <br> between the lines. \
Read the OCR letters to find those lines even when garbled: a short slot below a line that ends its sentence ("chainn vi" \
under "Coffee rang moc") is that text's second line, never the start of another text.
- BLOCKS: L# numbers follow the reading order block by block -- a column of text is numbered top to bottom before the \
column beside it starts. The slots of one op are consecutive lines of ONE block, stacked one under the other; lines side by \
side belong to different ops. (Only pieces of ONE drawn line that OCR cut in two -- same height, same size, a word gap \
apart -- go together.)
- List items, steps and item / price rows keep the client's order and pairing (each name with its own price, steps \
1, 2, 3 in order), even where the draft drew them in another order.
- Text the model invented: write what it was meant to say from its role, position and the PROMPT, a natural line of about the \
same length in the poster's language (usually Vietnamese). Never copy garbled OCR. Commit; do not hedge. But a FACT the \
model drew that is in no client text (an address, phone, price, date, percentage, website) is never written: skip that slot.
- Star ratings are always <i-stars n="..."></i-stars>, never star characters (the fonts have no star glyph). The stars \
ARE the rating: never write "5 sao" / "5 stars" / "5.0" next to a star row (a client text that only says "5 sao" is shown \
as the stars alone).
- html: inline spans only (color / weight / italic / size for emphasis), inline CSS only, no <img>, no URLs, no scripts. Sizes \
in px of the poster (as listed per line). The slot is a flex box centered both ways; for left-aligned text use box_style \
"justify-content:flex-start; text-align:left".
- One consistent type system: style.display_font for headlines, style.text_font for the rest (font keys from the catalog); use \
<i-font key="..." weight="700">...</i-font> only when a line needs another font. Colors: follow the draft; text must stay \
clearly readable on what is behind it (on the BASE the shell under a line may be gone if you skip it).

LOOK: REPRODUCE THE DRAFT'S STYLE -- the final poster should look like the draft with clean text, not like a different design.
- Typography per line: match what the model drew (look at the images; the measured weight / caps / strokes / shadow / outline \
are hints, the image wins when they disagree). Pick catalog fonts whose family matches the draft's letterforms (condensed, \
rounded, geometric, serif, script, display), then reproduce the rest with \
inline CSS: font-weight, font-style:italic, text-transform, letter-spacing for tracked headlines, -webkit-text-stroke or \
text-shadow for outlines, text-shadow for drop shadows and glows, background-clip:text with a linear-gradient for gradient or \
metallic (gold / silver) lettering. Keep effects as strong as in the draft -- no weaker, no stronger.
- Color harmony: put the draft's 3-5 main colors in style.palette (dominant background tone, 1-2 accents, dark and light text \
tones). Take every text, shell and icon color from that palette or from the measured colors; no new hues. Lines with the same \
role share the same color and font.
- Hierarchy: headline > price / offer / call-to-action > body > fine print, by size, weight and color contrast, as the draft \
shows it. Never make body text compete with the headline.
- Contrast: every text needs strong contrast with what is really behind it (check the BASE): light text on dark, dark on light; \
when the background is busy or mid-tone, add a subtle text-shadow or a shell, in the poster's palette.
- Lost shells: if the DRAFT shows a button, pill, badge or band behind a line but the BASE does not (the eraser removed it and \
no S# slot covers it), recreate it in that text op's box_style (background, border-radius, a little padding) in the draft's \
colors -- a call-to-action must never end up as bare text. Only then: a line inside an S# shell, or on a shell the BASE still shows, never gets its own background or border (no card inside a card).
- Alignment: keep the draft's alignment per block -- a centered headline stays centered, a left column stays left.
- Size: write every line at the size measured for its slot; fine print, labels and footers too -- never smaller "to be safe".
- Lists: keep the draft's list structure -- one item per row, each row keeps its own text next to its bullet / check / icon; \
never merge list rows into one block and never leave a bullet or icon without its text. Every list row (bullet / feature / step) starts with a capital letter, and all rows of one list carry the same "group" name with "group_kind": "list".
- Row marks: a line listed with "row mark in front" has a bullet / check / small icon at its left INSIDE the slot. Start that \
row's html with the mark and a space: "• " for a dot, <i-icon name="..." color="#hex"></i-icon> for a real icon or check \
mark (drawn at the size of the text) -- the same kind of mark on every row of the list, never a big ring for a dot. A row \
without that note whose mark the BASE still shows gets only the text; a mark the eraser removed that has no I# slot is \
drawn the same way, at the start of the row's html.
- Slanted (italic / oblique) lettering in the draft stays italic; tracked (widely spaced) headlines keep their letter-spacing.
- Weight and family: a heavy / black headline stays heavy (font-weight 800-900 in a font that has it), a script stays a \
script font, a wide headline is never set in a condensed font (or the reverse). Unset weight = the weight measured on the draft.
- Letter width: each line lists how wide the drawn letters are (em per character) and the catalog lists each font's width \
(mixed case / caps). Pick a font within about 0.05em of the line (caps width for all-caps lines): a wider font does not fit \
the slot and gets shrunk, a narrower one looks lost in it.
- Same role, same look: rows of one list / menu / table, the items of one card set, the labels of one badge set share ONE \
font, size, weight and color -- write the same font-size on all of them (the smallest that fits), never one row big and the \
next small. Left-aligned columns: "justify-content:flex-start; text-align:left" on every row so the left edges line up.
- GROUPS -- you decide, the renderer obeys. Slots that belong together get the same "group" name on their ops (e.g. \
"benefits", "menu-names", "menu-prices", "badges", "headline"), with the same "group_kind" and "align" on every op of the group:
  * "group_kind": "list" -- separate items of one list / menu / feature set / step set / badge set, one item per row. The \
renderer puts every row on ONE margin and draws all rows exactly alike: ONE size (the size that fits the longest row), font, weight and color.
  * "group_kind": "flow" -- ONE sentence or phrase that the draft broke over several lines and that you did not merge into a \
single op (a two-line headline, a paragraph). The renderer draws all its lines exactly alike: ONE size, font, weight and color.
  * "align": "left" | "center" | "right" -- how the draft aligned the lines of the group.
  Write every op of a group IDENTICALLY yourself: the same font key, the same weight, the same size, the same color, the same \
case and letter-spacing in html / box_style -- copy the first op's styling to the others and change only the words.
  Every list and every multi-line sentence that is not one merged op MUST be a group. Lines the draft deliberately drew \
different (a huge "FLASH SALE" over a small tagline, a big price next to its small unit) are NOT a group: "group": null.
- More rows than items: the image model sometimes drew more list rows or bullets than there are client items. A long item \
may continue on the next row (that row's bullet skipped); otherwise skip the extra rows AND their bullets / icons. \
Fill the rows top to bottom in order and skip only the LAST extra rows -- never leave an empty row in the middle of a list \
(a garbled or repeated row in the middle is just the next row of the list).
- More items than rows, or one client text holding several items ("A, B, C, D") where the draft drew bullets: ONE item per \
drawn bullet, each short enough for its row, first letter capital, in the client's order. Never pack two or three items \
into one row and never leave a drawn bullet empty while another row is crammed; items that get no row are dropped \
("missing") or two short ones share a row only where the draft drew them so.
- Text in a pill / button / badge / band: the shell fixes the room -- FIT applies strictly (facts exact: "Tặng buổi tư vấn \
đầu cho khách mới"), never a full sentence shrunk to fine print inside its shell.
- A badge / seal / shell the BASE still shows is never left EMPTY: when the lines the model drew inside it are invented \
or do not match any client text, write ONE short phrase taken from the client texts there (the key number or benefit: \
"14 ngày", "1 tỷ lợi khuẩn", "Tặng bình giữ nhiệt") and skip the other lines -- an empty badge is worse than a short one.
- No background bars: a list row or a body line never gets its own background / box in box_style. For contrast on a busy \
picture use a text-shadow (or a lighter / darker text color from the palette). Only a call-to-action or offer whose pill \
the draft showed gets a recreated shell.
- List icons are ONE set: when the rows of a list have an icon in front, EVERY row you write keeps exactly one icon op -- \
never skip the icon of a row you write, and when the model drew two small marks in front of one row, put both I# slots \
in ONE icon op. Same style and color on every row.
- Badges / coins / stickers with a big middle line: put the key word or number of the client text big in the big slot and \
the rest small in the small slots (Giảm / 50%; Tặng / quà); skip slots left over rather than filling them with made-up words.
- Never a shell inside a shell: when the BASE still shows a pill / card / band behind a line, or its S# slot is drawn as a \
shape, the text op gets NO background, border or box-shadow of its own.

OUTPUT: one JSON object
{"style": {"display_font": "<key>", "text_font": "<key>", "palette": ["#hex"], "notes": "..."},
 "ops": [{"id": "o1", "slots": ["L1"], "kind": "text|keep|skip|shape|icon", "client": "T0" | null, "html": "...",
          "box_style": "...", "group": "name" | null, "group_kind": "list" | "flow" | null,
          "align": "left" | "center" | "right" | null, "why": "a few words"}],
 "missing": ["T5"], "notes": "..."}

FONT CATALOG (key: family, class): {fonts}

LUCIDE ICON NAMES: {icons}"""

REVIEW = """You are the same designer, now the ART DIRECTOR reviewing the rendered poster before it goes to the client. Your first pass \
was a plan made without seeing the result; now you SEE it. Use that: judge the PROOF with your eyes, not the plan.

You get: the DRAFT (the look to reach), the PROOF (exactly what the client will see), PROOF-IDS (the same with every op's \
box and id), COMPARE crops (draft left, proof right, enlarged, for the busy areas), the rendered ops (what each wrote; \
drafted size -> rendered font size, weight, font), the CODE CHANGES (what the engine changed in your plan on its own, see \
below) and the ERRORS a checker measured.

CODE CHANGES are deliberate and must not be undone: diacritics corrected to the client's spelling, duplicate lines skipped, \
invented words removed, cut-short client texts restored, font keys mapped, left edges of a column aligned, icons of one row \
/ column set to one size, lines of one role set to the size of the smallest one. When a change made something look worse (a \
whole list became small because one row was long), fix the cause (shorten or rebreak that row, use a narrower font for the \
role) -- do not reverse the change.

Look at the PROOF as a whole and then area by area against the DRAFT, and write down every flaw a careful designer would not \
ship. Typical flaws:
- an orphan: a price with no item on its row, a bullet / icon / number with no text, a lone word;
- an empty or under-filled shell: a badge, coin, pill, button or card the draft filled with big text that now holds tiny \
text or nothing;
- text not centred in its shell, or touching its edge;
- one role in different sizes, weights, colors or fonts (list rows, prices, card titles, badge labels);
- icons of one set in different sizes or styles;
- text that is hard to read on what is behind it;
- style that drifted from the draft: font family and width, weight, case, italic, effects, colors;
- awkward line breaks, crowding, text much smaller than the draft drew it.
Then fix each flaw with patches, and also every ERROR.

Your tools (you may use any of them):
- rewrite html / box_style of an op: font, weight, size, color, effects, line breaks, letter-spacing, padding, alignment;
- MERGE slots: one op over several slots (e.g. all lines of a coin or badge in one op, the key word or number big on its own \
line) -- the op is drawn over the union of its slots; stacked slots get one line each with <br> between them, never all \
on one line;
- move a client text to a better slot, split one over two slots, or give a slot to another op;
- turn a slot into a shape (recreate a lost pill / band behind text), an icon, keep or skip;
- delete an orphan or junk op that carries no client text.
Change only what is wrong -- ops that look right stay untouched. Facts stay exact; step / list numbers stay in order (1, 2, 3 -- \
never renumber a step to make room for another text). Never drop a client text: a poster \
missing its words is worse than any flaw. Slots never move.

OUTPUT one JSON object: {"findings": [{"ops": ["o7"], "flaw": "...", "fix": "..."}], "patches": [{"id": "o6", "slots": \
[...], "kind": "...", "html": "...", "box_style": "...", "client": ..., "delete": false}], "why": "..."} -- findings first \
(what you see), then one patch per fix; a patch lists only the fields it changes; "delete": true removes that op; a patch \
with a NEW id (e.g. "n1") and its slots adds a new op. Empty "patches" only when the PROOF is already as good as the draft's \
look allows."""


def _rewritten(op: dict, ops: list[dict], patches: dict) -> bool:
    """Chữ của lệnh op (bỏ dấu) có nằm trong một lệnh khác (sau vá) không: vòng duyệt gộp / chuyển chữ sang lệnh khác rồi mới xoá."""
    mine = _flat(_plain(op.get("html")))
    for o in ops + [p for p in patches.values() if p.get("id") not in {x["id"] for x in ops}]:
        if o.get("id") == op["id"]:
            continue
        q = patches.get(o.get("id")) or {}
        if q.get("delete"):
            continue
        if mine and mine in _flat(_plain(q.get("html", o.get("html")))):
            return True
    return False


def _px(b) -> str:
    return f"{b[2] - b[0]:.0f}x{b[3] - b[1]:.0f}px"


def _fits(w: float, m: dict, n: int = 4) -> list[str]:
    """Font trong danh mục có độ rộng chữ gần nét nháp nhất (cùng nhóm: nét dày-mảnh -> serif / viết tay, còn lại sans) --
    tên cụ thể từng dòng: con số độ rộng chung chung bị VLM bỏ qua (08/10: 70 lệnh bị co, font rộng hơn nét nháp 23%,
    gần như toàn Montserrat -- font rộng nhất bộ)."""
    from .fonts import CATALOG, WIDTH
    cls = ("serif", "script") if (m.get("stroke_contrast") or 0) >= 2.2 else ("sans",)
    col = 1 if m.get("caps") else 0
    keys = [k for k in WIDTH if CATALOG[k][1] in cls and "Italic" not in CATALOG[k][0]]   # bản nghiêng: VLM tự chọn khi nháp nghiêng
    return sorted(keys, key=lambda k: abs(WIDTH[k][col] - w))[:n]


def _look(m: dict) -> str:
    """Kiểu chữ đo trên nháp: độ đậm (độ dày nét / cỡ chữ), nét dày-mảnh (chữ có chân / viết tay), viết hoa, bóng / viền."""
    out = []
    if m.get("stroke") and m.get("size"):
        r = m["stroke"] / m["size"]   # độ dày nét / em
        out.append("weight " + ("light" if r < 0.07 else "regular" if r < 0.11 else "bold" if r < 0.16 else "black")
                   + f" (stroke ~{m['stroke']:.0f}px)")
    if (m.get("stroke_contrast") or 0) >= 2.2:
        out.append("thick-thin strokes (serif / script)")
    if m.get("caps"):
        out.append("all caps")
    n = len(m.get("ocr") or "")
    if n >= 4 and m.get("size") and abs(m.get("angle") or 0) < 2:
        # ĐỘ RỘNG CHỮ: rộng khung / (số ký tự x cỡ) -- so với độ rộng font ở danh mục (08/10: designer chọn font rộng hơn chữ nháp
        # -> phải co ~0.66, phần lớn lỗi too_small); OCR sai chữ nhưng số ký tự gần đúng
        w = (m["box"][2] - m["box"][0]) / (n * m["size"])
        # (08/10 lượt 4: ghi kèm tên font cùng độ rộng thì VLM đổi sang chữ condensed cả chỗ nháp là chữ thường -- bỏ)
        out.append(f"letters ~{w:.2f}em wide" + (" (condensed)" if w < 0.45 else " (wide)" if w > 0.62 else ""))
    e = m.get("effect")
    if e:
        if abs(e.get("dx", 0)) + abs(e.get("dy", 0)) >= 1:
            out.append(f"drop shadow {e.get('color')} offset ({e['dx']:.0f},{e['dy']:.0f})px")
        else:
            out.append(f"outline / glow {e.get('color')}")
    return (", " + ", ".join(out)) if out else ""


def _curve_note(m: dict) -> str:
    """Ô chữ chạy theo đường cong: vòm lên hay võng xuống, dài bao nhiêu (để VLM viết vừa một dòng)."""
    pt = np.asarray(m["path"], np.float64)
    ln = float(np.sum(np.hypot(*np.diff(pt, axis=0).T)))
    mid, chord = pt[len(pt) // 2], (pt[0] + pt[-1]) / 2
    n = max(2, int(round(ln / max(1.0, 0.55 * (m.get("size") or 1.0)))))
    return (f", CURVED LINE: the letters run along an {'arch (bulging up)' if mid[1] < chord[1] else 'arc bulging down (a smile)'}, "
            f"room for about {n} characters")


def describe(M: dict) -> str:
    sh = "\n".join(f"{s['id']}: shell {_px(s['box'])}, "
                   + (f"outline only (no fill) {s['border']} ~{s['border_px']:.0f}px" if s.get("border") else f"fill {s['fill']}")
                   + f", radius ~{s['radius']:.0f}px, contains "
                   f"{', '.join(s['lines'] + s['icons']) or 'nothing'}" for s in M["S"]) or "(none)"
    de = "\n".join(f"{c['id']}: detail {_px(c['box'])}, color {c['color']}" + (f", inside {c['shell']}" if c.get("shell") else "")
                   + (f", {c['near']}" if c.get("near") else "") for c in M["I"]) or "(none)"
    tilt = lambda m: (_curve_note(m) if m.get("path") else f", drawn at {m['angle']:.0f} degrees" if abs(m.get("angle") or 0) >= 2   # noqa: E731
                      else "") + (", OCR missed this line: the letters read here are a guess" if m.get("missed") else "") \
        + (", row mark in front (inside this slot)" if m.get("mark") else "") \
        + (", PRINTED ON THE CLIENT'S PRODUCT: kind \"keep\"" if m.get("printed") else "")
    li = "\n".join(f"{m['id']}: \"{m['ocr']}\" (text ~{m['size']:.0f}px, ~{len(m['ocr'])} characters, color {m['color']}"
                   + (f", inside {m['shell']}" if m.get("shell") else "") + f"{tilt(m)}{_look(m)})" for m in M["L"]) or "(none)"
    return f"Shells:\n{sh}\n\nDetails:\n{de}\n\nText lines:\n{li}"


PROMPT_V1 = os.environ.get("TEXTFIX_PROMPT", "") == "v1"
# ICON GỢI Ý theo chủ đề (13/10): trước đây gửi cả 1544 tên Lucide = 48% lời dặn; trên 207 poster đã lưu VLM chỉ dùng 26 tên,
# 20 tên phủ 91% số lần dùng. Tên ngoài danh sách vẫn dùng được (render.nearest tìm tên gần nhất).
ICON_HINTS = {
    "marks": "check circle-check square-check badge-check x plus minus arrow-right chevron-right star heart sparkles",
    "contact": "phone smartphone mail map-pin globe clock calendar calendar-days message-circle send navigation",
    "shop": "shopping-cart shopping-bag store tag percent gift credit-card wallet banknote truck package receipt ticket",
    "people": "circle-user user users baby graduation-cap briefcase award trophy crown thumbs-up handshake smile",
    "food": "utensils coffee cup-soda pizza cake ice-cream-cone soup salad beef fish croissant wine beer apple cookie",
    "home": "house bed sofa lamp bath shower-head washing-machine refrigerator air-vent fan tv key wrench hammer paintbrush",
    "health": "leaf flower sprout droplets sun moon scissors dumbbell bike heart-pulse stethoscope pill shield-check",
    "tech": "wifi zap battery-charging camera headphones laptop monitor gamepad-2 qr-code download cloud lock rocket",
    "travel": "plane car bus ship map compass mountain palmtree tent umbrella luggage hotel",
    "learn": "book-open pencil lightbulb music palette mic languages calculator",
}


def _system(base: str) -> str:
    from .fonts import CATALOG, WIDTH
    fonts = "; ".join(f"{k}: {v[0]} ({v[1]}" + (f", {WIDTH[k][0]:.2f} / caps {WIDTH[k][1]:.2f}em" if k in WIDTH else "") + ")"
                      for k, v in CATALOG.items())
    if PROMPT_V1 and base is SYSTEM:
        return SYSTEM_V1.replace("{fonts}", fonts).replace("{icons}", "")
    return base.replace("{fonts}", fonts).replace("{icons}", "")   # 14/10: bộ icon đã gỡ, TextFix chỉ sửa chữ


def _obj(x) -> dict:
    """Đầu ra VLM về dict: đôi khi trả JSON lồng trong chuỗi, hoặc mảng."""
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except ValueError:
            return {}
    if isinstance(x, list):
        x = {"ops": x}
    return x if isinstance(x, dict) else {}


def _to_marks(P) -> dict:
    """slots -> marks (render dùng marks). Lệnh không phải object (VLM trả sai khuôn) bị bỏ, ghi vào "dropped"."""
    P = _obj(P)
    ops, bad = [], []
    for raw in P.get("ops") or []:
        op = _obj(raw) if isinstance(raw, str) else raw
        if not isinstance(op, dict) or not op:
            bad.append(str(raw)[:80])
            continue
        sl = op.get("slots") or op.get("marks") or []
        ops.append({**op, "marks": [m for m in (sl if isinstance(sl, list) else [sl]) if isinstance(m, str)]})
    return {**P, "ops": ops, **({"dropped": bad} if bad else {})}


def _texts(texts: list[dict]) -> str:
    # câu "ai" = chữ phụ trợ bước brief tự viết (tiêu đề / khẩu hiệu / nút / nhãn khi khách chỉ điền sự thật): không phải lời khách
    return "\n".join(f"T{i} ({t.get('role', '')}{', OURS' if t.get('source') == 'ai' else ''}): {t['text']}"
                     for i, t in enumerate(texts))


PRODUCT_NOTE = ("The client uploaded a photo of their product and the image model copied it into the draft. Text, logos and "
                "graphics printed on that product (label, package, screen) are part of the photo and MUST stay exactly as they "
                "are: give every slot on the product kind \"keep\" -- never rewrite, correct, translate, redraw or skip it, "
                "and never write a client text there, even when the product's name is one of the client texts. Lines the list "
                "marks PRINTED ON THE CLIENT'S PRODUCT were recognised by code as product text: they are always \"keep\".")


VIEWS = {   # ảnh gửi designer (thử: scripts/vlm_probe.py). D nháp, S nháp đánh dấu ô, Q 4 góc phóng 2x, X bản xoá (nền dựng),
    "A": "DSQ",    # Z vùng cắt phóng to quanh cụm ô
    "AX": "DSQX",
    "BX": "DSX",
    "CX": "SXZ",
}
VIEW = "BX"  # chốt 07/10 (vlm_probe 23 ảnh dev): ít lỗi nhất, giữ chữ sản phẩm tốt nhất (29/30), rẻ nhất (~12k token)


def _quarters(mk) -> list[bytes]:
    """4 góc chồng 10%, phóng 2x: đọc được ô nhỏ."""
    from PIL import Image
    W, H = mk.size
    out = []
    for (a, b) in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x0, y0 = int(a * W * 0.45), int(b * H * 0.45)
        q = mk.crop((x0, y0, x0 + int(W * 0.55), y0 + int(H * 0.55)))
        out.append(_png(q.resize((q.width * 2, q.height * 2), Image.LANCZOS)))
    return out


def _clusters(mk, M: dict, n_max: int = 4, side: int = 1024) -> list[bytes]:
    """Vùng cắt quanh CỤM ô, phóng cạnh dài = side: chỗ có ô được phóng, chỗ chỉ có ảnh thì không."""
    from PIL import Image
    out = []
    for b in _cluster_boxes(M, *mk.size, n_max=n_max):
        c = mk.crop(b)
        z = side / max(c.size)
        out.append(_png(c.resize((max(1, int(c.width * z)), max(1, int(c.height * z))), Image.LANCZOS)))
    return out


def _cluster_boxes(M: dict, W: int, H: int, n_max: int = 4) -> list[list[int]]:
    """CỤM ô: khung ô nới 4% cạnh ngắn chạm nhau = một cụm; gộp cặp cụm làm tăng diện tích ít nhất tới <= n_max."""
    pad = 0.04 * min(W, H)
    boxes = [[b["box"][0] - pad, b["box"][1] - pad, b["box"][2] + pad, b["box"][3] + pad] for b in M["L"] + M["S"] + M["I"]]
    if not boxes:
        return []
    ov = lambda a, b: a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]   # noqa: E731
    un = lambda a, b: [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]   # noqa: E731
    cl = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                if ov(cl[i], cl[j]):
                    cl[i] = un(cl[i], cl.pop(j))
                    merged = True
                    break
            if merged:
                break
    while len(cl) > n_max:   # gộp cặp làm tăng diện tích ít nhất
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])   # noqa: E731
        i, j = min(((i, j) for i in range(len(cl)) for j in range(i + 1, len(cl))),
                   key=lambda ij: area(un(cl[ij[0]], cl[ij[1]])) - area(cl[ij[0]]) - area(cl[ij[1]]))
        cl[i] = un(cl[i], cl.pop(j))
    return [[max(0, int(b[0])), max(0, int(b[1])), min(W, int(b[2])), min(H, int(b[3]))] for b in sorted(cl, key=lambda b: (b[1], b[0]))]


def plan(draft: np.ndarray, M: dict, prompt: str, texts: list[dict], call, product: bool = False, base: np.ndarray | None = None,
         view: str | None = None, meta: dict | None = None, system: str | None = None, desc: str | None = None,
         base_label: str | None = None) -> dict:
    """base: bản xoá đã dọn (nền dựng) cho các cấu hình có X. meta: nhận số token / thời gian (llm.json_call).
    system / desc / base_label: lời dặn, bảng mô tả ô và nhãn ảnh nền khác mặc định (textfix/lonly.py)."""
    mk = marked_image(draft, M)
    W, H = mk.size
    v = VIEWS[view or VIEW]
    imgs = []
    for c in v:
        if c == "D":
            imgs += ["DRAFT:", _png(draft)]
        elif c == "S":
            imgs += ["SLOTS (marked):", _png(mk)]
        elif c == "Q":
            imgs += ["ZOOM (4 quarters of SLOTS, 2x):", *_quarters(mk)]
        elif c == "X" and base is not None:
            imgs += [base_label or "BASE (erased poster, your slots are drawn on it):", _png(base)]
        elif c == "Z":
            imgs += ["ZOOM (slot clusters of SLOTS, enlarged):", *_clusters(mk, M)]
    user = (f"PROMPT given to the image model:\n{prompt}\n\nClient texts (exact words):\n{_texts(texts)}\n\nPoster size: {W}x{H}px.\n\n"
            + (PRODUCT_NOTE + "\n\n" if product else "") + f"{desc or describe(M)}")
    args = (system or _system(SYSTEM), [user, *imgs]) + ((meta,) if meta is not None else ())
    return _to_marks(call(*args))


STAR_WORDS = re.compile(r"(>)[\s(]*\d+(?:[.,]\d+)?\s*(?:sao|stars?)[\s)]*(<)", re.I)   # mẩu chữ CHỈ ghi số sao, giữa hai thẻ
SWAP_SIM = 0.6      # chữ VLM viết chệch được thay bằng chữ khách khi giống >= 0.6 và cùng chữ cái đầu
CLIENT_LIKE = 0.6   # chữ OCR của ô giống câu khách >= 0.6 (bỏ dấu, chữ thường) = chữ poster chứa câu khách
PRINTED = re.compile(r"\b(printed|on (the )?(product|package|packaging|label|can|bottle|jar|box|screen))\b", re.I)


def _client_like(ocr: str, texts: list[dict]) -> int | None:
    """Chữ OCR (méo) của ô gần trùng một câu khách (bỏ dấu, chữ thường, giống >= CLIENT_LIKE) -> chỉ số câu đó."""
    from difflib import SequenceMatcher
    f = _flat(ocr or "")
    if len(f) < 4:
        return None
    best = max(((SequenceMatcher(None, f, _flat(t["text"])).ratio(), i) for i, t in enumerate(texts)), default=(0, None))
    if best[0] >= CLIENT_LIKE:
        return best[1]
    # OCR chỉ đọc được MẨU của câu (chữ cong trên ruy băng: "TRIAN" của "QUÀ TẶNG TRI ÂN") -> mẩu nằm gọn trong câu
    for i, t in enumerate(texts):
        m = SequenceMatcher(None, f, _flat(t["text"]))
        # khớp LIỀN một đoạn (ký tự rải rác trong câu dài 150 ký tự không tính: "Noinlien Hic" trên thân nồi, dev3 p09)
        if len(f) >= 5 and m.find_longest_match(0, len(f), 0, len(_flat(t["text"]))).size >= 0.8 * len(f):
            return i
    return None


def validate(P: dict, M: dict, texts: list[dict] | None = None, product: bool = False) -> tuple[dict, list[str]]:
    """Phòng ngừa nhẹ, có ghi log: id ô không có -> bỏ id; lệnh không còn ô nào -> bỏ lệnh. texts (câu khách): chặn chữ (_guard).
    product: khách tải ảnh sản phẩm (chữ in trên sản phẩm có thật)."""
    log, ops = [f"bỏ lệnh sai khuôn: {d}" for d in P.get("dropped") or []], []
    by = M["by"]
    # MỘT Ô, MỘT LỆNH (13/10 accept1 b05: VLM ghi thêm lệnh keep / skip cho ô đã có lệnh chữ -> chữ nháp dán lại nằm dưới chữ mới,
    # hai lớp chữ đè nhau): ô nằm ở nhiều lệnh thuộc về lệnh "nặng" nhất -- viết chữ > vẽ (vỏ, icon) > giữ > bỏ; ngang nhau: lệnh
    # đứng trước. Các lệnh kia mất ô đó.
    rank = {"text": 0, "shape": 1, "icon": 1, "stars": 1, "keep": 2, "skip": 3}
    src, owner = list(P.get("ops", [])), {}
    for i, op in enumerate(src):
        for m in op.get("marks") or []:
            k = (rank.get(op.get("kind"), 3) if op.get("kind") != "text" or op.get("html") else 3, i)
            if m in by and (m not in owner or k < owner[m]):
                owner[m] = k
    for i, op in enumerate(src):
        lost = [m for m in op.get("marks") or [] if m in owner and owner[m][1] != i]
        if lost:
            log.append(f"{op.get('id')}: ô {lost} đã thuộc lệnh {[src[owner[m][1]].get('id') for m in lost]} -> bỏ khỏi lệnh này")
            src[i] = {**op, "marks": [m for m in op["marks"] if m not in lost], "slots": [m for m in op["marks"] if m not in lost]}
    for op in src:
        mk = [m for m in op.get("marks") or [] if m in by]
        if len(mk) < len(op.get("marks") or []):
            log.append(f"{op.get('id')}: bỏ id không có {sorted(set(op.get('marks') or []) - set(mk))}")
        if not mk:
            continue
        # CHẤM ĐẦU DÒNG VẼ BẰNG ICON VÒNG TRÒN (13/10 accept1 b06, c01, f02: <i-icon name="circle"> / "○" đầu dòng thành vòng rỗng
        # to trước chữ dù lời dặn cấm): trong lệnh chữ, icon tròn / ký tự vòng tròn đứng đầu dòng -> "• "
        if op.get("kind") == "text" and op.get("html"):
            h2 = re.sub(r"(^|<br\s*/?>)(\s*)<i-icon\b[^>]*\bname=[\"'](?:circle|dot|circle-dot|circle-small)[\"'][^>]*>(?:\s*</i-icon>)?\s*",
                        lambda m: m.group(1) + "• ", op["html"], flags=re.I)
            h2 = re.sub(r"(^|<br\s*/?>|>)(\s*)[○◯◦⚪⭕●⬤]\s*", lambda m: m.group(1) + "• ", h2)
            if h2 != op["html"]:
                log.append(f"{op.get('id')}: vòng tròn đầu dòng -> chấm '•'")
                op = {**op, "html": h2}
        # DÒNG TRỐNG TRONG LỆNH GỘP (dev3 g09_v1: "CƠM<br>NGON<br><br>TẬN<br><br>NƠI" -- VLM để trống dòng nó bỏ, khối chữ thủng lỗ):
        # các dòng của một lệnh chữ luôn liền nhau
        if op.get("kind") == "text" and re.search(r"(<br\s*/?>\s*){2,}", op.get("html") or "", re.I):
            log.append(f"{op.get('id')}: bỏ dòng trống giữa các dòng của lệnh")
            op = {**op, "html": re.sub(r"(<br\s*/?>\s*){2,}", "<br>", op["html"], flags=re.I)}
        # ICON VIẾT VÀO Ô DÒNG CHỮ (13/10 run 2638: VLM đặt icon vào L9, L11 -- hai dòng của đoạn mô tả): ô L là chữ, icon ở đó
        # là nhầm ô -> bỏ lệnh (nền đã xoá sạch)
        if op.get("kind") in ("icon", "stars") and all(m.startswith("L") for m in mk):
            log.append(f"{op.get('id')}: {op['kind']} viết vào ô dòng chữ {mk} -> skip")
            op = {**op, "kind": "skip", "html": "", "why": "guard: icon on a text line"}
        # Ô CHỮ CONG ĐI MỘT MÌNH (13/10 run ba93: VLM gộp ô vòm với ô thẳng bên cạnh thành một lệnh -> bộ dựng không có đường
        # nào cho "một vòm + một khung", dựng thành khối chữ thẳng đè lên hình giữa con dấu). Lệnh chữ có ô cong lẫn ô L khác:
        # cả câu viết theo đường cong (dài nhất nếu nhiều), các ô L còn lại để trống -- chữ nén trên vòm vẫn đúng dáng nháp.
        arc = [m for m in mk if m.startswith("L") and by[m].get("path")]
        if op.get("kind") == "text" and arc and sum(m.startswith("L") for m in mk) > 1:
            plen = lambda m: float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(by[m]["path"][:-1], by[m]["path"][1:])))   # noqa: E731
            main = max(arc, key=plen)
            rest = [m for m in mk if m.startswith("L") and m != main]
            log.append(f"{op.get('id')}: lệnh gộp ô cong {main} với {rest} -> viết cả câu theo đường cong {main}, {rest} để trống")
            ops.append({"id": f"{op.get('id')}x", "kind": "skip", "marks": rest, "slots": rest, "why": "guard: merged with a curved line"})
            mk = [m for m in mk if m not in rest]
            op = {**op, "html": re.sub(r"\s*<br\s*/?>\s*", " ", op.get("html") or "")}
        # chữ / logo IN TRÊN SẢN PHẨM mà VLM cho skip (tự ghi lý do "printed logo on product") -> bản xoá đã xoá nó, skip là mất
        # luôn (09/10 dev2 d25: con dấu trên bánh pía) -> keep (dán lại pixel nháp)
        # CHỈ khi khách tải ảnh sản phẩm: không có ảnh sản phẩm thì "chữ in trên ruy băng / sản phẩm" là chữ POSTER FLUX vẽ (09/10
        # dev3 g15: dải ruy băng "QUÀ TẶNG TRI ÂN" -> keep dán lại chữ nháp sai chính tả "QUẢ TỆNG")
        if product and op.get("kind") == "skip" and PRINTED.search(op.get("why") or "") and                 not re.search(r"duplicat|junk|garbl", op.get("why") or "", re.I):
            log.append(f"{op.get('id')}: skip chữ in trên sản phẩm ({op.get('why')!r}) -> keep")
            op = {**op, "kind": "keep"}
        # CHỮ IN TRÊN SẢN PHẨM KHÁCH TẢI (13/10): dòng mã nhận ra trên sản phẩm (mark_printed) luôn giữ nguyên điểm ảnh, VLM
        # ghi gì cũng vậy; lệnh chữ gộp cả dòng đó thì dòng đó tách ra thành lệnh giữ riêng
        pr = [m for m in mk if by[m].get("printed")]
        if pr and op.get("kind") != "keep":
            log.append(f"{op.get('id')}: {pr} là chữ in trên sản phẩm khách tải -> giữ nguyên (VLM ghi {op.get('kind')})")
            ops.append({"id": f"{op.get('id')}p", "kind": "keep", "marks": pr, "slots": pr, "why": "guard: printed on the client's product"})
            mk = [m for m in mk if m not in pr]
            if not mk:
                continue
        # KEEP CHỮ KHÁCH: ô giữ pixel nháp mà chữ OCR của nó gần trùng một câu khách = chữ poster model vẽ (thường sai chính tả)
        # -> viết lại bằng câu khách (09/10 dev3 g15 v0: VLM keep "QUÀ TẶNG TRI ÂN" trên ruy băng). KHÔNG làm khi khách tải ảnh
        # sản phẩm: tên sản phẩm in trên vỏ hộp trùng câu khách là chuyện thường, và chữ trên sản phẩm phải giữ
        if op.get("kind") == "keep" and texts and not product:
            ocr = " ".join(by[m].get("ocr", "") for m in mk if m.startswith("L"))
            k = _client_like(ocr, texts)
            if k is not None:
                log.append(f"{op.get('id')}: keep chữ nháp {ocr!r} trùng câu khách T{k} -> viết lại {texts[k]['text']!r}")
                op = {**op, "kind": "text", "html": texts[k]["text"], "client": f"T{k}", "why": "guard: client text was kept"}
        # SAO ĐÃ VẼ thì không viết thêm "5 sao" (12/10 run 38ef: hàng sao + chữ "5 sao"): bỏ mẩu chữ chỉ ghi số sao đứng cạnh thẻ sao
        if "<i-stars" in (op.get("html") or ""):
            h = STAR_WORDS.sub(lambda m: m.group(1) + m.group(2), ">" + op["html"] + "<")[1:-1]
            if h != op["html"]:
                log.append(f"{op.get('id')}: đã có hàng sao -> bỏ chữ ghi số sao")
                op = {**op, "html": h}
        ops.append({**op, "marks": mk, "slots": mk})
    # VỎ (12/10, xem 65 ô vỏ của 130 poster):
    #  - vỏ đang chứa dòng chữ ĐƯỢC VIẾT mà bị "skip" -> chữ nằm trần trên nền (2ad2_v1 khung giá, 4af9_v0 pill trắng) -> vẽ lại
    #    theo số đo (màu lòng / viền, bo góc);
    #  - vỏ "keep" (giữ điểm ảnh nháp) mà không lấp được nét chữ cũ (lòng không phẳng màu: keepable = False) -> vẽ lại theo số đo.
    live = {m for op in ops if op.get("kind") == "text" and op.get("html") for m in op["marks"]}
    for i, op in enumerate(ops):
        sid = [m for m in op["marks"] if m.startswith("S")]
        if not sid or len(sid) != len(op["marks"]) or op.get("kind") not in ("skip", "keep"):
            continue
        s0 = by[sid[0]]
        inside = [m for m in s0.get("lines") or [] if m in live]
        # màu lòng đo được gần trùng màu chữ bên trong: số đo sai (hoặc vỏ giả) -> không vẽ, chữ sẽ chìm
        same = s0.get("fill") not in (None, "none") and any(
            by[m].get("color") and float(np.linalg.norm(_hex_lab(by[m]["color"]) - _hex_lab(s0["fill"]))) < SHELL_INK_DE for m in inside)
        # lệnh chữ bên trong đã tự có nền / viền (VLM vẽ vỏ ngay trên lệnh chữ rồi bỏ ô vỏ -- dev3 g04): vẽ thêm là vỏ lồng vỏ
        own = any(re.search(r"background|border\s*:", (o.get("box_style") or "") + (o.get("html") or ""), re.I)
                  for o in ops if o.get("kind") == "text" and set(o["marks"]) & set(inside))
        if same or own:
            continue
        if (op["kind"] == "skip" and inside) or (op["kind"] == "keep" and s0.get("keepable") is False and inside):
            css = (f"border:{max(1.0, s0.get('border_px') or 2):.0f}px solid {s0['border']}" if s0.get("border")
                   else f"background:{s0.get('fill')}") + f"; border-radius:{s0.get('radius') or 0:.0f}px"
            log.append(f"{op.get('id')}: vỏ {sid[0]} {op['kind']} mà chứa chữ được viết {inside} -> vẽ lại theo số đo ({css})")
            ops[i] = {**op, "kind": "shape", "html": "", "box_style": css, "why": "guard: shell holding written text"}
    if texts:
        ops, lg = _guard(ops, M, texts)
        log += lg
    log += _pack_lists(ops, by)
    if texts:
        log += _list_order(ops, by)
    # ICON / CHẤM ĐẦU DÒNG TRƠ: chi tiết đi kèm một dòng (near "left of L5") mà dòng đó bị skip -> bỏ luôn chi tiết (09/10 dev3
    # p08, p11: icon / chấm đầu dòng không có chữ). Chạy SAU chặn chữ và dồn hàng: dòng bị bỏ ở hai bước đó cũng tính.
    dead = {m for op in ops if op.get("kind") == "skip" for m in op["marks"] if m.startswith("L")}
    live = {m for op in ops if op.get("kind") in ("text", "keep") for m in op["marks"]}
    for i, op in enumerate(ops):
        if op.get("kind") in ("icon", "stars") and all(m.startswith("I") for m in op["marks"]):
            tied = _tied(op, by)
            if tied and all(t in dead and t not in live for t in tied):
                log.append(f"{op.get('id')}: chi tiết đi kèm dòng bị bỏ {tied} -> skip")
                ops[i] = {**op, "kind": "skip", "html": "", "why": "guard: orphan bullet / icon"}
    log += _icon_sets(ops, by)
    return {**P, "ops": ops}, log


def _tied(op: dict, by: dict) -> list[str]:
    """Các dòng mà chi tiết của lệnh đi kèm (đứng bên trái / nằm trên dòng đó)."""
    near = [(by[m].get("near") or "").split() for m in op.get("marks") or [] if m in by]
    return [n[-1] for n in near if len(n) >= 2 and n[0] in ("left", "on")]


def _lists(ops: list[dict], by: dict) -> dict:
    """Danh sách VLM khai: {tên nhóm: [chỉ số lệnh chữ]} (group_kind "list", >= 2 lệnh, mỗi lệnh có dòng L, không nghiêng)."""
    G = {}
    for i, op in enumerate(ops):
        if op.get("kind") == "text" and op.get("html") and op.get("group") and op.get("group_kind") == "list":
            G.setdefault(op["group"], []).append(i)

    def ok(i):
        lm = [m for m in ops[i]["marks"] if m.startswith("L")]
        return bool(lm) and all(abs(by[m].get("angle") or 0) < 2 for m in lm)
    return {g: idx for g, idx in G.items() if len(idx) >= 2 and all(ok(i) for i in idx)}


def _list_order(ops: list[dict], by: dict) -> list[str]:
    """THỨ TỰ KHÁCH CHO (13/10 accept1 f06: VLM xếp bước 4 trước bước 3 cho khớp hình minh hoạ của nháp): các hàng của một danh
    sách VLM khai, xếp một cột, mỗi hàng viết một câu khách khác nhau -> từ trên xuống phải tăng dần theo thứ tự câu khách. Sai
    thì hoán nội dung giữa các hàng (chữ + câu khách), ô và kiểu của từng hàng giữ nguyên."""
    log = []
    for name, idx in _lists(ops, by).items():
        ks = [_tid(ops[i].get("client")) for i in idx]
        if any(k is None for k in ks) or len(set(ks)) != len(ks):
            continue
        top = lambda i: min(by[m]["box"][1] for m in ops[i]["marks"] if m.startswith("L"))   # noqa: E731
        bot = lambda i: max(by[m]["box"][3] for m in ops[i]["marks"] if m.startswith("L"))   # noqa: E731
        rows = sorted(idx, key=top)
        if any(top(b) < (top(a) + bot(a)) / 2 for a, b in zip(rows, rows[1:])):   # không phải một cột (lưới, hàng ngang)
            continue
        cur = [_tid(ops[i].get("client")) for i in rows]
        if cur == sorted(cur):
            continue
        if any(len([m for m in ops[i]["marks"] if m.startswith("L")]) != len([m for m in ops[rows[0]]["marks"] if m.startswith("L")])
               for i in rows):
            continue   # hàng một dòng và hàng hai dòng: hoán chữ làm lệch số dòng
        content = sorted(((_tid(ops[i].get("client")), ops[i]["html"], ops[i].get("client")) for i in rows), key=lambda t: t[0])
        for i, (k, html, cl) in zip(rows, content):
            ops[i] = {**ops[i], "html": html, "client": cl}
        log.append(f"danh sách '{name}': thứ tự {['T%d' % k for k in cur]} -> theo thứ tự khách {['T%d' % c[0] for c in content]}")
    return log


def _pack_lists(ops: list[dict], by: dict) -> list[str]:
    """HÀNG TRỐNG GIỮA DANH SÁCH (12/10 run 38ef: nháp vẽ một ý hai lần, VLM bỏ hàng lặp ở giữa -> danh sách thủng một hàng):
    hàng bị bỏ nằm GIỮA các hàng của một danh sách VLM khai, cùng cột, cùng số dòng -> dồn các ý lên theo thứ tự, hàng trống
    xuống cuối. Chỉ đổi ô giữa các lệnh (ý nào viết vào hàng nào); chữ, kiểu, nhóm giữ nguyên."""
    log = []

    def box(lm):
        return [min(by[m]["box"][0] for m in lm), min(by[m]["box"][1] for m in lm),
                max(by[m]["box"][2] for m in lm), max(by[m]["box"][3] for m in lm)]
    for name, idx in _lists(ops, by).items():
        if any(not all(m.startswith("L") for m in ops[i]["marks"]) for i in idx):
            continue
        mb = {i: box(ops[i]["marks"]) for i in idx}
        x0, x1 = min(b[0] for b in mb.values()), max(b[2] for b in mb.values())
        y0, y1 = min(b[1] for b in mb.values()), max(b[3] for b in mb.values())
        holes = []
        for j, op in enumerate(ops):
            lm = op.get("marks") or []
            if op.get("kind") != "skip" or not lm or not all(m.startswith("L") for m in lm):
                continue
            b = box(lm)
            if y0 < (b[1] + b[3]) / 2 < y1 and min(b[2], x1) - max(b[0], x0) >= 0.5 * (b[2] - b[0]) \
                    and not any(min(b[3], q[3]) - max(b[1], q[1]) > 0.5 * (b[3] - b[1]) for q in mb.values()):
                holes.append(j)
        if not holes:
            continue
        units = sorted(idx + holes, key=lambda i: box(ops[i]["marks"])[1])
        mem = [i for i in units if i in idx]
        rest = [i for i in units if i not in idx]
        marks = [list(ops[i]["marks"]) for i in units]
        if any(len(marks[k]) != len(ops[mem[k]]["marks"]) for k in range(len(mem))):
            continue   # hàng trống có số dòng khác ý sẽ dồn vào: để nguyên
        for k, i in enumerate(mem):
            ops[i] = {**ops[i], "marks": marks[k], "slots": marks[k]}
        for k, i in enumerate(rest):
            ops[i] = {**ops[i], "marks": marks[len(mem) + k], "slots": marks[len(mem) + k]}
        log.append(f"danh sách '{name}': hàng trống ở giữa -> dồn các ý lên, hàng trống xuống cuối ({[ops[i].get('id') for i in rest]})")
    return log


def _icon_sets(ops: list[dict], by: dict) -> list[str]:
    """ICON CỦA MỘT DANH SÁCH là MỘT BỘ (12/10 run 93f1: ba hàng -- icon to, hai icon nhỏ chồng nhau, không icon): khi từ nửa số
    hàng (>= 2) của danh sách VLM khai có icon đầu hàng, thì mỗi hàng đúng MỘT icon: hàng có nhiều icon giữ cái trên cùng, hàng
    có ô icon mà VLM bỏ -> vẽ lại bằng icon hay dùng nhất của bộ. Gắn "icon_set" cho cả bộ: bước dựng cho cùng cỡ, thẳng một cột."""
    log = []

    def top(j):
        return min(by[m]["box"][1] for m in ops[j]["marks"])
    for name, idx in _lists(ops, by).items():
        rows = {i: set(ops[i]["marks"]) for i in idx}
        icons, skipped = {i: [] for i in idx}, {i: [] for i in idx}
        for j, op in enumerate(ops):
            mk = op.get("marks") or []
            if not mk or not all(m.startswith("I") for m in mk) or op.get("kind") not in ("icon", "skip"):
                continue
            row = next((i for i, r in rows.items() if set(_tied(op, by)) & r), None)
            if row is not None:
                (icons if op["kind"] == "icon" else skipped)[row].append(j)
        have = [i for i in idx if icons[i]]
        if len(have) < 2 or 2 * len(have) < len(idx):
            continue
        htmls = [ops[j].get("html") or "" for i in idx for j in icons[i]]
        tpl = max(sorted(set(htmls)), key=htmls.count)
        for i in idx:
            js = sorted(icons[i], key=top)
            for j in js[1:]:
                log.append(f"{ops[j].get('id')}: hàng {ops[i].get('id')} đã có icon {ops[js[0]].get('id')} -> skip")
                ops[j] = {**ops[j], "kind": "skip", "html": "", "why": "guard: one icon per list row"}
            if not js and skipped[i]:
                j = min(skipped[i], key=top)
                log.append(f"{ops[j].get('id')}: hàng {ops[i].get('id')} mất icon (các hàng khác của '{name}' có) -> vẽ lại")
                ops[j] = {**ops[j], "kind": "icon", "html": tpl, "why": "guard: list row lost its icon"}
                js = [j]
            for j in js[:1]:
                ops[j] = {**ops[j], "icon_set": name}
    return log


def uncovered(P: dict, texts: list[dict]) -> list[int]:
    """Câu khách KHÔNG có trên poster, đo bằng mã (09/10 dev2 d05: VLM bỏ "Thêm huyết 5.000đ" mà không khai "missing"): câu
    có chữ (bỏ dấu, chữ thường) không nằm trong chữ của mọi lệnh viết / giữ. -> chỉ số câu."""
    have = set()
    for op in P.get("ops", []):
        if op.get("kind") in ("text", "group", "shape") and op.get("html"):   # thẻ inline nối chữ ("TR<span>Ư</span>ỜNG") hoặc
            h = op["html"]                                                        # tách cột ("<span>tên</span><span>giá</span>")
            have |= set(WORD.findall(_fold(_plain(h)))) | set(WORD.findall(_fold(_plain(re.sub(r"<[^>]+>", " ", h)))))
    out = []
    for i, t in enumerate(texts):
        w = set(WORD.findall(_fold(t["text"])))
        if w and not w <= have:
            out.append(i)
    return out


def score(P: dict, errs: list[dict]) -> tuple:
    """So bản lượt đầu / sau vòng sửa (nhỏ hơn = tốt hơn): trước hết số câu khách được viết (08/10 m03: vòng sửa bỏ tiêu đề
    "KHAI TRƯƠNG" để hết lỗi too_small và được giữ vì ít lỗi hơn), rồi số lỗi, rồi mức chồng / tràn."""
    told = {_tid(op.get("client")) for op in P.get("ops", []) if op.get("kind") == "text" and op.get("html")} - {None}
    return -len(told), len(errs), sum(e.get("frac", 0) + e.get("px", 0) / 100 for e in errs)


def _fold(s: str) -> str:
    """Bỏ dấu tiếng Việt, chữ thường: "TRƯỞNG" == "trương" == "TRUONG"."""
    import unicodedata
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _plain(h: str) -> str:
    import html as H
    # <br> / <BR> / thẻ khối = dấu cách; thẻ inline (span, i-font) không tách chữ (08/10 m01: "FLASH<BR>SALE" thành "FLASHSALE")
    h = re.sub(r"<\s*(br|/?div|/?p)\b[^>]*>", " ", h or "", flags=re.I)
    return " ".join(H.unescape(re.sub(r"<[^>]+>", "", h)).split())


def _flat(s: str) -> str:
    return re.sub(r"\W", "", _fold(s))


def _tid(c) -> int | None:
    c = str(c or "")
    return int(c[1:]) if c[:1] in "Tt" and c[1:].isdigit() else int(c) if c.isdigit() else None


WORD = re.compile(r"\w+", re.UNICODE)


def _guard(ops: list[dict], M: dict, texts: list[dict]) -> tuple[list[dict], list[str]]:
    """CHẶN CHỮ SAI bằng mã (08/10, BX 23 ảnh dev: VLM chép chữ từ ảnh nháp thay vì câu khách):
    1. DẤU: chữ khớp một chữ của câu khách khi bỏ dấu nhưng khác dấu ("TRƯỞNG", "TRƯỜNG" cho "TRƯƠNG") -> thay bằng chữ của
       khách (giữ hoa / thường). Chỉ xét lệnh đang viết câu khách: có "client", hoặc cả dòng (bỏ dấu) nằm trong một câu khách.
    2. CHỮ BỊA: lệnh khai viết câu khách T# mà không chữ nào có trong bất kỳ câu khách nào ("SƯ PRING" chép từ mẩu OCR vỡ) -> skip.
    2b. SỐ BỊA: lệnh không khai câu khách mà có số không nằm trong câu khách nào -> skip.
    3. LẶP: nháp hay vẽ một dòng hai lần (ngày, giá, nhãn huy hiệu); hai lệnh viết cùng một chữ (bỏ dấu, >= 3 ký tự) mà câu
       khách không lặp -> giữ lệnh có chữ OCR của ô giống nhất, lệnh kia skip (nền xoá sạch ở đó)."""
    from difflib import SequenceMatcher
    log, txt = [], [t["text"] for t in texts]
    ftxt = [" ".join(WORD.findall(_fold(t))) for t in txt]
    vocab_all = {w for t in txt for w in WORD.findall(t)}
    out = []
    for op in ops:
        if op.get("kind") != "text" or not op.get("html"):
            out.append(op)
            continue
        plain = _plain(op["html"])
        fp = " ".join(WORD.findall(_fold(plain)))
        k = _tid(op.get("client"))
        own = txt[k] if k is not None and 0 <= k < len(txt) else None
        words = WORD.findall(plain)
        if own is not None and words and not any(c.isdigit() for c in plain) and \
                not {_fold(w) for w in words} & {_fold(w) for w in vocab_all} and \
                not any(_flat(plain) in _flat(t) for t in txt):   # chữ dính / tách khác câu khách vẫn là chữ khách
            log.append(f"{op.get('id')}: chữ không có trong câu khách ({plain!r} khai {op.get('client')}) -> skip")
            out.append({**op, "kind": "skip", "html": "", "why": "guard: invented text"})
            continue
        nums = re.findall(r"\d+", plain)
        # so NGUYÊN số (không phải chuỗi con: "00" nằm trong "1.590.000đ", "36" trong "-36%" -> "00:36" trên màn hình nồi cơm lọt
        # qua, 09/10 dev2 d27)
        tnums = {n for t in txt for n in re.findall(r"\d+", t)}
        if own is None and nums and not all(n in tnums for n in nums):
            # SỐ BỊA: lệnh không khai câu khách mà viết số không có trong câu khách nào (địa chỉ / giá / điện thoại / ngày nháp tự
            # vẽ, 08/10 "123 Đường Số 1, Quận 1, TP. HCM") -> skip: dữ kiện không bao giờ bịa
            log.append(f"{op.get('id')}: số không có trong câu khách ({plain!r}) -> skip")
            out.append({**op, "kind": "skip", "html": "", "why": "guard: invented fact"})
            continue
        src = [own] if own is not None else [t for t, f in zip(txt, ftxt) if fp and f" {fp} " in f" {f} "]
        if src:
            cand = {}   # chữ bỏ dấu -> các dạng có dấu trong câu của lệnh
            for t in src:
                for w in WORD.findall(t):
                    cand.setdefault(_fold(w), set()).add(w.lower())

            def fix(m):
                w = m.group(0)
                opts = cand.get(_fold(w)) or set()
                if len(opts) != 1 or w.lower() in opts:
                    return w
                c = next(iter(opts))
                return c.upper() if w.isupper() else (c[:1].upper() + c[1:]) if w[:1].isupper() else c
            parts = re.split(r"(<[^>]+>|&\w+;|&#\d+;)", op["html"])
            new = "".join(p if p.startswith("<") or p.startswith("&") else WORD.sub(fix, p) for p in parts)
            if new != op["html"]:
                log.append(f"{op.get('id')}: sửa dấu theo câu khách {plain!r} -> {_plain(new)!r}")
                op = {**op, "html": new}
        if own is not None:
            # 1a. CHỮ CHỆCH (13/10 run 1f85: VLM viết "ngàn" cho "nghìn"; luật 1b xoá "ngàn", tiêu đề mất một chữ): chữ không có
            #     trong câu khách nào, đứng đúng chỗ MỘT chữ của câu này (hai bên khớp) -> thay bằng chữ của khách.
            fv = {_fold(w) for w in vocab_all}
            parts = re.split(r"(<[^>]+>|&\w+;|&#\d+;)", op["html"])
            mine = [w for p_ in parts if not p_.startswith(("<", "&")) for w in WORD.findall(p_)]
            theirs = WORD.findall(own)
            swap = {}
            for tag, i1, i2, j1, j2 in SequenceMatcher(None, [_fold(w) for w in mine], [_fold(w) for w in theirs], autojunk=False).get_opcodes():
                if tag == "replace" and i2 - i1 == j2 - j1:
                    for a, b in zip(range(i1, i2), range(j1, j2)):
                        w = mine[a]
                        # chỉ khi hai chữ GẦN GIỐNG nhau (ngàn / nghìn: 0.67; 13/10 accept1 a01: VLM viết lại câu cho ngắn,
                        # luật thay bừa "nguyên" -> "giúp", "hương" -> "việc", câu thành vô nghĩa)
                        if _fold(w) not in fv and not any(ch.isdigit() for ch in w + theirs[b]) and _fold(w)[:1] == _fold(theirs[b])[:1] \
                                and SequenceMatcher(None, _fold(w), _fold(theirs[b])).ratio() >= SWAP_SIM:
                            c = theirs[b]
                            swap[a] = c.upper() if w.isupper() else (c[:1].upper() + c[1:]) if w[:1].isupper() and a == 0 else c
            if swap:
                n = [-1]

                def put(m, n=n, swap=swap):
                    n[0] += 1
                    return swap.get(n[0], m.group(0))
                new = "".join(p_ if p_.startswith(("<", "&")) else WORD.sub(put, p_) for p_ in parts)
                log.append(f"{op.get('id')}: chữ chệch {[mine[a] for a in swap]} -> chữ của khách {list(swap.values())}: {_plain(new)!r}")
                op = {**op, "html": new}
                plain = _plain(new)
            # 1b. CHỮ THỪA: lệnh viết câu khách mà chen chữ (chữ cái) không có trong câu khách nào -- chép chữ méo của nháp (09/10
            #     dev2 d24: nháp "CUỐI TUÃ / TUẬN" -> "CUỐI TỬA<br>TUẦN") -> bỏ chữ đó. Số để luật 2b / câu ngắn lo.
            fv = {_fold(w) for w in vocab_all}
            parts = re.split(r"(<[^>]+>|&\w+;|&#\d+;)", op["html"])
            drop = [w for p in parts if not p.startswith(("<", "&")) for w in WORD.findall(p)
                    if not any(c.isdigit() for c in w) and _fold(w) not in fv]
            if drop and len(drop) < len(WORD.findall(plain)):
                new = "".join(p if p.startswith(("<", "&")) else
                              re.sub(r"[ \t]{2,}", " ", WORD.sub(lambda m: "" if m.group(0) in drop else m.group(0), p))
                              for p in parts)
                log.append(f"{op.get('id')}: bỏ chữ không có trong câu khách {drop} -> {_plain(new)!r}")
                op = {**op, "html": new}
        out.append(op)
    # 3. lặp
    groups = {}
    for i, op in enumerate(out):
        if op.get("kind") == "text" and op.get("html"):
            f = " ".join(WORD.findall(_fold(_plain(op["html"]))))
            if len(f.replace(" ", "")) >= 3:
                groups.setdefault(f, []).append(i)
    for f, idx in groups.items():
        # cùng cụm chữ nằm trong HAI câu khách khác nhau (tên "An Gia Tổng Vệ Sinh" và ưu đãi "... đặt lịch tổng vệ sinh trước Tết",
        # 12/10 run 5d27: dòng giữa huy hiệu bị xoá oan) -> mỗi lệnh khai một câu khách riêng có chứa cụm đó là một lần viết hợp lệ
        own = {k for i in idx for k in [_tid(out[i].get("client"))] if k is not None and k < len(ftxt) and f" {f} " in f" {ftxt[k]} "}
        allowed = max(1, sum(1 for t in ftxt if t == f), len(own))
        if len(idx) <= allowed:
            continue
        ocr = lambda i: _fold(" ".join(M["by"][m].get("ocr", "") for m in out[i]["marks"] if m.startswith("L")))   # noqa: E731
        # ô nào nháp vẽ đúng câu đó thì giữ; NHIỀU ô cùng giống (>= 0.5) -> giữ ô chữ TO nhất (13/10 accept1 p02: tên sản phẩm
        # nháp vẽ cả ở tiêu đề lẫn hàng đầu danh sách; giữ hàng danh sách thì vùng tiêu đề trống)
        big = lambda i: max((M["by"][m].get("size") or 0.0) for m in out[i]["marks"] if m.startswith("L")) if any(   # noqa: E731
            m.startswith("L") for m in out[i]["marks"]) else 0.0
        sim = {i: SequenceMatcher(None, ocr(i), f).ratio() for i in idx}
        idx = sorted(idx, key=lambda i: (-(sim[i] >= 0.5), -big(i) if sim[i] >= 0.5 else 0.0, -sim[i]))
        if M.get("lonly"):   # chế độ chỉ dòng chữ: nháp vẽ câu đó hai chỗ thì viết cả hai, bỏ một là một nút trống (mix1 i05)
            log.append(f"lặp chữ {f!r} ở {[out[i].get('id') for i in idx]}: giữ cả (chế độ chỉ dòng chữ, nháp vẽ nhiều chỗ)")
            continue
        for i in idx[allowed:]:
            log.append(f"{out[i].get('id')}: lặp chữ {_plain(out[i]['html'])!r} (giữ {out[idx[0]].get('id')}) -> skip")
            out[i] = {**out[i], "kind": "skip", "html": "", "why": "guard: duplicate"}
    # 4. CÂU NGẮN (<= 3 chữ: lương "15-25 triệu", giá, nhãn) không được cắt bớt -- câu dài mới được rút gọn (08/10 h06: vòng sửa
    #    bỏ "triệu" cho vừa ô). Một lệnh viết câu đó mà thiếu chữ -> trả nguyên câu khách vào đúng chỗ chữ trong html.
    for k, t in enumerate(txt):
        tw = WORD.findall(t)
        mine = [i for i, op in enumerate(out) if op.get("kind") == "text" and op.get("html") and _tid(op.get("client")) == k]
        if not tw or len(tw) > 3 or len(mine) != 1:
            continue
        op = out[mine[0]]
        plain = _plain(op["html"])
        lost = {_fold(w) for w in tw} - {_fold(w) for w in WORD.findall(plain)}
        if not lost or not plain or plain not in op["html"]:
            continue
        near = {_fold(w) for o in out if o is not op and o.get("kind") == "text" for w in WORD.findall(_plain(o.get("html") or ""))}
        if lost & near:   # chữ thiếu đã nằm ở lệnh khác (huy hiệu tách "Giảm" / "50%" thành hai ô)
            continue
        log.append(f"{op.get('id')}: câu ngắn bị cắt {plain!r} -> {t!r}")
        out[mine[0]] = {**op, "html": op["html"].replace(plain, t, 1)}
    return out, log


def _ids_image(proof: np.ndarray, M: dict, P: dict):
    """Bản dựng + khung và id từng lệnh vẽ (màu theo loại) -- để VLM trỏ đúng lệnh khi duyệt."""
    from PIL import Image, ImageDraw, ImageFont
    from .render import KIND_COL, op_box
    im = Image.fromarray(proof).convert("RGB")
    d = ImageDraw.Draw(im)
    f = ImageFont.load_default(size=max(12, im.width // 60))
    for op in P["ops"]:
        b = op_box(op, M) if op.get("kind") not in ("skip", "keep") else None
        if not b:
            continue
        col = KIND_COL.get(op.get("kind"), (230, 30, 30))
        b = [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])]
        d.rectangle(b, outline=col, width=2)
        tw, th = d.textbbox((0, 0), op["id"], font=f)[2:]
        d.rectangle([b[0], max(0, b[1] - th - 2), b[0] + tw + 2, max(0, b[1] - th - 2) + th + 2], fill=col)
        d.text((b[0] + 1, max(0, b[1] - th - 2)), op["id"], fill=(255, 255, 255), font=f)
    return im


def _compare(draft: np.ndarray, proof: np.ndarray, M: dict, n_max: int = 3, side: int = 1280) -> list[bytes]:
    """Ảnh SO SÁNH vùng đông ô: nháp (trái) | bản dựng (phải), cùng vùng cắt, phóng -- toàn cảnh 1024px quá nhỏ để thấy chữ
    trong huy hiệu / chân trang (08/10)."""
    from PIL import Image
    H, W = draft.shape[:2]
    out = []
    for b in _cluster_boxes(M, W, H, n_max=n_max):
        a, c = Image.fromarray(draft).crop(b), Image.fromarray(proof).crop(b)
        im = Image.new("RGB", (a.width * 2 + 12, a.height), "white")
        im.paste(a, (0, 0))
        im.paste(c, (a.width + 12, 0))
        z = min(side / im.width, side / im.height, 3.0)
        out.append(_png(im.resize((max(1, int(im.width * z)), max(1, int(im.height * z))), Image.LANCZOS)))
    return out


def review(draft: np.ndarray, proof: np.ndarray, M: dict, P: dict, errs: list[dict], res: list[dict], prompt: str,
           texts: list[dict], call, product: bool = False, meta: dict | None = None, changes: list[str] | None = None) -> dict:
    """VÒNG DUYỆT (luôn chạy, 08/10 thay vòng sửa chỉ-khi-có-lỗi): VLM xem bản dựng cả tấm như giám đốc nghệ thuật -- sửa lỗi
    khách quan (render.check) VÀ lỗi thẩm mỹ tự thấy (mồ côi, chữ nhỏ / mảnh trong huy hiệu, lệch tâm vỏ, cùng vai khác cỡ /
    độ đậm, icon không đều...). Gửi: nháp, bản dựng, bản dựng có id lệnh, danh sách lệnh kèm cỡ / độ đậm / font DỰNG THẬT (không
    cần OCR bản dựng: code biết chính xác đã viết gì ở đâu), lỗi. -> P mới (vá / xoá / thêm lệnh); rào chắn ở slots.accept."""
    by_res = {r["id"]: r for r in res or []}
    rows = []
    for op in P["ops"]:
        r = by_res.get(op["id"]) or {}
        row = {k: op.get(k) for k in ("id", "marks", "kind", "client", "html", "box_style") if op.get(k) not in (None, "")}
        Ls = [M["by"][m] for m in op.get("marks") or [] if m.startswith("L") and m in M["by"]]
        if r.get("fs"):
            row["rendered"] = {"drafted_px": round(float(np.median([m["size"] for m in Ls]))) if Ls else None,
                               "font_px": round(r["fs"]), "weight": r.get("fw"), "font": (r.get("ff") or "").split(",")[0].strip("'\" ")}
            if r.get("shrink", 1) < 0.9 and len(Ls) == 1 and len(Ls[0].get("ocr") or "") >= 4:
                # VÌ SAO nhỏ: font rộng hơn nét nháp (gợi ý font đúng độ rộng) hay chữ dài hơn chỗ nháp vẽ
                from .fonts import WIDTH
                from .render import font_key
                L0 = Ls[0]
                fk = font_key(row["rendered"]["font"])
                dw = (L0["box"][2] - L0["box"][0]) / (len(L0["ocr"]) * L0["size"])
                if fk in WIDTH:
                    row["rendered"]["why_small"] = (f"font {WIDTH[fk][1 if L0.get('caps') else 0] / dw:.2f}x wider than the drafted "
                                                    f"letters; same-width fonts (only if their letter shape matches the draft): "
                                                    f"{', '.join(_fits(dw, L0))}; text "
                                                    f"{len(_plain(op.get('html') or '')) / len(L0['ocr']):.2f}x the drafted length")
        rows.append(row)
    ch = [c for c in changes or [] if not c.startswith("--")]
    user = (f"PROMPT given to the image model:\n{prompt}\n\nClient texts (exact words):\n{_texts(texts)}\n\n"
            + (PRODUCT_NOTE + "\n\n" if product else "") + f"{describe(M)}\n\nStyle: {json.dumps(P.get('style'), ensure_ascii=False)}"
            f"\n\nRendered ops:\n{json.dumps(rows, ensure_ascii=False)}\n\nCODE CHANGES (engine log, op ids):\n"
            + ("\n".join(ch) or "(none)") + "\n\nERRORS:\n"
            + ("\n".join(json.dumps(e, ensure_ascii=False) for e in errs if REVIEW_MODE == "full" or e["type"] in SEVERE) or "(none)"))
    full = REVIEW_MODE == "full"
    imgs = ["DRAFT:", _png(draft), "PROOF:", _png(proof), "PROOF-IDS:", _png(_ids_image(proof, M, P))] + \
        (["COMPARE (draft | proof):", *_compare(draft, proof, M)] if full else [])
    sysmsg = _system(SYSTEM) + "\n\n" + REVIEW + ("" if full else "\n\n" + ONLY_ERRORS)
    out = _obj(call(*((sysmsg, [user, *imgs]) + ((meta,) if meta is not None else ()))))
    new, patches = [], {p.get("id"): p for p in out.get("patches") or [] if isinstance(p, dict) and p.get("id")}
    for op in P["ops"]:
        p = patches.pop(op["id"], None)
        if p is None:
            new.append(op)
        elif p.get("delete") and op.get("kind") == "text" and _tid(op.get("client")) is not None and op.get("html") and                 not _rewritten(op, P["ops"], patches):
            # xoá lệnh đang viết (một phần) câu khách mà không lệnh nào khác viết lại phần đó -> không xoá (08/10 server bánh mì
            # v1: vòng duyệt xoá dòng "Bánh Mì", poster chỉ còn "Cô Ba")
            new.append(op)
        elif p.get("delete"):   # xoá = không vẽ gì; ô vẫn đã quyết (skip), không thành "ô bỏ sót" (08/10 h08: cả lượt bị loại)
            new.append({"id": op["id"], "kind": "skip", "marks": op.get("marks") or [], "why": "review: delete"})
        else:
            q = {**op, **{k: v for k, v in p.items() if k not in ("id", "delete")}}
            if "slots" in p:
                q["marks"] = list(p["slots"] or [])
            new.append({k: v for k, v in q.items() if v is not None or k == "client"})
    for pid, p in patches.items():   # lệnh MỚI (ô chưa quyết)
        if p.get("slots") and not p.get("delete"):
            new.append({**{k: v for k, v in p.items() if k != "delete"}, "marks": list(p["slots"])})
    # ô bị lệnh khác lấy đi (gộp ô) thì bỏ khỏi lệnh cũ; lệnh cũ hết ô thì bỏ
    regrab = {p.get("id") for p in out.get("patches") or [] if isinstance(p, dict) and p.get("slots")}   # lệnh vá có ô mới thắng
    owner = {}
    for op in sorted(new, key=lambda o: o["id"] not in regrab):
        for m in op.get("marks") or []:
            owner.setdefault(m, op["id"])
    new = [{**op, "marks": [m for m in op.get("marks") or [] if owner.get(m) == op["id"]]} for op in new]
    new = [op for op in new if op["marks"]]
    return _to_marks({**P, "ops": new, "review_why": out.get("why"), "review_findings": out.get("findings"),
                      "review_patches": len(out.get("patches") or [])})


REVIEW_ROUNDS = 2   # chế độ "full": dựng -> duyệt -> dựng -> duyệt; dừng sớm khi VLM không sửa gì / bản duyệt bị loại
# CHẾ ĐỘ (08/10, sau lượt 3-4: duyệt thẩm mỹ tốn ~20 s / poster, ~2 / 15 poster đẹp hơn, ~5 xấu đi -> không đáng):
#   "errors" (mặc định) -- MỘT lượt, CHỈ khi lượt đầu có lỗi nặng (SEVERE), chỉ sửa đúng các lỗi đó (~1 / 3 poster)
#   "full"   -- duyệt thẩm mỹ cả tấm, REVIEW_ROUNDS lượt, có giám khảo (slots.judge)
#   "off"    -- không bao giờ gọi VLM lần hai (độ trễ cố định; lỗi nặng còn nguyên trên poster)
# 08/10 (sau đợt server 1): mặc định "off" -- lỗi nặng sửa bằng code (render.autofix); bật lại bằng TEXTFIX_REVIEW=errors
# nếu quan sát thấy code không cứu được
REVIEW_MODE = os.environ.get("TEXTFIX_REVIEW", "off")
ONLY_ERRORS = """MODE: FIX ONLY THE ERRORS. Fix exactly the ERRORS listed and nothing else -- every op not involved in an error \
stays untouched; no aesthetic changes. Output "findings" for the errors only."""


def review_rounds() -> int:
    return REVIEW_ROUNDS if REVIEW_MODE == "full" else 1


def review_needed(errs: list[dict]) -> bool:
    """Có chạy vòng duyệt cho bản này không (theo REVIEW_MODE)."""
    if REVIEW_MODE == "off":
        return False
    return REVIEW_MODE == "full" or any(e["type"] in SEVERE for e in errs)
SEVERE = ("empty_plan", "unassigned", "content_overlap", "boxes_overlap", "overflow", "low_contrast", "mark_twice")


JUDGE = """You are an art director choosing which of two finished renders of the same poster to send to the client. The DRAFT \
shows the intended look. Pick the better poster: closer to the draft's look (type style, weight, size, colors, layout), \
cleaner, every text easy to read, nothing cut off, overlapping, crammed, tiny or orphaned, badges / pills / buttons properly \
filled. Judge only what you see. Reply with one JSON object: {"better": 1 or 2, "why": "a few words"}."""


def judge(draft: np.ndarray, before: np.ndarray, after: np.ndarray, call, meta: dict | None = None) -> tuple[bool, str]:
    """GIÁM KHẢO thẩm mỹ cho vòng duyệt (08/10 lượt 4: bản duyệt qua được rào đo lường nhưng xấu đi -- vòng tròn -50% phình
    cắt chữ, huy hiệu lộn xộn hơn): VLM so trước / sau cạnh nhau (thứ tự xáo theo nội dung ảnh, chống thiên vị vị trí) -> True
    khi bản SAU đẹp hơn. Lỗi / trả sai khuôn -> giữ bản sau (rào đo lường đã qua)."""
    import hashlib
    swap = hashlib.md5(after[::16, ::16].tobytes()).digest()[0] % 2 == 1
    one, two = (after, before) if swap else (before, after)
    try:
        out = _obj(call(*((JUDGE, ["DRAFT:", _png(draft), "POSTER 1:", _png(one), "POSTER 2:", _png(two)])
                         + ((meta,) if meta is not None else ()))))
        b = int(str(out.get("better")).strip()[:1])
        assert b in (1, 2)
    except (ValueError, TypeError, AttributeError, AssertionError):
        return True, "judge: không đọc được -> giữ bản duyệt"
    return (b == 1) == swap, str(out.get("why") or "")[:200]


def salvage(P: dict, errs: list[dict], P2: dict, errs2: list[dict]) -> dict | None:
    """Bản duyệt bị loại (accept = False) thường chỉ hỏng ở một hai lệnh, phần còn lại là sửa tốt (08/10 h08: xoá một huy hiệu
    làm ô bỏ sót -> mất cả ba huy hiệu đã dựng lại đẹp). Hoàn lại ĐÚNG các lệnh gây lỗi nặng mới / làm mất câu khách về bản
    trước, giữ các vá khác. None khi không có gì để cứu."""
    key = lambda e: (e["type"], tuple(sorted(e.get("ops") or [])), tuple(sorted(e.get("slots") or [])))   # noqa: E731
    seen = {key(e) for e in errs if e["type"] in SEVERE}
    old = {o["id"]: o for o in P["ops"]}
    bad = set()
    for e in errs2:
        if e["type"] in SEVERE and key(e) not in seen:
            bad |= set(e.get("ops") or [])
            bad |= {o["id"] for o in P["ops"] for s in e.get("slots") or [] if s in (o.get("marks") or [])}
    told = lambda Q: {_tid(o.get("client")): o["id"] for o in Q["ops"] if o.get("kind") == "text" and o.get("html")}   # noqa: E731
    lost = set(told(P)) - set(told(P2)) - {None}
    bad |= {i for c, i in told(P).items() if c in lost}
    if not bad:
        return None
    new = [old[o["id"]] if o["id"] in old and o["id"] in bad else o for o in P2["ops"] if o["id"] in old or o["id"] not in bad]
    new += [old[i] for i in bad if i in old and i not in {o["id"] for o in new}]
    owner = {}
    for op in sorted(new, key=lambda o: o["id"] not in bad):   # lệnh hoàn lại lấy lại ô của nó
        for m in op.get("marks") or []:
            owner.setdefault(m, op["id"])
    new = [{**op, "marks": [m for m in op.get("marks") or [] if owner.get(m) == op["id"]]} for op in new]
    return {**P2, "ops": [op for op in new if op["marks"]], "salvaged": sorted(bad)}


def accept(P: dict, errs: list[dict], P2: dict, errs2: list[dict]) -> bool:
    """Giữ bản sau vòng duyệt? Sửa thẩm mỹ không đo được, nên tin VLM -- trừ khi nó làm hỏng điều đo được: viết ít câu khách hơn,
    hoặc thêm lỗi NẶNG (ô bỏ sót, chồng nhau, tràn, chìm nền). too_small / uneven_size được phép tăng (chữ to lên trong huy hiệu
    có thể chạm sàn co)."""
    sev = lambda E: sum(1 for e in E if e["type"] in SEVERE)   # noqa: E731
    return score(P2, [])[0] <= score(P, [])[0] and sev(errs2) <= sev(errs)


# ---------------------------------------------------------------------------------------------------- nền
