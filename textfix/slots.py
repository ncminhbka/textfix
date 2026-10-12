"""DESIGNER "ĐIỀN Ô": nháp - bản xoá = lớp phủ model đã vẽ -> các Ô đúng toạ độ; VLM chỉ quyết mỗi ô là gì và điền cho đẹp; code
dựng đúng khung từng ô (render.py), theo lớp vỏ -> chi tiết -> chữ.

  M = build(draft, plate)                         # S# vỏ, I# chi tiết nhỏ, L# dòng chữ (OCR nguyên khung, số đo nét)
  P = plan(draft, M, prompt, texts, call)         # VLM: {style, ops: [{slots, kind, html, box_style, client}], missing}
  P, log = validate(P, M, texts)                 # + chặn chữ: dấu theo câu khách, chữ bịa, dòng lặp
  base = base_image(draft, plate, M, P)           # bản xoá + dán lại pixel nháp ở dòng "keep" (chữ in trên sản phẩm / màn hình)
  P2 = repair(draft, poster, M, P, errs, prompt, texts, call)   # có lỗi (render.check): một vòng, chỉ các lệnh / ô lỗi

VLM không đưa toạ độ: chỉ trỏ id ô. Mỗi ô thuộc đúng một lệnh (lệnh chữ được gộp nhiều dòng liền nhau thành một khối).
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


def lines(img: np.ndarray) -> list[dict]:
    """Dòng OCR + số đo nét: cao thân chữ -> cỡ chữ (px), màu, góc nghiêng."""
    import cv2
    from .glyph import measure
    from .ocr import read_lines
    H, W = img.shape[:2]
    # "dòng" không có chữ cái / chữ số (hàng sao "★★★★★", gạch, chấm) không phải chữ: để lớp phủ tách thành chi tiết I
    raw = [l for l in read_lines(img) if any(ch.isalnum() for ch in l["text"])]
    raw = [q for l in raw for q in _split_gaps(img, l)]
    L = []
    for i, l in enumerate(raw, 1):
        ang = float(l.get("angle") or 0.0)
        if abs(ang) >= 2:   # dòng NGHIÊNG: xoay thẳng ảnh quanh dòng rồi đo (khung thẳng của dòng nghiêng làm cao thân chữ phồng)
            P_ = np.asarray(l["poly"], np.float32)
            c = P_.mean(0)
            R = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0)
            img_r = cv2.warpAffine(img, R, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            q = np.hstack([P_, np.ones((4, 1), np.float32)]) @ R.T
            m = measure(img_r, [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())], l["text"], [])
        else:
            m = measure(img, l["box"], l["text"], [o["box"] for o in raw if o is not l])
        lt = [c for c in l["text"] if c.isalnum()]
        caps = bool(lt) and sum(c.isupper() or c.isdigit() for c in lt) >= 0.8 * len(lt)
        h = m["h"] if m else 0.7 * (l["box"][3] - l["box"][1])
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
                  "effect": {k: v for k, v in (m.get("effect") or {}).items() if k in ("dx", "dy", "color")} if m and m.get("effect") else None})
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
    ink = (np.linalg.norm(lab - np.median(border, 0), axis=2) >= INK_DE).any(0)
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
        part = part.strip()
        if not part or b - a < 2:
            continue
        bx = [float(x0 + a), float(y0), float(x0 + b), float(y1)]
        out.append({**l, "text": part, "box": bx, "poly": [[bx[0], bx[1]], [bx[2], bx[1]], [bx[2], bx[3]], [bx[0], bx[3]]]})
    return out or [l]


def build(draft: np.ndarray, plate: np.ndarray) -> dict:
    """Ô của nháp: L# (dòng OCR nguyên khung + số đo nét, vỏ chứa nó), S# vỏ, I# chi tiết nhỏ (overlay.overlay)."""
    import cv2
    from .overlay import overlay
    H, W = draft.shape[:2]
    L = lines(draft)
    O = overlay(draft, plate, [{"box": l["box"], "poly": l["poly"]} for l in L])
    A = _lab(draft)
    for j, l in enumerate(L):
        sh = next((s for s in O["S"] if j in s["lines"]), None)
        l["shell"] = sh["id"] if sh else None
        # KHUNG DÒNG = KHUNG OCR, nguyên bản (12/10: bỏ hẳn bước nắn khung theo nét mực -- trên nền ảnh chụp nét mực đo thiếu,
        # khung bị cắt hẹp / đẩy mép trái vào mỗi dòng một kiểu: 11-25% dòng, danh sách thò thụt, chữ co). Bản xoá là xoá cả
        # ảnh nên dấu thanh nhô khỏi khung không đè vào đâu; vòng co của render chỉ kiểm tràn NGANG với dòng đơn.
        l["ocr_box"] = list(l["box"])
    # VỎ GIẢ (12/10 dev3 2871 "Pool Party": một mảng chữ tiêu đề to, đặc, bị coi là vỏ -> VLM vẽ mảng trắng sau chữ): màu lòng
    # gần trùng màu mực của dòng nằm trong nó -> đó là chữ, bỏ ô vỏ
    fake = [s for s in O["S"] if s.get("fill") not in (None, "none") and any(
        L[j].get("color") and float(np.linalg.norm(_hex_lab(L[j]["color"]) - s["_fill_lab"])) < SHELL_INK_DE for j in s["lines"])]
    for s in fake:
        for j in s["lines"]:
            L[j]["shell"] = None
    O["S"] = [s for s in O["S"] if not any(s is f for f in fake)]
    for c in O["I"]:
        if any(c.get("shell") == f["id"] for f in fake):
            c["shell"] = None
    S =[{k: v for k, v in s.items() if not k.startswith("_")} for s in O["S"]]
    for s in S:
        s["lines"] = [L[j]["id"] for j in s["lines"]]
    shells = {s["id"]: {"mask": s["_mask"], "fill_lab": s["_fill_lab"]} for s in O["S"]}   # cho "keep" vỏ (không ghi JSON)
    # chi tiết không bao giờ ĐÈ lên chữ; bỏ chi tiết nằm trong khung dòng (gạch đầu dòng OCR gộp vào dòng) và lồng trong chi tiết khác
    I = [c for c in O["I"] if all(_cov(c["box"], l["box"]) < 0.3 and _cov(l["box"], c["box"]) < 0.5 for l in L)]
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])   # noqa: E731
    I = [c for c in I if not any(o is not c and _cov(o["box"], c["box"]) >= 0.8 and area(o["box"]) > area(c["box"]) for o in I)]
    I = _rows(I)
    if L:   # vụn nhỏ hơn nửa chữ thân không phải chi tiết
        lh = float(np.median([l["box"][3] - l["box"][1] for l in L]))
        I = [c for c in I if max(c["box"][2] - c["box"][0], c["box"][3] - c["box"][1]) >= DETAIL_MIN * lh]
    for k, c in enumerate(I, 1):
        c["id"] = f"I{k}"
    # HÌNH HỌC CHỮ (geo.py): chữ chạy theo cung (OCR cắt mẩu / bỏ hai đầu cung) -> một ô L cung; một dòng nhiều cỡ OCR tách mẩu
    # -> một ô L có runs. Nét chữ dò cung: lớp phủ ngoài vỏ + trong vỏ điểm khác màu lòng vỏ.
    from . import geo
    glog, glog2 = [], []
    if geo.ENABLED:   # TẮT mặc định tới khi dựng (B4) + mô tả cho VLM (B3) xong: ô cung gộp mà dựng thẳng còn tệ hơn mẩu OCR
        from .overlay import DIFF
        Bp = _lab(plate)
        ink = cv2.GaussianBlur(np.linalg.norm(A - Bp, axis=2), (0, 0), 1.0) > DIFF   # lớp phủ thô (mở 5 px làm vỡ nét mảnh)
        for sh in O["S"]:
            ink[sh["_mask"]] = (np.linalg.norm(A - sh["_fill_lab"], axis=2) > INK_IN_SHELL)[sh["_mask"]]
        L, I, glog = geo.apply_arcs(L, I, geo.arcs(ink, W, H), CAP_EM)
        L, glog2 = geo.runs(L)
    for s in S:   # dòng trong vỏ: tính lại theo tâm ô (ô đã gộp)
        x0, y0, x1, y1 = s["box"]
        s["lines"] = [l["id"] for l in L if x0 <= (l["box"][0] + l["box"][2]) / 2 <= x1 and y0 <= (l["box"][1] + l["box"][3]) / 2 <= y1]
    for s in S:
        s["icons"] = [c["id"] for c in I if c["shell"] == s["id"]]
    for s in S:   # vỏ này GIỮ được điểm ảnh nháp không (lấp được nét chữ cũ bằng màu lòng)
        s["keepable"] = _keepable(A, shells[s["id"]], [l["box"] for l in L if l["id"] in s["lines"]])
    for c in I:   # quan hệ với dòng gần nhất (gợi ý cho VLM: avatar cạnh tên, sao dưới tên, icon đầu dòng)
        cx, cy = (c["box"][0] + c["box"][2]) / 2, (c["box"][1] + c["box"][3]) / 2
        best = None
        for l in L:
            x0, y0, x1, y1 = l["box"]
            dx, dy = max(x0 - cx, 0, cx - x1), max(y0 - cy, 0, cy - y1)
            d = float(np.hypot(dx, dy))
            if best is None or d < best[0]:
                side = "left of" if cx < x0 and dx >= dy else "right of" if cx > x1 and dx >= dy else "above" if cy < y0 else \
                    "below" if cy > y1 else "on"
                best = (d, l["id"], side)
        c["near"] = f"{best[2]} {best[1]}" if best and best[0] <= 3 * max(8.0, c["box"][3] - c["box"][1]) else None
    # mask: lớp phủ (nháp - bản xoá đã dọn), cho "keep" dán lại trọn mảng bị xoá (không ghi JSON)
    return {"L": L, "S": S, "I": I, "size": [W, H], "by": {m["id"]: m for m in L + S + I}, "mask": O["mask"], "shells": shells,
            "geo_log": glog + glog2 + [f"vỏ giả (màu lòng = màu chữ) {[round(v) for v in f['box']]} -> bỏ ô vỏ" for f in fake]}


def _hex_lab(hx: str) -> np.ndarray:
    return _lab(np.uint8([[[int(hx[i:i + 2], 16) for i in (1, 3, 5)]]]))[0, 0]


def _line_zone(shape, box) -> tuple[slice, slice]:
    """Vùng nét của một dòng: khung OCR nới 0.3 cao mỗi bên, 0.6 cao phía trên (dấu thanh nhô khỏi khung)."""
    H, W = shape[:2]
    x0, y0, x1, y1 = box
    h = y1 - y0
    return (slice(int(max(0, y0 - 0.6 * h)), int(min(H, y1 + 0.3 * h))), slice(int(max(0, x0 - 0.3 * h)), int(min(W, x1 + 0.3 * h))))


def _shell_ink(A: np.ndarray, sh: dict, boxes: list) -> np.ndarray:
    """Mặt nạ NÉT CHỮ CŨ trong một vỏ: điểm trong vỏ, trong vùng nét của các dòng, khác màu lòng vỏ >= INK_IN_SHELL."""
    ink = np.zeros(A.shape[:2], bool)
    for b in boxes:
        ys, xs = _line_zone(A.shape, b)
        ink[ys, xs] = (np.linalg.norm(A[ys, xs] - sh["fill_lab"], axis=2) >= INK_IN_SHELL) & sh["mask"][ys, xs]
    return ink


def _keepable(A: np.ndarray, sh: dict, boxes: list) -> bool:
    if not boxes:
        return True
    zone = np.zeros(A.shape[:2], bool)
    for b in boxes:
        zone[_line_zone(A.shape, b)] = True
    zone &= sh["mask"]
    return bool(zone.sum() >= 20 and _shell_ink(A, sh, boxes)[zone].mean() <= KEEP_INK_MAX)


def attach_shells(M: dict, draft: np.ndarray, plate: np.ndarray) -> dict:
    """M đọc từ plan.json (không có mặt nạ): tính lại lớp phủ từ nháp + bản xoá + khung dòng đã lưu, gắn mặt nạ / màu lòng cho
    từng ô vỏ đã lưu (khớp theo khung) -> M["shells"], S["keepable"]. Dùng cho dựng lại tại máy (replay, probe)."""
    from .overlay import overlay
    box = lambda l: l.get("ocr_box") or l["box"]   # noqa: E731
    poly = lambda b: [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]   # noqa: E731
    O = overlay(draft, plate, [{"box": box(l), "poly": l.get("poly") or poly(box(l))} for l in M["L"]])
    A = _lab(draft)
    M["shells"] = {}
    M.setdefault("mask", O["mask"])
    for s in M["S"]:
        best = max(O["S"], key=lambda o: min(_cov(o["box"], s["box"]), _cov(s["box"], o["box"])), default=None)
        if best is None or min(_cov(best["box"], s["box"]), _cov(s["box"], best["box"])) < 0.6:
            s["keepable"] = False
            continue
        M["shells"][s["id"]] = {"mask": best["_mask"], "fill_lab": best["_fill_lab"]}
        s["keepable"] = _keepable(A, M["shells"][s["id"]], [M["by"][i]["box"] for i in s.get("lines", []) if i in M["by"]])
    return M


def public(M: dict) -> dict:
    """Ô để ghi JSON."""
    return {"L": [{k: m[k] for k in ("id", "box", "ocr_box", "ocr", "size", "color", "angle", "shell", "curve", "runs", "geo", "path", "scale") if k in m}
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
        x0, y0, x1, y1 = m["box"]
        x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)   # ảnh debug không được làm hỏng phiên bản
        d.rectangle([x0, y0, x1, y1], outline=col, width=wd)
        tw, th = d.textbbox((0, 0), m["id"], font=f)[2:]
        tx, ty = (x0, max(0, y0 - th - 2)) if corner == "tl" else (max(0, x1 - tw - 2), min(img.shape[0] - th - 2, y1 + 1))
        d.rectangle([tx, ty, tx + tw + 2, ty + th + 2], fill=col)
        d.text((tx + 1, ty), m["id"], fill=(255, 255, 255), font=f)
    return im


# ---------------------------------------------------------------------------------------------------- VLM
SYSTEM = """You are a senior graphic designer finishing a poster. An image model was given the PROMPT below and drew the DRAFT. \
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
- "keep" never applies to POSTER lettering, however decorated it is \
(a headline on a ribbon or banner, curved, 3D or metallic letters, badge and sticker text): the draft's letters are often \
misspelt, so it is always rewritten as a text op with the client's spelling.
- A product's screen, display, dial, label or packaging text is part of the picture: "keep" it (or "skip"), never write \
words or numbers there.
- Every slot id appears in exactly ONE op -- never leave a slot out (use "skip" to drop it).

TEXT
- Decide what the model meant to write on each line and write it at about the size it drew, keeping its line breaks (<br> \
between the lines of a multi-line op). Write about as many characters per line as the model wrote there -- never much more: \
crammed text gets shrunk and looks bad.
- A client text longer than the lines the model gave it: shorten or rephrase it so it fits naturally; facts (names, prices, \
numbers, dates, phones, addresses, emails, links) are never changed, abbreviated or invented -- keep them exact (Quận 3 \
stays "Quận 3", never "Q.3") or leave them out; the unit belongs to the number (15-25 triệu, 35.000đ, 30 phút, 50%): never drop it. Set \
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
- COLUMNS: slot numbers run row by row across the whole poster, so two text blocks standing side by side have their lines \
numbered alternately (L6 left, L7 right, L8 left, L9 right). Never merge by consecutive numbers: check the boxes. The slots of \
one op must be stacked one under the other in the SAME column (their x ranges overlap); lines side by side at the same height \
with a wide gap between them belong to different blocks and different ops. (Only pieces of ONE drawn line that OCR cut in \
two -- same height, same size, a word gap apart -- go together.)
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
- Size: write every line at the size measured for its slot; fine print, labels and footers too -- never smaller "to be safe". \
If a text does not fit, shorten it (facts stay exact) rather than shrinking it.
- Lists: keep the draft's list structure -- one item per row, each row keeps its own text next to its bullet / check / icon; \
never merge list rows into one block and never leave a bullet or icon without its text. Every list row (bullet / feature / step) starts with a capital letter, and all rows of one list carry the same "group" name with "group_kind": "list".
- Row marks: when the draft shows a bullet / check / small icon in front of a row, look at the BASE image. Mark still there \
-> write only the text. Mark erased and no I# slot for it -> draw it yourself at the start of that row's html ("• " or an \
<i-icon>), the same mark on every row of the group. Copy the mark the draft drew: a small dot -> "• " (a text bullet, the \
size of the text); an <i-icon> only where the draft drew a real icon or check mark, never a big ring for a dot.
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
- Text in a pill / button / badge / band: the shell fixes the room. A sentence much longer than the line the model drew there \
is shortened to its key words (facts exact: "Tặng buổi tư vấn đầu cho khách mới") so it stays at the drawn size -- never a \
full sentence shrunk to fine print inside its shell.
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


def describe(M: dict) -> str:
    sh = "\n".join(f"{s['id']}: shell {_px(s['box'])}, "
                   + (f"outline only (no fill) {s['border']} ~{s['border_px']:.0f}px" if s.get("border") else f"fill {s['fill']}")
                   + f", radius ~{s['radius']:.0f}px, contains "
                   f"{', '.join(s['lines'] + s['icons']) or 'nothing'}" for s in M["S"]) or "(none)"
    de = "\n".join(f"{c['id']}: detail {_px(c['box'])}, color {c['color']}" + (f", inside {c['shell']}" if c.get("shell") else "")
                   + (f", {c['near']}" if c.get("near") else "") for c in M["I"]) or "(none)"
    tilt = lambda m: f", drawn at {m['angle']:.0f} degrees" if abs(m.get("angle") or 0) >= 2 else ""   # noqa: E731
    li = "\n".join(f"{m['id']}: \"{m['ocr']}\" (text ~{m['size']:.0f}px, ~{len(m['ocr'])} characters, color {m['color']}"
                   + (f", inside {m['shell']}" if m.get("shell") else "") + f"{tilt(m)}{_look(m)})" for m in M["L"]) or "(none)"
    return f"Shells:\n{sh}\n\nDetails:\n{de}\n\nText lines:\n{li}"


def _system(base: str) -> str:
    from .fonts import CATALOG, WIDTH
    from .icons import available
    fonts = "; ".join(f"{k}: {v[0]} ({v[1]}" + (f", {WIDTH[k][0]:.2f} / caps {WIDTH[k][1]:.2f}em" if k in WIDTH else "") + ")"
                      for k, v in CATALOG.items())
    return base.replace("{fonts}", fonts).replace("{icons}", ", ".join(sorted(available())))


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
                "graphics printed on that product (label, package, screen) are part of the photo: give every slot on them kind "
                "\"keep\", never redraw or skip them.")


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
         view: str | None = None, meta: dict | None = None) -> dict:
    """base: bản xoá đã dọn (nền dựng) cho các cấu hình có X. meta: nhận số token / thời gian (llm.json_call)."""
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
            imgs += ["BASE (erased poster, your slots are drawn on it):", _png(base)]
        elif c == "Z":
            imgs += ["ZOOM (slot clusters of SLOTS, enlarged):", *_clusters(mk, M)]
    user = (f"PROMPT given to the image model:\n{prompt}\n\nClient texts (exact words):\n{_texts(texts)}\n\nPoster size: {W}x{H}px.\n\n"
            + (PRODUCT_NOTE + "\n\n" if product else "") + f"{describe(M)}")
    args = (_system(SYSTEM), [user, *imgs]) + ((meta,) if meta is not None else ())
    return _to_marks(call(*args))


STAR_WORDS = re.compile(r"(>)[\s(]*\d+(?:[.,]\d+)?\s*(?:sao|stars?)[\s)]*(<)", re.I)   # mẩu chữ CHỈ ghi số sao, giữa hai thẻ
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
    for op in P.get("ops", []):
        mk = [m for m in op.get("marks") or [] if m in by]
        if len(mk) < len(op.get("marks") or []):
            log.append(f"{op.get('id')}: bỏ id không có {sorted(set(op.get('marks') or []) - set(mk))}")
        if not mk:
            continue
        # chữ / logo IN TRÊN SẢN PHẨM mà VLM cho skip (tự ghi lý do "printed logo on product") -> bản xoá đã xoá nó, skip là mất
        # luôn (09/10 dev2 d25: con dấu trên bánh pía) -> keep (dán lại pixel nháp)
        # CHỈ khi khách tải ảnh sản phẩm: không có ảnh sản phẩm thì "chữ in trên ruy băng / sản phẩm" là chữ POSTER FLUX vẽ (09/10
        # dev3 g15: dải ruy băng "QUÀ TẶNG TRI ÂN" -> keep dán lại chữ nháp sai chính tả "QUẢ TỆNG")
        if product and op.get("kind") == "skip" and PRINTED.search(op.get("why") or "") and                 not re.search(r"duplicat|junk|garbl", op.get("why") or "", re.I):
            log.append(f"{op.get('id')}: skip chữ in trên sản phẩm ({op.get('why')!r}) -> keep")
            op = {**op, "kind": "keep"}
        # KEEP CHỮ KHÁCH: ô giữ pixel nháp mà chữ OCR của nó gần trùng một câu khách = chữ poster model vẽ (thường sai chính tả)
        # -> viết lại bằng câu khách (09/10 dev3 g15 v0: VLM keep "QUÀ TẶNG TRI ÂN" trên ruy băng)
        if op.get("kind") == "keep" and texts:
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
        idx = sorted(idx, key=lambda i: -SequenceMatcher(None, ocr(i), f).ratio())
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
SEVERE = ("unassigned", "content_overlap", "boxes_overlap", "overflow", "low_contrast", "mark_twice")


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
def base_image(draft: np.ndarray, plate: np.ndarray, M: dict, P: dict) -> np.ndarray:
    """Bản xoá + DÁN LẠI pixel nháp ở các ô "keep" (chữ / logo / hình in trên sản phẩm, màn hình: bản xoá đã xoá cả chúng).
    Vùng dán = khung ô nới 0.25 cao + TRỌN các mảng lớp phủ chạm khung (vòng con dấu, phần logo ngoài khung dòng), trong giới
    hạn khung nới 1 cao ô mỗi phía (mảng lớp phủ to chạy xa -- dải, thẻ -- không bị kéo theo)."""
    import cv2
    H, W = draft.shape[:2]
    m = np.zeros((H, W), np.uint8)
    ov = M.get("mask")
    cc = None
    if ov is not None:
        _, cc = cv2.connectedComponents(cv2.morphologyEx(ov.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), connectivity=8)
    for op in P.get("ops", []):
        if op.get("kind") != "keep":
            continue
        for i in op.get("marks") or []:
            if i not in M["by"] or i in (M.get("shells") or {}):   # vỏ có mặt nạ: dán theo mặt nạ ở dưới, không dán cả khung
                continue
            x0, y0, x1, y1 = M["by"][i]["box"]
            h = y1 - y0
            p = 0.25 * h
            m[int(max(0, y0 - p)):int(min(H, y1 + p)), int(max(0, x0 - p)):int(min(W, x1 + p))] = 1
            if cc is not None:
                X0, Y0, X1, Y1 = int(max(0, x0 - h)), int(max(0, y0 - h)), int(min(W, x1 + h)), int(min(H, y1 + h))
                ids = np.unique(cc[int(max(0, y0)):int(min(H, y1)), int(max(0, x0)):int(min(W, x1))])
                ids = ids[ids > 0]
                if len(ids):
                    sel = np.isin(cc[Y0:Y1, X0:X1], ids)
                    m[Y0:Y1, X0:X1] |= cv2.dilate(sel.astype(np.uint8), np.ones((5, 5), np.uint8))
    # VỎ GIỮ (12/10): vỏ S "keep" -> dán điểm ảnh nháp theo MẶT NẠ vỏ (dáng, viền, bóng, chuyển màu đúng như nháp), rồi lấp nét
    # chữ cũ trong vùng các dòng của nó bằng nền quanh nét (inpaint) -- chữ mới do lệnh chữ viết lên. Dòng "keep" (chữ in trên
    # vật) không lấp.
    kept = {i for op in P.get("ops", []) if op.get("kind") == "keep" for i in op.get("marks") or []}
    # chi tiết "keep" (đường viền, hoạ tiết, mảng ảnh) kéo theo mảng lớp phủ chạm khung: không được kéo theo chữ nháp của dòng
    # không giữ (chữ méo dán lại lên poster)
    if m.any():
        for l in M["L"]:
            if l["id"] not in kept:
                ys, xs = _line_zone((H, W), l["box"])
                m[ys, xs] = 0
    wipe = np.zeros((H, W), bool)
    for sid in sorted(i for i in kept if i in (M.get("shells") or {})):
        sh = M["shells"][sid]
        m |= cv2.dilate(sh["mask"].astype(np.uint8), np.ones((7, 7), np.uint8))
        boxes = [M["by"][i]["box"] for i in M["by"][sid].get("lines", []) if i in M["by"] and i not in kept]
        if boxes:   # không lấp sát mép vỏ: nét viền của vỏ (vỏ chỉ có viền, vỏ có viền màu) khác màu lòng nhưng không phải chữ
            e = 2 * int(3 + (M["by"][sid].get("border_px") or 0)) + 1
            wipe |= _shell_ink(_lab(draft), sh, boxes) & cv2.erode(sh["mask"].astype(np.uint8), np.ones((e, e), np.uint8)).astype(bool)
    if not m.any():
        return plate.copy()
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0)[..., None]
    out = (draft * a + plate * (1 - a)).round().clip(0, 255).astype(np.uint8)
    if wipe.any():
        k = 2 * KEEP_HALO + 1
        out = cv2.inpaint(out, cv2.dilate(wipe.astype(np.uint8), np.ones((k, k), np.uint8)), 3, cv2.INPAINT_TELEA)
    return out
