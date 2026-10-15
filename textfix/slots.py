"""DÒNG CHỮ của nháp: code đo mỗi dòng chữ thành một ô L# đúng toạ độ (OCR + số đo nét + chữ cong + cụm nét "orphan"), vẽ ảnh
đánh dấu, gửi VLM một lời gọi. Ghép câu <-> dòng và dựng ở fill.py / render.py (chặng B).

  M = build(draft, ink)                           # L# dòng chữ
  A = fill.parse(plan(draft, M, prompt, texts, call, system, desc, base))   # VLM: {assign, styles, accent, missing}

VLM không đưa toạ độ: chỉ trỏ id dòng.
"""

from __future__ import annotations

import io
import json
import re

import numpy as np

CAP_EM = 0.72        # cao chữ hoa / em (cũng ~ cao chữ số, nét lên b d h k l t)
TALL_MIN = 0.15      # dòng có >= 15% ký tự cao -> ước cỡ theo cao nét cao (glyph h_tall)
X_EM = 0.53          # cao chữ thường (x-height) / em
GAP_SPLIT = 1.0      # trong một dòng OCR, khoảng KHÔNG nét rộng >= 1 cao dòng = hai mục riêng (khoảng cách từ ~0.3 cao dòng;
                     # OCR gộp "địa chỉ [icon] website" ở chân trang thành một dòng)
LEADER_H = 0.3       # hàng chấm nối: cột có nét cao <= 0.3 cao dòng là chấm / gạch nối, coi như khoảng trống
GAP_MIN = 0.4        # khoảng không nét quanh chỗ OCR chèn dấu cách kép: >= 0.4 cao dòng (khoảng cách từ thường ~0.3)
ICON_W = 1.6         # khối kẹp giữa hai khoảng không ở chỗ cắt, rộng <= 1.6 cao dòng = icon giữa hai mục
INK_DE = 25.0        # điểm nét = khác màu nền khung dòng (trung vị viền khung) >= 25 ΔE
from .color import hex_lab as _hex_lab, lab as _lab  # noqa: E402,F401


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


VERT_TALL = 2.2    # dòng viết DỌC theo khung OCR: cao >= 2,2 lần rộng, đọc được >= 3 ký tự (khi nét không đủ để nói hướng)


def _vertical(img: np.ndarray, l: dict, sure: bool) -> dict:
    """Dòng viết DỌC (chữ chạy dọc mép poster, dọc bao bì): OCR trả khung cao - hẹp với góc 0 và chữ rác (đọc sai chiều). Đọc lại
    theo hai chiều +90 / -90, lấy chiều tin hơn: ghi góc, chữ đọc được và đa giác theo chiều đọc.
    sure: nét đã nói dòng dọc (lines.angle) -> dọc kể cả khi bộ đọc không ra chữ (chữ trên bao bì); không thì chỉ khi khung
    OCR cao >= VERT_TALL lần rộng và bộ đọc ra chữ. Không phải dòng dọc: trả nguyên."""
    from .ocr import MIN_REC, read_rect
    b = l["box"]
    w, h = b[2] - b[0], b[3] - b[1]
    if not sure and (h < VERT_TALL * w or sum(ch.isalnum() for ch in l["text"]) < 3):
        return l
    c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    best = max(((*read_rect(img, c, h, w, a), a) for a in (90.0, -90.0)), key=lambda t: t[1])
    readable = best[1] >= MIN_REC and any(ch.isalnum() for ch in best[0])
    if not (readable or sure):
        return l
    x0, y0, x1, y1 = b
    poly = [[x1, y0], [x1, y1], [x0, y1], [x0, y0]] if best[2] > 0 else [[x0, y1], [x0, y0], [x1, y0], [x1, y1]]
    return {**l, "text": best[0] if readable else l["text"], "angle": best[2], "poly": poly, "vertical": True}


def orient(img: np.ndarray, ink: np.ndarray, raw: list[dict]) -> list[dict]:
    """Hướng mỗi dòng OCR (raw: {box, poly, angle = góc OCR, text}) theo nét + OCR (textfix.lines); dòng dọc đọc lại để chọn
    chiều. Sửa raw tại chỗ (angle, và text / poly / vertical với dòng dọc). -> số đo nét từng dòng (lines.measure_lines)."""
    from .erase import line_region
    from .lines import Ink, measure_lines
    H, W = img.shape[:2]
    G = measure_lines(img, Ink(ink), [line_region(l, H, W, 0.1) for l in raw], [float(l.get("angle") or 0.0) for l in raw],
                      [l["box"] for l in raw])
    for k, (l, g) in enumerate(zip(raw, G)):
        l["angle"] = g["angle"]
        if g["x0"] is not None:   # khung bộ dò ôm cả thứ đứng trước chữ: khung dòng bắt đầu từ nét chữ đầu tiên (lines.lead)
            b = l["box"]
            l["box"] = [g["x0"], b[1], b[2], b[3]]
            l["poly"] = [[g["x0"], b[1]], [b[2], b[1]], [b[2], b[3]], [g["x0"], b[3]]]
            l["lead_icon"] = True
        if g["shape"] != "tilt":
            raw[k] = _vertical(img, l, g["shape"] == "vertical")
    return G


def _ink(g: dict) -> dict:
    """Số đo nét ghi kèm dòng (plan.json): bỏ phần đã dùng (kiểu, góc) và id mảnh (chỉ có nghĩa trong lượt chạy)."""
    return {k: v for k, v in g.items() if k not in ("shape", "angle", "ids", "x0")}


def reorient(img: np.ndarray, ink: np.ndarray, L: list[dict]) -> None:
    """orient cho dòng đã lưu trong plan.json (dựng lại lượt cũ theo luật hiện hành): góc gốc là angle_ocr; dòng cong giữ nguyên."""
    todo = [l for l in L if not l.get("path")]
    raw = [{"box": l["box"], "poly": l.get("poly"), "angle": l.get("angle_ocr", l.get("angle")) or 0.0, "text": l.get("ocr") or ""}
           for l in todo]
    G = orient(img, ink, raw)
    for l, r, g in zip(todo, raw, G):
        l.update(box=r["box"], angle=r["angle"], poly=r["poly"], ocr=r["text"], ink=_ink(g))
        if r.get("lead_icon"):
            l["lead_icon"] = True
        l.pop("vertical", None)
        if r.get("vertical"):
            l["vertical"] = True
        l["shape"] = shape_of(l)


def lines(img: np.ndarray, ink: np.ndarray) -> list[dict]:
    """Dòng OCR + hình học và số đo từ nét Hi-SAM của chính dòng (textfix.lines): hướng, cao chữ, màu, nét."""
    import cv2
    from .glyph import measure
    from .ocr import read_lines
    H, W = img.shape[:2]
    # "dòng" không có chữ cái / chữ số (hàng sao "★★★★★", gạch, chấm) không phải chữ: để nguyên điểm ảnh nháp
    raw = [q for l in read_lines(img) if any(ch.isalnum() for ch in l["text"]) for q in _split_gaps(img, l)]
    raw_ang = [float(l.get("angle") or 0.0) for l in raw]   # góc OCR đo (ghi lại cho debug)
    G = orient(img, ink, raw)
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
                  "angle_ocr": round(raw_ang[i - 1], 1), "ink": _ink(G[i - 1]),
                  **({"vertical": True} if l.get("vertical") else {})})
    return L


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
         (người dùng chốt 14/10). VLM không nhắc tới thì GIỮ nguyên điểm ảnh nháp (fill.resolve)."""
    H, W = draft.shape[:2]
    ink = np.asarray(ink, bool)
    L = lines(draft, ink)
    A = _lab(draft)
    for l in L:
        # KHUNG DÒNG = KHUNG OCR, nguyên bản (12/10: bỏ hẳn bước nắn khung theo nét mực -- trên nền ảnh chụp nét mực đo thiếu,
        # khung bị cắt hẹp / đẩy mép trái vào mỗi dòng một kiểu: 11-25% dòng, danh sách thò thụt, chữ co).
        l["ocr_box"] = list(l["box"])
    I: list = []
    from . import geo
    glog, glog2, masks = [], [], {}
    if geo.ENABLED:
        L, I, glog, masks = _geo_lines(L, I, draft, A, ink, [])
        if geo.RUNS:
            L, glog2 = geo.runs(L)
    L, olog = _orphans(L, draft, ink, masks)
    L, glog = _renumber(L, [], [], glog + glog2 + olog)
    for l in L:
        l["shape"] = shape_of(l)
    _rows(L)
    _typo_notes(L, glog)
    return {"L": L, "size": [W, H], "by": {m["id"]: m for m in L}, "geo_log": glog}


def _rows(L: list) -> None:
    """HÀNG NHIỀU CỠ (lines.rows): các dòng ngang chung đường chân, sát nhau, lệch cỡ ("Giảm" | "30%" | "tất") -> l["row"] = "R#"."""
    from .lines import rows
    ink = [l.get("ink") or {} for l in L]
    R = rows([l["box"] for l in L], [k.get("h_cap") or 0.0 for k in ink],
             [k.get("base") if l["shape"] == "straight" and k.get("h_cap") else None for l, k in zip(L, ink)])
    for l, r in zip(L, R):
        l.pop("row", None)
        if r is not None:
            l["row"] = f"R{r + 1}"


def shape_of(l: dict) -> str:
    """Kiểu đường của một dòng: curve (có đường chân) / vertical / tilt / straight."""
    if l.get("path"):
        return "curve"
    if l.get("vertical"):
        return "vertical"
    return "tilt" if abs(l.get("angle") or 0) >= 2 else "straight"


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
        a, ao, ia = float(l.get("angle") or 0.0), l.get("angle_ocr"), (l.get("ink") or {}).get("ink_angle")
        src = f"nét đo {ia:+.1f}°" if ia is not None else "nét không đủ mảnh để đo hướng"
        if not l.get("vertical") and not l.get("path") and abs(a) >= 2:
            how = f"{src}, OCR đo {ao:+.1f}°: cùng thấy nghiêng" if ao is not None else "vùng chữ của bộ dò nghiêng (dòng OCR sót)"
            notes.append({"kind": f"nghiêng {a:+.0f}°", "how": how})
        elif ao is not None and abs(ao) >= 2 and not l.get("vertical") and not l.get("path"):
            notes.append({"kind": "dựng thẳng", "how": f"OCR đo {ao:+.1f}° nhưng {src}: hai nguồn không cùng thấy nghiêng"})
        if l.get("missed") and not l.get("orphan"):
            notes.append({"kind": "dòng OCR sót", "how": "; ".join(x for x in lg if "OCR sót" in x)[:300] or "vùng chữ của bộ dò không có dòng OCR, bộ đọc đọc ra chữ"})
        if l.get("orphan"):
            notes.append({"kind": "cụm nét OCR không đọc", "how": "; ".join(x for x in lg if "cụm nét" in x)[:300] or "nét Hi-SAM ngoài mọi dòng OCR"})
        if (l.get("ink") or {}).get("ornament"):
            notes.append({"kind": "trang trí đầu dòng", "how": "mảnh nét đầu dòng nhỏ, giữa chiều cao, cách chữ (lines.lead): không thuộc chữ, giữ nguyên điểm ảnh nháp"})
        if l.get("lead_icon"):
            notes.append({"kind": "khung dời khỏi icon", "how": "khung bộ dò ôm cả thứ đứng trước chữ -> mép trái dời tới nét chữ đầu tiên"})
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
    from . import ocr
    from .erase import line_region
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
                   "angle": float(ang), "caps": bool(caps), "stroke": None, "stroke_contrast": None, "effect": None,
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


def public(M: dict) -> dict:
    """Ô để ghi JSON."""
    return {"L": [{k: m[k] for k in ("id", "box", "ocr_box", "ocr", "size", "color", "angle", "curve", "runs", "geo", "path", "scale", "missed", "orphan", "printed", "lead_icon", "vertical", "shape", "row", "angle_ocr", "ink", "sag", "merged", "typo", "poly") if k in m}
                  for m in M["L"]]}


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
    items = [(m, (230, 20, 20), 2, "tl") for m in M["L"]]
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


def _system(base: str) -> str:
    """Lời dặn VLM + danh mục font (khoá, họ font, lớp, bề rộng chữ). Không còn danh mục icon: TextFix chỉ sửa chữ."""
    from .fonts import CATALOG, WIDTH
    fonts = "; ".join(f"{k}: {v[0]} ({v[1]}" + (f", {WIDTH[k][0]:.2f} / caps {WIDTH[k][1]:.2f}em" if k in WIDTH else "") + ")"
                      for k, v in CATALOG.items())
    return base.replace("{fonts}", fonts)

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


def _texts(texts: list[dict]) -> str:
    # câu "ai" = chữ phụ trợ bước brief tự viết (tiêu đề / khẩu hiệu / nút / nhãn khi khách chỉ điền sự thật): không phải lời khách
    return "\n".join(f"T{i} ({t.get('role', '')}{', OURS' if t.get('source') == 'ai' else ''}): {t['text']}"
                     for i, t in enumerate(texts))


PRODUCT_NOTE = ("The client uploaded a photo of their product and the image model copied it into the draft. Text, logos and "
                "graphics printed on that product (label, package, screen) are part of the photo and MUST stay exactly as they "
                "are: give every slot on the product kind \"keep\" -- never rewrite, correct, translate, redraw or skip it, "
                "and never write a client text there, even when the product's name is one of the client texts. Lines the list "
                "marks PRINTED ON THE CLIENT'S PRODUCT were recognised by code as product text: they are always \"keep\".")


def plan(draft: np.ndarray, M: dict, prompt: str, texts: list[dict], call, system: str, desc: str, base: np.ndarray,
         product: bool = False, meta: dict | None = None, fresh: bool = False) -> dict:
    """Một lời gọi VLM: lời dặn (system), PROMPT của FLUX, câu khách, bảng mô tả dòng (desc), ba ảnh DRAFT / SLOTS / BASE.
    meta: nhận số token / thời gian (llm.json_call). -> câu trả lời JSON (fill.parse đọc)."""
    mk = marked_image(draft, M)
    W, H = mk.size
    imgs = ["DRAFT:", _png(draft), "SLOTS (marked):", _png(mk),
            "BASE (the draft with the old letters wiped; your text is drawn on it):", _png(base)]
    user = (f"PROMPT given to the image model:\n{prompt}\n\nClient texts (exact words):\n{_texts(texts)}\n\nPoster size: {W}x{H}px.\n\n"
            + (PRODUCT_NOTE + "\n\n" if product else "") + desc)
    args = (system, [user, *imgs]) + ((meta,) if meta is not None else ())
    return _obj(call(*args, fresh=True) if fresh else call(*args))

CLIENT_LIKE = 0.6   # chữ OCR của ô giống câu khách >= 0.6 (bỏ dấu, chữ thường) = chữ poster chứa câu khách


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


def _fold(s: str) -> str:
    """Bỏ dấu tiếng Việt, chữ thường: "TRƯỞNG" == "trương" == "TRUONG"."""
    import unicodedata
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _flat(s: str) -> str:
    return re.sub(r"\W", "", _fold(s))


def _tid(c) -> int | None:
    c = str(c or "")
    return int(c[1:]) if c[:1] in "Tt" and c[1:].isdigit() else int(c) if c.isdigit() else None
