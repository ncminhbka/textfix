"""Dòng OCR của nháp + số đo từ nét Hi-SAM của chính dòng (hướng, cao chữ, màu, nét); dòng viết dọc.
"""

from __future__ import annotations

import numpy as np

from .split import _split_gaps

CAP_EM = 0.72        # cao chữ hoa / em (cũng ~ cao chữ số, nét lên b d h k l t)


TALL_MIN = 0.15      # dòng có >= 15% ký tự cao -> ước cỡ theo cao nét cao (glyph h_tall)


X_EM = 0.53          # cao chữ thường (x-height) / em


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
    from ..erase.mask import line_region
    from . import measure_lines
    from .ink import Ink
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


def shape_of(l: dict) -> str:
    """Kiểu đường của một dòng: curve (có đường chân) / vertical / tilt / straight."""
    if l.get("path"):
        return "curve"
    if l.get("vertical"):
        return "vertical"
    return "tilt" if abs(l.get("angle") or 0) >= 2 else "straight"
