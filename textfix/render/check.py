"""Bộ kiểm sau khi dựng: chồng nội dung, tràn, co quá nhỏ, chìm nền (tương phản WCAG); đổi màu khi chìm nền.
"""

from __future__ import annotations

import re

import numpy as np

from .geometry import _ink_poly, _multi_ov, _ov, _poly_ov

OVERLAP = 0.1   # chồng nhau > 10% diện tích phần tử nhỏ hơn = lỗi báo cho vòng sửa


CONTRAST_PX = 2.0   # điểm nền mà màu chữ trên đó có tỉ lệ tương phản WCAG < 2 = chữ chìm (chữ thân cần >= 4.5, chữ lớn >= 3)


CONTRAST_FRAC = 0.25   # > 25% nền sau khối chữ là điểm chìm = lỗi low_contrast (khối chữ không có bóng / viền)


DRAWN_RGB = 120.0   # màu chữ cách màu đo trên nét nháp <= 120 (khoảng cách RGB) = đúng màu nháp vẽ


def _lum(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def _low_contrast(r: dict, bg: np.ndarray, img: np.ndarray | None = None) -> float | None:
    """Phần nền (trong khung nét chữ của lệnh) mà màu chữ của lệnh chìm trên đó. Chữ có bóng / viền / tô gradient (màu chữ
    không đo được một màu, 08/10 h08_s1: tiêu đề vàng nhạt chìm hẳn vào nền hồng nhạt mà không bị bắt): đo trên ĐIỂM ẢNH THẬT
    -- điểm khác nền (bản dựng so với nền không chữ) là nét chữ; phần nét chữ có tương phản < 1.5 với nền ngay dưới nó."""
    if r.get("empty"):
        return None
    H, W = bg.shape[:2]
    x0, y0, x1, y1 = r["ink"]
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    m = re.match(r"rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?", r.get("fg") or "")
    if r.get("fx") or not m or (m.group(4) is not None and float(m.group(4)) < 0.5):
        if img is None:
            return None
        a, b = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32), bg[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        ink = np.abs(a - b).max(1) > 30
        if ink.sum() < 20:
            return None
        la, lb = _lum(a[ink]), _lum(b[ink])
        ratio = (np.maximum(la, lb) + 0.05) / (np.minimum(la, lb) + 0.05)
        r["px_based"] = True
        return round(float((ratio < 1.5).mean()), 2)
    lf = float(_lum(np.array([[float(m.group(i)) for i in (1, 2, 3)]])[0]))
    lb = _lum(bg[y0:y1, x0:x1].reshape(-1, 3))
    ratio = (np.maximum(lf, lb) + 0.05) / (np.minimum(lf, lb) + 0.05)
    return round(float((ratio < CONTRAST_PX).mean()), 2)


def _rgb(c: str | None) -> tuple | None:
    if not c:
        return None
    m = re.match(r"#([0-9a-f]{6})$", c.strip(), flags=re.I)
    if m:
        return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    m = re.match(r"rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)", c)
    return tuple(float(m.group(i)) for i in (1, 2, 3)) if m else None


def _recolor(r: dict, bg: np.ndarray, draft_color: str | None, pal: list | None = None, drawn: str | None = None) -> dict | None:
    """Chữ chìm nền: màu chữ thay thế, ưu tiên HỢP BẢNG MÀU (08/10 replay: tiêu đề vàng nhạt -> gần đen, đọc được nhưng thô) --
    màu các chữ đọc tốt khác trên poster, rồi màu nháp tối / sáng dần, cuối cùng trắng / gần đen: màu ĐẦU TIÊN có <= 10% nền
    chìm (tương phản WCAG < CONTRAST_PX). Không màu nào đạt: màu ít chìm nhất + quầng màu ngược độ sáng."""
    H, W = bg.shape[:2]
    x0, y0, x1, y1 = r["ink"]
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    lb = _lum(bg[y0:y1, x0:x1].reshape(-1, 3))
    # MÀU NHÁP VẼ (màu chữ gần màu đo trên nét nháp): nháp là chuẩn thẩm mỹ -> giữ màu, chỉ thêm quầng màu ngược độ sáng cho đọc
    # được, ở MỌI cỡ (14/10 mix1: chữ trắng trên pill pastel bị đổi sang xanh đen -- x04, v06, f02, f05; viền tiêu đề cũ phá chữ
    # neon). Màu khác hẳn màu nháp -> chọn màu như dưới.
    fg, dc = _rgb(r.get("fg")), _rgb(drawn)
    if fg and dc and float(np.linalg.norm(np.subtract(fg, dc))) <= DRAWN_RGB:
        lf = float(_lum(np.array([fg], np.float32))[0])
        return {"color": "#%02x%02x%02x" % tuple(int(round(v)) for v in fg), "frac": None,
                "halo": "rgba(0,0,0,.55)" if lf > 0.4 else "rgba(255,255,255,.6)"}
    cands = list(pal or [])   # màu chữ khác của chính poster trước: hợp bảng màu (màu nháp pha đen ra xám đục -- 08/10 h08_s1)
    d = _rgb(draft_color) or _rgb(r.get("fg"))
    if d:
        for t in (0.35, 0.6, 0.8):
            for to in (0, 255):
                cands.append("#%02x%02x%02x" % tuple(int(round(v + (to - v) * t)) for v in d))
    cands += ["#ffffff", "#161616"]
    best = None
    for c in dict.fromkeys(cands):
        rgb = _rgb(c)
        if rgb is None:
            continue
        lf = float(_lum(np.array([rgb], np.float32))[0])
        f = float(((np.maximum(lf, lb) + 0.05) / (np.minimum(lf, lb) + 0.05) < CONTRAST_PX).mean())
        if f <= 0.1:
            best = (c, f, lf)
            break
        if best is None or f < best[1] - 0.02:
            best = (c, f, lf)
    c, f, lf = best
    return {"color": c, "frac": round(f, 2), "halo": ("rgba(0,0,0,.55)" if lf > 0.4 else "rgba(255,255,255,.6)")
            if f > CONTRAST_FRAC else None}


SEVERE = ("empty_plan", "content_overlap", "overflow", "low_contrast")
INFO = ("touching",)   # chỉ báo: không tính khi so bản sửa với bản trước


def score(errs: list[dict]) -> tuple[int, int]:
    """(số lỗi nặng, số lỗi tính) -- vòng sửa giữ bản sửa khi cặp này giảm."""
    return sum(e["type"] in SEVERE for e in errs), sum(e["type"] not in INFO for e in errs)


TOUCH_GAP = 0.3   # hai phần tử cùng hàng cách nhau < 0,3 cỡ chữ (hẹp hơn một dấu cách) = chữ dính nhau ("333123", "Giảm20%")


def touch_gap(a: dict, b: dict) -> float | None:
    """Khe ngang giữa nét chữ hai phần tử CÙNG HÀNG (hộp nét chồng dọc >= nửa hộp thấp hơn), tính theo cỡ chữ nhỏ hơn; âm là
    chồng ngang. None: không cùng hàng, phần tử xoay / theo đường cong, hoặc rỗng."""
    if a.get("empty") or b.get("empty") or a.get("angle") or b.get("angle") or a.get("polys") or b.get("polys"):
        return None
    A, B = a["ink"], b["ink"]
    if min(A[3], B[3]) - max(A[1], B[1]) < 0.5 * min(A[3] - A[1], B[3] - B[1]):
        return None
    sz = min(float(a.get("size") or 0), float(b.get("size") or 0))
    return max(B[0] - A[2], A[0] - B[2]) / sz if sz > 0 else None


def check_units(U: list[dict], res: list | None, shrink_min: float | None = 0.7) -> list[dict]:
    """LỖI KHÁCH QUAN sau khi dựng (không cần mắt): nội dung hai đơn vị chồng nhau, tràn khung dù đã co tới sàn, phải co dưới
    shrink_min (chữ dài so với chỗ nháp vẽ; mục của tập kéo theo không tính), chữ chìm nền, hai phần tử cùng hàng dính nhau
    (touching: chỉ báo, không vào vòng sửa)."""
    errs = []
    res = res or []
    for i, a in enumerate(res):
        for b in res[i + 1:]:
            g = touch_gap(a, b)
            if g is not None and -0.5 < g < TOUCH_GAP:   # chồng sâu hơn là content_overlap
                errs.append({"type": "touching", "ops": [a["id"], b["id"]], "gap": round(g, 2)})
            if a.get("empty") or b.get("empty") or a["unit"] == b["unit"]:
                continue
            if a.get("polys") or b.get("polys"):
                o = _multi_ov(a.get("polys") or [_ink_poly(a)], b.get("polys") or [_ink_poly(b)])
            else:
                o = _poly_ov(_ink_poly(a), _ink_poly(b)) if a.get("angle") or b.get("angle") else _ov(a["ink"], b["ink"])
            if o > OVERLAP:
                errs.append({"type": "content_overlap", "ops": [a["id"], b["id"]], "frac": round(o, 2)})
        if a["over"] > 1:
            errs.append({"type": "overflow", "ops": [a["id"]], "px": round(a["over"])})
        elif shrink_min and a.get("shrink", 1) < shrink_min and not a.get("evened"):
            errs.append({"type": "too_small", "ops": [a["id"]], "shrink": round(a["shrink"], 2)})
        if (a.get("low") or 0) > (0.6 if a.get("px_based") else CONTRAST_FRAC):
            errs.append({"type": "low_contrast", "ops": [a["id"]], "frac": a["low"], "color": a.get("fg")})
    return errs
