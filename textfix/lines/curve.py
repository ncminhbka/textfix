"""CHỮ CONG: chữ chạy theo cung tròn / đường lượn (tiêu đề vòm, vòng con dấu, chữ "nụ cười"). OCR chỉ cho khung thẳng,
cắt cung thành mẩu xoay lệch; ở đây dò từ NÉT: chuỗi mảnh nét cùng cỡ nối nhau, khớp đường tròn / bậc 3 qua tâm các mảnh,
rồi gộp các mẩu OCR của một cung thành MỘT ô L có đường chân (path). Vùng chữ của bộ dò mà OCR không trả dòng nào -> ô mới.
"""

from __future__ import annotations

import math
import os

import numpy as np

from .detect import CAP_EM, X_EM

ENABLED = os.environ.get("TEXTFIX_GEO", "on") == "on"   # bật khi B3 (mô tả) + B4 (dựng theo đường chân) xong
                                                             # (12/10: 7 / 8 nháp g08-g10, g14), gộp lại chỉ thêm rủi ro

NPATH = 16           # số điểm đường chân ghi vào ô
CHAIN_GAP = 0.8      # hai mảnh nét cùng chuỗi: khe <= 0.8 cao mảnh lớn hơn ...
CHAIN_H = 1.8        # ... và cao lệch <= 1.8 lần
MARK_H = 0.4         # mảnh thấp hơn 0.4 cao trung vị chuỗi = dấu thanh / chấm (không dựng chuỗi, gộp vào khung sau)
ARC_MIN_PARTS = 5    # chuỗi >= 5 mảnh mới khớp cung
ARC_FIT = 0.15       # lệch khớp đường tròn (trung vị khoảng cách tâm mảnh tới đường tròn) <= 0.15 cao chữ
ARC_SAG = 0.3        # độ võng của cung (dây cung^2 / 8R) >= 0.3 cao chữ: thấp hơn là dòng thẳng / nghiêng (OCR lo)
ARC_SPAN = 3.2       # vùng chữ của bộ dò chỉ là cung khi dài >= 3.2 cao dòng. 18 vùng võng >= 0.3 của accept1 / dev2 / dev3 (13/10):
                     # 17 cung thật dài 3.7 - 12.3 cao dòng; 1 vùng dài 2.8 là hai dòng nghiêng xếp chồng ("GIẢM" / "30%", g06)
ARC_LINE = 0.18      # và đường thẳng khớp tệ: lệch trung vị >= 0.18 cao chữ
ARC_BLIND = 7        # cung >= 7 mảnh mà OCR không đọc được chữ nào vẫn là chữ (OCR hay trượt hẳn chữ cong)


BAND_MIN_H = 0.035   # DẢI CHỮ hiển thị: cao chữ >= 3.5% cạnh ngắn ảnh (chữ thân để OCR lo)
BAND_TILT = 4.0      # dải nghiêng: >= 4 độ


def _circle(pts: np.ndarray) -> tuple[float, float, float]:
    """Khớp đường tròn đại số (Kåsa)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(cx), float(cy), float(math.sqrt(max(c + cx ** 2 + cy ** 2, 1e-6)))


def _line_res(pts: np.ndarray) -> float:
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return float(np.median(np.abs(c @ vt[1])))


WAVE_DEV = 0.3       # đường cong tự do: lệch khỏi đường thẳng >= 0.3 cao chữ ...
WAVE_FIT = 0.15      # ... và khớp đa thức bậc 3 (theo trục chính) lệch trung vị <= 0.15 cao chữ
GROW = 1.35          # chữ TO DẦN / NHỎ DẦN: cao chữ hai đầu lệch >= 1.35 lần, đổi đều (tương quan với vị trí >= 0.8)


def _scale(u: np.ndarray, ext: np.ndarray, uq: np.ndarray) -> list[float] | None:
    """Hồ sơ cỡ dọc dòng (tỉ lệ so với cỡ trung vị) tại các điểm uq -- chỉ khi chữ to / nhỏ dần rõ rệt."""
    if len(u) < 6 or np.ptp(u) <= 0:
        return None
    k, b0 = np.polyfit(u, ext, 1)
    e0, e1 = k * u.min() + b0, k * u.max() + b0
    if min(e0, e1) <= 0 or max(e0, e1) / min(e0, e1) < GROW or abs(np.corrcoef(u, ext)[0, 1]) < 0.8:
        return None
    med = float(np.median(ext))
    return [round(float((k * x + b0) / med), 3) for x in uq]


def _baseline(center: np.ndarray, half: np.ndarray) -> list[list[float]]:
    """Điểm giữa cao chữ -> điểm đường chân: dịch nửa cao chữ theo pháp tuyến "xuống" của hướng đọc (trái -> phải)."""
    t = np.gradient(center, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)
    n = np.c_[-t[:, 1], t[:, 0]]
    return [[round(float(x), 1), round(float(y), 1)] for x, y in center + n * half[:, None]]


def _fit_circle(pts, ext, hm):
    cx, cy, r = _circle(pts)
    res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
    keep = res <= max(2 * np.median(res), 0.1 * hm)   # khớp lại không có điểm lệch (dấu dính, nét chữ Q / g thò xuống)
    if keep.sum() >= ARC_MIN_PARTS:
        pts, ext = pts[keep], ext[keep]
        cx, cy, r = _circle(pts)
        res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
    chord = float(np.hypot(*(pts.max(0) - pts.min(0))))
    sag = chord ** 2 / (8 * r) if r > 0 else 0.0
    if np.median(res) > ARC_FIT * hm or sag < ARC_SAG * hm or _line_res(pts) < ARC_LINE * hm or r > 8 * chord:
        return None
    up = cy > pts[:, 1].mean()   # tâm dưới chuỗi: cung lồi phía trên
    ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    # góc liền mạch quanh đỉnh (cung trên: 180..360) / quanh đáy (cung dưới: 0..180)
    ang = np.where(ang < 0, ang + 360, ang) if up else np.where(ang < -90, ang + 360, ang)
    da = math.degrees(0.5 * hm / max(r, 1))   # nới hai đầu nửa cao chữ (tâm lát đầu / cuối cách mép chữ)
    if up:   # đi trái -> phải theo chiều kim đồng hồ quanh đỉnh
        a0, a1 = float(ang.min()) - da, float(ang.max()) + da
    else:    # cung dưới: trái (góc lớn) -> phải (góc nhỏ)
        a0, a1 = float(ang.max()) + da, float(ang.min()) - da
    # đường chân: chữ cung trên đứng ngoài đường tròn (chân phía tâm), cung dưới ngược lại
    rb = r - hm / 2 if up else r + hm / 2
    aq = np.linspace(a0, a1, NPATH)
    path = [[round(float(cx + rb * math.cos(math.radians(a))), 1), round(float(cy + rb * math.sin(math.radians(a))), 1)] for a in aq]
    u = ang if up else -ang
    return {"geo": "arc", "path": path, "scale": _scale(u, ext, aq if up else -aq),
            "curve": {"cx": round(cx, 1), "cy": round(cy, 1), "r": round(rb, 1), "a0": round(a0, 1), "a1": round(a1, 1),
                      "up": bool(up)}}


def _axis(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    e1 = vt[0] if vt[0][0] >= 0 else -vt[0]   # hướng đọc trái -> phải
    e2 = np.array([-e1[1], e1[0]])            # pháp tuyến "xuống" của hướng đọc
    return c, e1, e2


def _fit_wave(pts, ext, hm):
    c, e1, e2 = _axis(pts)
    u, v = (pts - c) @ e1, (pts - c) @ e2
    if np.ptp(u) < 3 * hm or len(u) < 8:
        return None
    p3 = np.polyfit(u, v, 3)
    res = np.median(np.abs(np.polyval(p3, u) - v))
    lin = np.polyval(np.polyfit(u, v, 1), u)
    dev = float(np.max(np.abs(np.polyval(p3, u) - lin)))
    if res > WAVE_FIT * hm or dev < WAVE_DEV * hm:
        return None
    uq = np.linspace(u.min() - 0.5 * hm, u.max() + 0.5 * hm, NPATH)
    center = c + np.outer(uq, e1) + np.outer(np.polyval(p3, uq), e2)
    sc = _scale(u, ext, uq)
    half = 0.5 * hm * (np.array(sc) if sc else np.ones(NPATH))
    return {"geo": "wave", "path": _baseline(center, half), "scale": sc}


def _fit_grow(pts, ext, hm):
    """Dòng thẳng (thẳng / nghiêng) mà chữ to dần / nhỏ dần: đường chân thẳng qua chân các lát + hồ sơ cỡ."""
    c, e1, e2 = _axis(pts)
    u = (pts - c) @ e1
    uq = np.linspace(u.min() - 0.5 * hm, u.max() + 0.5 * hm, NPATH)
    sc = _scale(u, ext, uq)
    if not sc:
        return None
    bot = pts + np.outer(ext / 2, e2)   # chân các lát
    ub, vb = (bot - c) @ e1, (bot - c) @ e2
    k, b0 = np.polyfit(ub, vb, 1)
    center = c + np.outer(uq, e1) + np.outer(k * uq + b0, e2)
    return {"geo": "grow", "path": [[round(float(x), 1), round(float(y), 1)] for x, y in center], "scale": sc}


def _chains(comps: list[dict]) -> list[list[int]]:
    """Nối mảnh nét thành chuỗi (khe nhỏ, cao gần nhau)."""
    n = len(comps)
    par = list(range(n))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i in range(n):
        a = comps[i]
        for j in range(i + 1, n):
            b = comps[j]
            hi, lo = max(a["h"], b["h"]), min(a["h"], b["h"])
            if hi > CHAIN_H * lo:
                continue
            gx = max(b["x0"] - a["x1"], a["x0"] - b["x1"], 0)
            gy = max(b["y0"] - a["y1"], a["y0"] - b["y1"], 0)
            # nối CẠNH nhau (cùng dòng): hai mảnh chồng nhau theo chiều dọc >= 0.3 cao mảnh thấp (cung nghiêng tới ~45 độ vẫn
            # chồng); dòng trên / dưới sát nhau (khe dòng ~0.3 cao chữ) không chồng dọc -> không nối
            ov = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
            if ov >= 0.3 * lo and math.hypot(gx, gy) <= CHAIN_GAP * hi:
                par[f(i)] = f(j)
    g = {}
    for i in range(n):
        g.setdefault(f(i), []).append(i)
    return list(g.values())


def arcs(ink: np.ndarray, W: int, H: int, dbg: list | None = None) -> list[dict]:
    """Các cung chữ trong mặt nạ nét. -> [{box, curve, size, parts}]"""
    import cv2
    m = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, cc, st, cen = cv2.connectedComponentsWithStats(m, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 12 or h < 8 or w > 0.5 * W or h > 0.4 * H:
            continue
        comps.append({"i": i, "x0": x, "y0": y, "x1": x + w, "y1": y + h, "h": h, "c": cen[i], "a": a})
    if not comps:
        return []
    hs = np.array([c["h"] for c in comps])
    out = []
    for ch in _chains(comps):
        if len(ch) < 2:
            continue
        hm = float(np.median(hs[ch]))
        body = [k for k in ch if comps[k]["h"] >= MARK_H * hm]
        if len(body) < 2:
            continue
        # ĐIỂM KHỚP: tâm nét theo LÁT dọc rộng ~0.4 cao chữ trên từng mảnh (chữ đậm / có viền dính liền thành vài mảnh to)
        pts, ext = [], []
        for k in body:
            c = comps[k]
            sub = cc[c["y0"]:c["y1"], c["x0"]:c["x1"]] == c["i"]
            step = max(3, int(0.4 * hm))
            for x in range(0, sub.shape[1], step):
                sl = sub[:, x:x + step]
                ys = np.nonzero(sl.any(1))[0]
                if len(ys) and sl.sum() >= 0.05 * step * hm:
                    pts.append((c["x0"] + x + sl.shape[1] / 2, c["y0"] + (ys[0] + ys[-1]) / 2))
                    ext.append(ys[-1] - ys[0] + 1)
        pts = np.array(pts, np.float64)
        if len(ext):   # cao chữ theo lát (mảnh dính cả từ trên cung nghiêng cao phồng)
            hm = float(np.median(ext))
        if len(pts) < ARC_MIN_PARTS + 1 or np.ptp(pts[:, 0]) < 3 * hm:
            if dbg is not None and len(pts):
                dbg.append({"box": [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())],
                            "why": "ít điểm / chuỗi ngắn"})
            continue
        ext = np.array(ext, np.float64)
        g = _fit_circle(pts, ext, hm) or _fit_wave(pts, ext, hm) or _fit_grow(pts, ext, hm)
        if g is None:
            if dbg is not None:
                dbg.append({"box": [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())],
                            "why": "không cong / không đổi cỡ", "line": round(_line_res(pts) / hm, 2), "n": len(pts)})
            continue
        sel = np.isin(cc, [comps[k]["i"] for k in ch])
        ys, xs = np.nonzero(sel)
        out.append({"box": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], **g,
                    "h": hm, "mask": sel, "parts": len(body)})
    return out


def apply_arcs(L: list[dict], I: list[dict], found: list[dict], cap_em: float) -> tuple[list[dict], list[dict], list[str]]:
    """Thay các ô L / I nằm trên một cung bằng MỘT ô L cung. Chữ OCR gợi ý = các mẩu OCR theo thứ tự trái -> phải."""
    log = []
    for a in found:
        mk = a["mask"]
        H, W = mk.shape

        def on(b):
            x0, y0, x1, y1 = (int(max(0, v)) for v in b)
            sub = mk[y0:min(H, y1), x0:min(W, x1)]
            return sub.size > 0 and sub.mean() >= 0.08
        # ô đã là ô cung của vùng khác không gộp lần nữa (13/10 accept1 g04: hai vòm chữ lồng nhau, vòm sau nuốt vòm trước)
        cov_L = [l for l in L if not l.get("path") and on(l["box"])]
        cov_I = [c for c in I if on(c["box"])]
        # MẨU Ở HAI ĐẦU CUNG (12/10 g01: chữ cái đầu / cuối vòm không nối được vào chuỗi nét, còn lại thành ô rời; g02: mẩu OCR
        # "NHA GAO" nghiêng 42 độ nằm ngay trên vòng con dấu; 13/10 run ba93: bộ dò thấy vòm "TIỆM BÁNH NHÀ" và mẩu "GẠO" là hai
        # vùng rời): ô có tâm nằm TRÊN đường tròn của cung (lệch <= 0.6 cao chữ), cùng cỡ chữ, và BẮT ĐẦU trong vòng 2 cao chữ kể
        # từ đầu cung (tính theo mép gần của ô, không theo tâm: một từ dài có tâm cách đầu cung cả nửa chiều dài của nó) -> cũng
        # thuộc cung; cung dài ra tới hết các ô đó
        c = a.get("curve")
        if c:
            h = a["h"]
            rc = c["r"] + (h / 2 if c["up"] else -h / 2)   # bán kính đường TÂM chữ
            pad = math.degrees(2 * h / max(rc, 1.0))
            lo, hi = min(c["a0"], c["a1"]), max(c["a0"], c["a1"])
            mid = (lo + hi) / 2

            def ang(x, y, mid=mid, c=c):   # góc liền mạch quanh giữa cung
                t = math.degrees(math.atan2(y - c["cy"], x - c["cx"]))
                return t + 360 * round((mid - t) / 360)

            def spread(m, c=c, rc=rc, h=h, ang=ang):   # các góc của ô trên đường tròn: tâm + những góc khung nằm sát đường tròn
                pts = m.get("poly") or [[x, y] for x in (m["box"][0], m["box"][2]) for y in (m["box"][1], m["box"][3])]
                x, y = (m["box"][0] + m["box"][2]) / 2, (m["box"][1] + m["box"][3]) / 2
                return [ang(x, y)] + [ang(px, py) for px, py in pts if abs(math.hypot(px - c["cx"], py - c["cy"]) - rc) <= 1.2 * h]

            def near(m, c=c, rc=rc, h=h, lo=lo, hi=hi, pad=pad, spread=spread):
                if m.get("path"):
                    return False
                x, y = (m["box"][0] + m["box"][2]) / 2, (m["box"][1] + m["box"][3]) / 2
                return abs(math.hypot(x - c["cx"], y - c["cy"]) - rc) <= 0.6 * h and any(lo - pad <= t <= hi + pad for t in spread(m)) \
                    and (m["size"] <= 2 * h / 0.72 if m.get("size") else min(m["box"][2] - m["box"][0], m["box"][3] - m["box"][1]) <= 3 * h)
            more_L = [l for l in L if l not in cov_L and near(l)]
            more_I = [m for m in I if m not in cov_I and near(m)]
            if more_L or more_I:
                aa = [t for m in more_L + more_I for t in spread(m)[1:]]
                lo, hi = min([lo] + aa), max([hi] + aa)
                a0, a1 = (lo, hi) if c["a0"] <= c["a1"] else (hi, lo)
                c = {**c, "a0": round(a0, 1), "a1": round(a1, 1)}
                a = {**a, "curve": c, "path": [[round(c["cx"] + c["r"] * math.cos(math.radians(t)), 1),
                                                round(c["cy"] + c["r"] * math.sin(math.radians(t)), 1)] for t in np.linspace(a0, a1, NPATH)]}
                xs = [q[0] for q in a["path"]] + [a["box"][0], a["box"][2]] + [v for m in more_L + more_I for v in (m["box"][0], m["box"][2])]
                ys = [q[1] for q in a["path"]] + [a["box"][1], a["box"][3]] + [v for m in more_L + more_I for v in (m["box"][1], m["box"][3])]
                a["box"] = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                cov_L, cov_I = cov_L + more_L, cov_I + more_I
        if not cov_L and a["parts"] < ARC_BLIND:
            continue   # cung ngắn không có chữ OCR nào: có thể là hoạ tiết tròn (hạt, chấm) -- không phải chữ
        cov_L.sort(key=lambda l: (l["box"][0] + l["box"][2]) / 2)
        if cov_L:
            first = cov_L[0]
        else:   # OCR không đọc được chữ cong (cung dài, nhiều mảnh cùng cỡ đều nhau = chữ): ô mới, chữ OCR rỗng
            first = {"id": f"L{1 + max([int(l['id'][1:]) for l in L] or [0])}", "ocr": "", "caps": True, "color": None,
                     "stroke": None, "base": None}
            L = L + [first]
        caps = all(l.get("caps") for l in cov_L) if cov_L else True
        new = {**first, "box": a["box"], "ocr_box": a["box"], "ocr": " ".join(l["ocr"] for l in cov_L),
               "size": float(a["h"] / (cap_em if caps else 0.6)), "angle": 0.0, "geo": a["geo"], "path": a["path"],
               "scale": a.get("scale"), **({"curve": a["curve"]} if a.get("curve") else {}),
               "sag": a.get("sag"), "merged": [l.get("ocr") or "" for l in cov_L],   # bằng chứng, ghi cho debug
               "poly": [[a["box"][0], a["box"][1]], [a["box"][2], a["box"][1]], [a["box"][2], a["box"][3]], [a["box"][0], a["box"][3]]]}
        zone = mk.copy()   # vùng chữ của ô: vùng dò được + khung các mẩu đã gộp (_geo_lines thêm dải dọc đường chân)
        for m in cov_L + cov_I:
            if m.get("poly"):
                import cv2
                one = np.zeros(zone.shape, np.uint8)
                cv2.fillPoly(one, [np.asarray(m["poly"], np.int32)], 1)
                zone |= one.astype(bool)
            else:
                x0, y0, x1, y1 = (int(max(0, v)) for v in m["box"])
                zone[y0:y1, x0:x1] = True
        new["_mask"] = zone
        gone = {id(l) for l in cov_L}
        L = [new if l is first else l for l in L if id(l) not in gone or l is first]
        I = [c for c in I if c not in cov_I]
        log.append(f"chữ {a['geo']}: {[l['id'] for l in cov_L]} + {[c['id'] for c in cov_I]} -> {first['id']}"
                   + (f" {a['curve']}" if a.get("curve") else "") + (f" cỡ {a['scale'][0]}..{a['scale'][-1]}" if a.get("scale") else ""))
    return L, I, log


def circle_of(cen: np.ndarray, h: float) -> dict:
    """Đường TÂM một dòng cong (trái -> phải) có khớp đường tròn không -> {"curve": ...} cùng khuôn với `_fit_circle` (r = bán kính
    đường CHÂN), hoặc {} (cong tự do: không có "phần kéo dài" để nối mẩu rời vào)."""
    cx, cy, r = _circle(cen)
    chord = float(np.hypot(*(cen[-1] - cen[0])))
    if float(np.median(np.abs(np.hypot(cen[:, 0] - cx, cen[:, 1] - cy) - r))) > ARC_FIT * h or r > 8 * chord:
        return {}
    up = bool(cy > cen[:, 1].mean())   # tâm dưới chữ: cung lồi phía trên
    ang = np.degrees(np.unwrap(np.arctan2(cen[:, 1] - cy, cen[:, 0] - cx)))
    return {"curve": {"cx": round(cx, 1), "cy": round(cy, 1), "r": round(r - h / 2 if up else r + h / 2, 1),
                      "a0": round(float(ang[0]), 1), "a1": round(float(ang[-1]), 1), "up": up}}


def region_line(mask: np.ndarray, pad: float) -> dict | None:
    """Một VÙNG CHỮ của bộ dò (ocr.regions: vùng LÕI, đã co) -> hình học dòng chữ: đường TÂM (NPATH điểm, hướng đọc trái -> phải, đã
    nới `pad` ở hai đầu), cao khung dòng h (bề dày lõi + 2 pad), võng / lệch khớp đường tròn (theo h), góc nghiêng, chiều dài.
    Đường tâm = đa thức bậc <= 3 qua tâm các lát cắt NGANG TRỤC CHÍNH của vùng (vùng xiên / cong đều đo được như nhau)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    pts = np.c_[xs, ys].astype(np.float64)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts[:: max(1, len(pts) // 4000)] - c, full_matrices=False)
    e1 = vt[0] if vt[0][0] >= 0 else -vt[0]
    e2 = np.array([-e1[1], e1[0]])
    u, v = (pts - c) @ e1, (pts - c) @ e2
    step = 3.0
    k = np.floor((u - u.min()) / step).astype(int)
    us, mid, thick = [], [], []
    for b in range(k.max() + 1):
        vv = v[k == b]
        if len(vv):
            us.append(u.min() + (b + 0.5) * step)
            mid.append((vv.min() + vv.max()) / 2)
            thick.append(vv.max() - vv.min() + 1)
    us, mid, thick = np.array(us), np.array(mid), np.array(thick)
    if len(us) < 6:
        return None
    t = float(np.median(thick))
    h = t + 2 * pad
    deg = 3 if len(us) >= 16 else 2 if len(us) >= 8 else 1
    co = np.polyfit(us, mid, deg)
    uq = np.linspace(us.min(), us.max(), NPATH)
    cen = c + np.outer(uq, e1) + np.outer(np.polyval(co, uq), e2)
    for end, nb in ((0, 1), (-1, -2)):   # nới hai đầu theo tiếp tuyến (lõi bị co cả theo chiều dài)
        d = cen[end] - cen[nb]
        cen[end] = cen[end] + d / max(1e-6, float(np.linalg.norm(d))) * pad
    chord = cen[-1] - cen[0]
    ln = float(np.linalg.norm(chord)) or 1.0
    nrm = np.array([-chord[1], chord[0]]) / ln
    sag = float(np.max(np.abs((cen - cen[0]) @ nrm)))
    cx, cy, r = _circle(cen)
    res = float(np.median(np.abs(np.hypot(cen[:, 0] - cx, cen[:, 1] - cy) - r)))
    return {"center": cen, "h": h, "t": t, "sag": sag / h, "res": res / h, "tilt": math.degrees(math.atan2(chord[1], chord[0])),
            "span": float(np.sum(np.hypot(*np.diff(cen, axis=0).T))), "circle": (cx, cy, r)}


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

    from . import ocr
    H, W = A.shape[:2]
    ink = np.asarray(ink0, bool)   # nét mực = nét chữ của Hi-SAM (14/10; trước: chỗ bước xoá FLUX đã gỡ)
    found, log, masks = [], [], {}
    cen_in = lambda b, m: bool(m[int(min(H - 1, max(0, (b[1] + b[3]) / 2))), int(min(W - 1, max(0, (b[0] + b[2]) / 2)))])   # noqa: E731
    regs = ocr.regions(draft)
    log.append(f"bộ dò: {len(regs)} vùng chữ" if regs else "bộ dò: KHÔNG lấy được bản đồ vùng chữ (phiên bản RapidOCR?) -- phần chữ khó không chạy")
    for R in regs:
        g = region_line(R["mask"], R["pad"])
        if g is None or g["h"] < BAND_MIN_H * min(W, H):   # chỉ chữ hiển thị (cao dòng >= 3.5% cạnh ngắn)
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
        curved = g["sag"] >= ARC_SAG and g["span"] >= ARC_SPAN * g["h"]
        stack = _stacked(near, ink, g["tilt"], ext) if curved else [0]
        if curved and len(stack) == 1:
            found.append({"box": box, "geo": "arc", "path": _baseline(cen, np.full(len(cen), ext / 2)), "scale": None, "h": ext,
                          "mask": near, "parts": ARC_BLIND, "sag": round(g["sag"], 2), **circle_of(cen, ext)})
            continue
        if not missed or on or g["span"] < 1.5 * g["h"]:
            continue
        # (b) dòng bị sót: phải là LỚP PHỦ bị gỡ (nét mực chiếm >= 10% vùng; chữ in trên vật / hoạ tiết ảnh còn nguyên trên nền)
        if float((ink & near).sum()) < 0.1 * float(near.sum()):
            continue
        ang = g["tilt"] if abs(g["tilt"]) >= BAND_TILT else 0.0
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
    L, I, lg = apply_arcs(L, I, found, CAP_EM)
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
