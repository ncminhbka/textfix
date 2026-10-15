"""Hình học phần tử dựng: khung, góc xoay, đường cong (path), đa giác nét; chồng lấn giữa các phần tử.
"""

from __future__ import annotations

import numpy as np

ROT_MIN = 2.0   # độ: dòng model vẽ nghiêng >= 2 độ thì dựng nghiêng đúng góc đó (dưới 2 độ là nhiễu đo của OCR)


def _ids(op: dict) -> list[str]:
    ids = op.get("marks") or []
    if op.get("kind") == "text" and any(m.startswith("L") for m in ids):
        ids = [m for m in ids if m.startswith("L")]
    return ids


def op_geom(op: dict, M: dict) -> dict | None:
    """Khung DỰNG của lệnh = hợp khung các ô. Dòng NGHIÊNG (trung vị góc OCR các dòng L >= ROT_MIN): khung trong hệ trục xoay
    theo góc đó (ôm sát dòng nghiêng), div dựng thẳng rồi xoay. -> {rect (chưa xoay), angle, poly (4 góc thật), aabb}.
    Lệnh có fix.box (autofix cắt / thu khung): rect = fix.box (hệ trục của dòng), góc giữ như đo."""
    g = _geom(op, M)
    fb = (op.get("fix") or {}).get("box")
    if g is None or not fb:
        return g
    t = np.radians(g["angle"])
    c, sn = np.cos(t), np.sin(t)
    cx, cy, w, h = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2, fb[2] - fb[0], fb[3] - fb[1]
    corners = [[cx + u * c - v * sn, cy + u * sn + v * c] for u, v in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    W, H = M["size"]
    return {"rect": list(fb), "angle": g["angle"], "poly": corners,
            "aabb": [max(0.0, min(xs)), max(0.0, min(ys)), min(float(W), max(xs)), min(float(H), max(ys))]}


def _geom(op: dict, M: dict) -> dict | None:
    by, (W, H) = M["by"], M["size"]
    ids = [m for m in _ids(op) if m in by]
    if not ids:
        return None
    Ls = [by[m] for m in ids if m.startswith("L")]
    ang = 0.0
    if Ls:
        # GÓC CỦA KHỐI = góc của dòng DÀI nhất so với cao chữ (13/10 dev3 g13: "ĂN" hai chữ cái OCR đo -4 độ, "NGON" bên dưới
        # -19 độ, cùng một nhãn; trung vị ra -11, sai cả hai). Góc đo trên dòng càng dài càng chắc; dòng vài chữ cái là nhiễu.
        def reach(m):
            pl = m.get("poly")
            if pl and len(pl) == 4:
                e = sorted(float(np.hypot(pl[i][0] - pl[(i + 1) % 4][0], pl[i][1] - pl[(i + 1) % 4][1])) for i in range(4))
                return e[-1] / max(1.0, e[0])
            return (m["box"][2] - m["box"][0]) / max(1.0, m["box"][3] - m["box"][1])
        tilted = [m for m in Ls if abs(m.get("angle") or 0.0) >= ROT_MIN]
        if len(tilted) * 2 >= len(Ls):   # đa số dòng nghiêng thì khối nghiêng
            ang = float(max(tilted, key=reach).get("angle") or 0.0)
    if abs(ang) < ROT_MIN:
        bs = [by[m]["box"] for m in ids]
        r = [max(0.0, min(b[0] for b in bs)), max(0.0, min(b[1] for b in bs)), min(float(W), max(b[2] for b in bs)),
             min(float(H), max(b[3] for b in bs))]
        if r[2] - r[0] < 2 or r[3] - r[1] < 2:
            return None
        return {"rect": r, "angle": 0.0, "poly": [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], "aabb": r}
    t = np.radians(ang)
    c, sn = np.cos(t), np.sin(t)
    pts = []
    for m in ids:
        mk = by[m]
        pts += mk["poly"] if "poly" in mk else [[mk["box"][0], mk["box"][1]], [mk["box"][2], mk["box"][1]],
                                               [mk["box"][2], mk["box"][3]], [mk["box"][0], mk["box"][3]]]
    P_ = np.asarray(pts, float)
    u, v = P_[:, 0] * c + P_[:, 1] * sn, -P_[:, 0] * sn + P_[:, 1] * c   # sang hệ trục của dòng
    u0, u1, v0, v1 = u.min(), u.max(), v.min(), v.max()
    uc, vc, w, h = (u0 + u1) / 2, (v0 + v1) / 2, u1 - u0, v1 - v0
    cx, cy = uc * c - vc * sn, uc * sn + vc * c
    corners = [[uu * c - vv * sn, uu * sn + vv * c] for uu, vv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    return {"rect": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], "angle": ang, "poly": corners,
            "aabb": [max(0.0, min(xs)), max(0.0, min(ys)), min(float(W), max(xs)), min(float(H), max(ys))]}


def path_of(op: dict, M: dict) -> list | None:
    """Đường chân của lệnh chữ viết vào ô cung / lượn (textfix/lines/curve.py) -> các điểm, hướng đọc trái -> phải; lệnh thường: None.
    Lệnh chữ chỉ dựng theo các ô L của nó (_ids), nên ô S / I đi kèm không đổi gì. Lệnh có ô cong KHÔNG BAO GIỜ lùi về khối chữ
    thẳng trong khung bao (13/10 run ba93: chữ thẳng đè lên hình giữa con dấu): ô cong đi một mình;
    lệnh chưa qua validate mà gộp nhiều ô L thì viết theo đường cong dài nhất."""
    if op.get("kind") != "text":
        return None
    pts = [M["by"][m]["path"] for m in op.get("marks") or [] if m.startswith("L") and m in M["by"] and len(M["by"][m].get("path") or []) >= 2]
    if not pts:
        return None
    return max(pts, key=lambda pt: sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(pt[:-1], pt[1:])))


def _multi_ov(A: list, B: list) -> float:
    """Phần giao hai HỢP đa giác lồi (chữ theo đường = nhiều khúc) / diện tích hợp nhỏ hơn."""
    import cv2
    pa, pb = [np.asarray(x, np.float32) for x in A], [np.asarray(x, np.float32) for x in B]
    sm = min(sum(abs(cv2.contourArea(x)) for x in pa), sum(abs(cv2.contourArea(x)) for x in pb))
    if sm <= 0:
        return 0.0
    return float(sum(cv2.intersectConvexConvex(x, y)[0] for x in pa for y in pb)) / sm


def op_box(op: dict, M: dict) -> list[float] | None:
    """Khung thẳng ôm khung dựng của lệnh (để kiểm, cắt ảnh lỗi, vẽ debug)."""
    g = op_geom(op, M)
    return g["aabb"] if g else None


def _ov(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    s = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / s if w > 0 and h > 0 and s > 0 else 0.0


def _ink_poly(r: dict) -> list:
    """Đa giác nét mực của kết quả dựng: chữ xoay = nét chưa xoay xoay quanh tâm khung; chữ thẳng = khung nét."""
    i0 = r.get("ink0")
    if not i0 or not r.get("angle"):
        x0, y0, x1, y1 = r["ink"]
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    t = np.radians(r["angle"])
    c, sn = np.cos(t), np.sin(t)
    (x0, y0, x1, y1), (cx, cy) = i0["r"], i0["c"]
    return [[cx + (x - cx) * c - (y - cy) * sn, cy + (x - cx) * sn + (y - cy) * c] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


def _poly_ov(a, b) -> float:
    """Phần giao hai đa giác lồi (khung có thể nghiêng) / diện tích đa giác nhỏ hơn."""
    import cv2
    pa, pb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    s = min(cv2.contourArea(pa), cv2.contourArea(pb))
    if s <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    return float(inter) / s
