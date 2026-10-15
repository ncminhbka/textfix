"""Hình học các dòng nháp dùng khi ghép và bố cục: khung nét, khối xếp chồng, hàng, thứ tự đọc, khe trong dòng, căn lề.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

SPLIT_RATIO = 1.4   # các dòng của MỘT câu lệch cỡ quá 1,4 lần ("Giảm" nhỏ / "15%" to): mỗi dòng giữ cỡ của nó


STACK_GAP = 1.0     # các dòng của một khối: khe dọc <= 1 cỡ chữ, chồng ngang


ROW_GAP = 2.0       # mẩu cùng hàng của một câu (OCR cắt "NGÀY" / "• HỘI"): chồng dọc >= nửa cao, khe ngang <= 2 cỡ -> một phần tử


CENTER_TOL = 0.3    # tâm các dòng lệch <= 0,3 cỡ: căn giữa


EDGE_TOL = 0.6      # mép lệch <= 0,6 cỡ: thẳng mép đó


def _reading(boxes: list) -> list[int]:
    """Thứ tự đọc: hàng (tâm dọc lệch < nửa cao) từ trên xuống, trong hàng trái -> phải."""
    idx = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    rows = []
    for i in idx:
        b = boxes[i]
        cy, h = (b[1] + b[3]) / 2, b[3] - b[1]
        if rows and abs(cy - rows[-1]["cy"]) < 0.5 * max(h, rows[-1]["h"]):
            rows[-1]["ids"].append(i)
        else:
            rows.append({"cy": cy, "h": h, "ids": [i]})
    return [i for r in rows for i in sorted(r["ids"], key=lambda i: boxes[i][0])]


def ink_box(ink: np.ndarray, l: dict) -> list | None:
    """Hình bao NÉT Hi-SAM của dòng: mảng nét chạm vùng dòng (khung nới 0,15 cao) và >= nửa nằm trong khung; ít nét -> None."""
    import cv2
    H, W = ink.shape[:2]
    x0, y0, x1, y1 = l["box"]
    h = y1 - y0
    g = 0.15 * h
    X0, Y0, X1, Y1 = int(max(0, x0 - g)), int(max(0, y0 - g)), int(min(W, x1 + g)), int(min(H, y1 + g))
    sub = ink[Y0:Y1, X0:X1].astype(np.uint8)
    if X1 <= X0 or Y1 <= Y0 or sub.sum() < 0.03 * max(1.0, (x1 - x0) * h):
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    core = lab[max(0, int(y0) - Y0):int(y1) - Y0, max(0, int(x0) - X0):int(x1) - X0]
    inside = np.bincount(core.ravel(), minlength=n)
    keep = [k for k in range(1, n) if inside[k] >= 0.5 * st[k, cv2.CC_STAT_AREA]]
    if not keep:
        return None
    sel = st[keep]
    return [X0 + float(sel[:, 0].min()), Y0 + float(sel[:, 1].min()), X0 + float((sel[:, 0] + sel[:, 2]).max()),
            Y0 + float((sel[:, 1] + sel[:, 3]).max())]


def align_of(boxes: list, size: float) -> str | None:
    """Lề của các khung xếp chồng: tâm thẳng (<= CENTER_TOL cỡ) -> giữa; không thì mép lệch ít nhất (<= EDGE_TOL cỡ); một khung: None."""
    if len(boxes) < 2:
        return None
    sp = {"left": np.ptp([b[0] for b in boxes]), "right": np.ptp([b[2] for b in boxes]), "center": np.ptp([(b[0] + b[2]) / 2 for b in boxes])}
    # các dòng dài bằng nhau: thẳng tâm lẫn thẳng mép -- không nói được lề (14/10 mix1 t02: "Tối 2-4-6" / "Tối 3-5-7" kéo cả tập
    # tên khoá học căn trái thành căn giữa; m01 tiêu đề hai dòng gần bằng nhau căn trái thành giữa) -> None, để chỗ khác quyết
    if sp["center"] <= CENTER_TOL * size and min(sp["left"], sp["right"]) <= CENTER_TOL * size:
        return None
    if sp["center"] <= CENTER_TOL * size:
        return "center"
    best = min(sp, key=sp.get)
    return best if sp[best] <= EDGE_TOL * size else "center"


def lone_align(b: list, others: list, size: float) -> str:
    """Lề của MỘT dòng: thẳng mép trái / phải với một dòng ngay trên / dưới (khe <= 2 cỡ, chồng ngang) mà tâm lệch -> mép đó; không: giữa."""
    for o in others:
        if max(o[1] - b[3], b[1] - o[3]) > 2 * size or min(o[2], b[2]) - max(o[0], b[0]) <= 0:
            continue
        off = abs((o[0] + o[2]) / 2 - (b[0] + b[2]) / 2) > CENTER_TOL * size
        if off and abs(o[0] - b[0]) <= CENTER_TOL * size:
            return "left"
        if off and abs(o[2] - b[2]) <= CENTER_TOL * size:
            return "right"
    return "center"


def set_align(B: list, size: float) -> str:
    """Lề của TẬP: theo các cặp xếp chồng cùng cột; không có cặp nào (một hàng, lưới thưa, toả tròn) -> giữa."""
    pairs = [(a, b) for i, a in enumerate(B) for b in B[i + 1:] if min(a[2], b[2]) - max(a[0], b[0]) > 0.3 * min(a[2] - a[0], b[2] - b[0])]
    votes = Counter(x for x in (align_of([a, b], size) for a, b in pairs) if x)
    return votes.most_common(1)[0][0] if votes else "center"


def _gaps(ink: np.ndarray, b: list, parts: list[str], size: float) -> list[float] | None:
    """Ranh giới giữa các câu trên MỘT dòng nét: không ngưỡng -- kết hợp CHỮ (ranh giới dự kiến theo tỉ lệ số ký tự các câu) với NÉT
    Hi-SAM (các khe cột không nét): mỗi ranh giới chọn khe có điểm = bề rộng khe / (1 + khoảng cách tới chỗ dự kiến tính theo cỡ chữ)
    cao nhất -> tâm các khe, trái -> phải; dòng không có khe nào -> None."""
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    col = ink[max(0, y0):y1, max(0, x0):x1].any(axis=0)
    runs, i = [], 0
    while i < len(col):
        if not col[i]:
            j = i
            while j < len(col) and not col[j]:
                j += 1
            if i > 0 and j < len(col):
                runs.append((j - i, x0 + (i + j) / 2))
            i = j
        else:
            i += 1
    n = [len(p) + 1 for p in parts]
    want = [x0 + (x1 - x0) * sum(n[:k + 1]) / sum(n) for k in range(len(parts) - 1)]
    out = []
    for w in want:
        cand = [r for r in runs if r[1] not in out and (not out or r[1] > out[-1])]
        if not cand:
            return None
        out.append(max(cand, key=lambda r: r[0] / (1 + abs(r[1] - w) / max(1.0, size)))[1])
    return out


def _stacked(Ls: list) -> bool:
    """Các dòng (đã xếp theo thứ tự đọc) xếp chồng thành một khối: khe dọc <= STACK_GAP cỡ và chồng ngang, liên tiếp."""
    for a, b in zip(Ls, Ls[1:]):
        A, B = a["box"], b["box"]
        sz = max(a["size"], b["size"])
        if not (B[1] - A[3] <= STACK_GAP * sz and min(A[2], B[2]) - max(A[0], B[0]) > 0):
            return False
    return True


def _rows(Ls: list, boxes: list, sizes: list) -> list[list[int]]:
    """Gộp các dòng cùng hàng, sát nhau, CÙNG CỠ (mẩu OCR cắt của một dòng chữ) -> các nhóm chỉ số, theo thứ tự đọc. Dòng cùng hàng
    mà lệch cỡ >= SPLIT_RATIO là nháp NHẤN bằng cỡ ("Giảm" nhỏ | "30%" to | "tất" nhỏ, lines.rows): mỗi dòng một phần tử, cỡ của nó."""
    groups = []
    for i in range(len(Ls)):
        b = boxes[i]
        for g in groups:
            c = boxes[g[-1]]
            sz = max(sizes[i], sizes[g[-1]])
            if sz >= SPLIT_RATIO * min(sizes[i], sizes[g[-1]]):
                continue
            if min(b[3], c[3]) - max(b[1], c[1]) >= 0.5 * min(b[3] - b[1], c[3] - c[1]) and max(b[0] - c[2], c[0] - b[2]) <= ROW_GAP * sz:
                g.append(i)
                break
        else:
            groups.append([i])
    return groups
