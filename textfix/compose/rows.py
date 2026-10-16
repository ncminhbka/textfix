"""HÀNG CỦA DANH SÁCH / BẢNG trong nháp, nhận bằng hình học thuần: cột dòng thẳng lề, cùng cỡ, đều nhịp; các cột có dòng cùng
hàng ghép thành bảng. Dùng khi ghép câu (compose.assign): một câu VLM ghi cho hai dòng ở HAI HÀNG không gộp thành một khối, và
hàng có câu liền kề (tên món của giá) nhận câu đó.

  G = lists(M)      # [{"rows": [[id dòng], ...], "cols": n, "lines": [...]}]
  row_of(G)         # {id dòng: (danh sách, hàng)}

16/10: đã thử gỡ hàng thừa + dàn đều các hàng còn lại bằng điểm ảnh; người dùng bỏ (dàn đều chỉ được 1 / 5 danh sách, còn lại
để chấm / icon trơ trọi): FLUX vẽ thừa hàng thì VLM viết chữ lấp theo.
"""

from __future__ import annotations

import numpy as np

from .geometry import CENTER_TOL, EDGE_TOL

SIZE_RATIO = 1.25   # các hàng một cột: cỡ nháp hai hàng liền nhau lệch <= 1,25 lần (như tập cùng vai, compose.layout.SET_RATIO)
ROW_GAP = 2.0       # ... khe dọc giữa hai hàng liền nhau <= 2 cỡ chữ
PITCH_BREAK = 1.6   # ... nhịp (khoảng cách tâm) > 1,6 lần nhịp trung vị của cột: hết danh sách, bắt đầu khối khác
LIST_MIN = 3        # một cột >= 3 hàng mới là danh sách (hai dòng xếp chồng thường là tiêu đề + phụ đề); bảng >= 2 hàng
ROW_OVERLAP = 0.5   # hai dòng cùng hàng: chồng dọc >= nửa cao dòng thấp hơn


def _straight(l: dict) -> bool:
    return not (l.get("path") or abs(l.get("angle") or 0) >= 2 or l.get("vertical") or l.get("orphan"))


def _aligned(A: list, B: list, s: float) -> bool:
    """Hai khung thẳng một lề: trái / phải lệch <= EDGE_TOL cỡ, hoặc tâm lệch <= CENTER_TOL cỡ."""
    return (abs(A[0] - B[0]) <= EDGE_TOL * s or abs(A[2] - B[2]) <= EDGE_TOL * s
            or abs((A[0] + A[2]) / 2 - (B[0] + B[2]) / 2) <= CENTER_TOL * s)


def _same_row(A: list, B: list) -> bool:
    return min(A[3], B[3]) - max(A[1], B[1]) >= ROW_OVERLAP * min(A[3] - A[1], B[3] - B[1])


def columns(L: list[dict]) -> list[list[dict]]:
    """Các CỘT: chuỗi dòng thẳng, thẳng lề, cỡ gần nhau, mỗi dòng nối với dòng gần nhất ngay dưới (khe <= ROW_GAP cỡ); chuỗi
    cắt ở nhịp lạc (> PITCH_BREAK nhịp trung vị). -> các cột >= 2 dòng, dòng trên xuống."""
    Ls = sorted((l for l in L if _straight(l)), key=lambda l: (l["box"][1] + l["box"][3]) / 2)
    nxt, prev = {}, {}
    for i, a in enumerate(Ls):
        A, sa = a["box"], a["size"]
        best = None
        for b in Ls[i + 1:]:
            B, s = b["box"], max(sa, b["size"])
            if B[1] < A[1] + 0.5 * (A[3] - A[1]) or max(sa, b["size"]) > SIZE_RATIO * min(sa, b["size"]):
                continue
            if B[1] - A[3] > ROW_GAP * s or not _aligned(A, B, s):
                continue
            if best is None or B[1] < best["box"][1]:
                best = b
        if best is not None:
            j = best["id"]
            # một dòng chỉ nhận một dòng ngay trên: dòng trên gần hơn thắng
            if j not in prev or A[3] > prev[j]["box"][3]:
                if j in prev:
                    nxt.pop(prev[j]["id"], None)
                prev[j], nxt[a["id"]] = a, best
    out = []
    starts = [l for l in Ls if l["id"] not in {b["id"] for b in nxt.values()}]
    for s0 in starts:
        ch = [s0]
        while ch[-1]["id"] in nxt:
            ch.append(nxt[ch[-1]["id"]])
        if len(ch) < 2:
            continue
        cy = [(l["box"][1] + l["box"][3]) / 2 for l in ch]
        pitch = np.diff(cy)
        med = float(np.median(pitch))
        cur = [ch[0]]
        for l, p in zip(ch[1:], pitch):
            if p > PITCH_BREAK * med:
                if len(cur) >= 2:
                    out.append(cur)
                cur = []
            cur.append(l)
        if len(cur) >= 2:
            out.append(cur)
    return out


def lists(M: dict) -> list[dict]:
    """DANH SÁCH / BẢNG: cột >= LIST_MIN hàng, hoặc >= 2 cột không chồng ngang có >= 2 cặp dòng cùng hàng (tên món | giá) ghép
    thành bảng >= 2 hàng. Hàng = các dòng cùng hàng (chồng dọc >= ROW_OVERLAP), trên xuống. -> [{"rows", "cols", "lines"}]."""
    C = columns(M["L"])
    parent = list(range(len(C)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i, a in enumerate(C):
        for j in range(i + 1, len(C)):
            b = C[j]
            ax = (min(l["box"][0] for l in a), max(l["box"][2] for l in a))
            bx = (min(l["box"][0] for l in b), max(l["box"][2] for l in b))
            if min(ax[1], bx[1]) > max(ax[0], bx[0]):   # chồng ngang: hai cột cùng chỗ, không phải hai cột của một bảng
                continue
            if sum(any(_same_row(x["box"], y["box"]) for y in b) for x in a) >= 2:
                parent[root(j)] = root(i)
    G = {}
    for i, c in enumerate(C):
        G.setdefault(root(i), []).append(c)
    out = []
    for cols in G.values():
        Ls = sorted({l["id"]: l for c in cols for l in c}.values(), key=lambda l: (l["box"][1] + l["box"][3]) / 2)
        rows = []
        for l in Ls:
            if rows and any(_same_row(l["box"], o["box"]) for o in rows[-1]):
                rows[-1].append(l)
            else:
                rows.append([l])
        if len(rows) >= LIST_MIN or (len(cols) >= 2 and len(rows) >= 2):
            out.append({"rows": [[l["id"] for l in sorted(r, key=lambda l: l["box"][0])] for r in rows], "cols": len(cols),
                        "lines": [l["id"] for l in Ls]})
    return out


def row_of(G: list[dict]) -> dict:
    """{id dòng: (chỉ số danh sách, chỉ số hàng)}."""
    return {m: (gi, ri) for gi, g in enumerate(G) for ri, r in enumerate(g["rows"]) for m in r}
