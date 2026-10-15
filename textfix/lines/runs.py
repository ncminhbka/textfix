"""HÀNG NHIỀU CỠ: nháp nhấn một phần câu bằng cỡ ("Giảm [30%] tất", "THANH MÁT mỗi ngày").

Bộ dò cắt chữ khác cỡ hẳn thành các dòng riêng; các dòng ngang CHUNG ĐƯỜNG CHÂN, đứng sát nhau và lệch cỡ là một hàng nhiều cỡ
(chặng bố cục rải câu theo cỡ từng dòng của hàng). Đo ĐOẠN nhiều cỡ bên trong một dòng đã thử (15/10, 60 nháp có đáp án): 32 ca,
không ca nào thật -- toàn dấu câu, mẩu nét vỡ, chữ giun cỡ lộn xộn -- nên không làm.
"""

from __future__ import annotations

RUN = 1.6        # lệch cỡ >= 1,6 lần (chữ thường không nét lên thấp hơn chữ hoa CÙNG cỡ 0,53 / 0,72 = 1,36 lần)
BASE = 0.25      # chung đường chân: lệch <= 0,25 cao chữ của dòng nhỏ hơn
ROW_GAP = 1.5    # sát nhau: khe ngang <= 1,5 cao chữ của dòng nhỏ hơn


def rows(boxes: list[list[float]], heights: list[float], bases: list[float | None]) -> list[int | None]:
    """Chỉ số HÀNG NHIỀU CỠ của mỗi dòng (None: không thuộc hàng nào). Hai dòng nối nhau khi chung đường chân (lệch <= BASE),
    sát nhau (khe <= ROW_GAP) và lệch cỡ >= RUN -- hai mục cùng cỡ trên một hàng (tên | giá) là hai dòng độc lập.
    boxes (x0, y0, x1, y1); heights: cao chữ đo; bases: đường chân đo (None: dòng không ngang / không đo được)."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if bases[i] is None or bases[j] is None:
                continue
            lo, hi = sorted((heights[i], heights[j]))
            a, b = sorted((i, j), key=lambda k: boxes[k][0])
            gap = boxes[b][0] - boxes[a][2]
            if hi >= RUN * lo and abs(bases[i] - bases[j]) <= BASE * lo and -0.1 * lo <= gap <= ROW_GAP * lo:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = [None] * n
    for k, g in enumerate(sorted((g for g in groups.values() if len(g) >= 2), key=min)):
        for i in g:
            out[i] = k
    return out
