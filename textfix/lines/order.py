"""Đánh số dòng theo thứ tự đọc (cắt XY theo khối) -> L1, L2, ...
"""

from __future__ import annotations

import re


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
