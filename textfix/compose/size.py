"""CỠ CHỮ theo font thật (cao chữ hoa / chữ thường của font đã chọn) và CHỖ (khung nét nới vào nền trống cùng màu).
"""

from __future__ import annotations

import numpy as np

TALL_CHARS = set("bdfhklt" + "gjpqy")   # chữ thường có nét lên / xuống: cao xấp xỉ chữ hoa


def letter_em(font: str | None, text: str, upper: bool = False) -> float:
    """Cao (em) của những chữ CAO NHẤT trong câu khi dựng bằng font: chữ hoa / chữ số / nét lên-xuống -> cao chữ hoa của font, câu
    toàn chữ thường thấp ("mỗi ma ơi") -> cao chữ thường. Cao nét đo trên nháp (lines: h_cap = p90 cao mảnh chính) cũng là cao
    những chữ cao nhất, nên cỡ = h_cap / letter_em."""
    import unicodedata

    from ..render.fonts import heights
    cap, x = heights(font)
    base = [unicodedata.normalize("NFD", c)[0] for c in text if c.isalnum()]
    tall = upper or any(c.isupper() or c.isdigit() or c in TALL_CHARS for c in base)
    return cap if tall else x


def _line_size(l: dict, font: str | None, text: str, upper: bool) -> float:
    """Cỡ chữ (px) để chữ dựng cao bằng chữ nháp của dòng: cao nét đo / cao chữ của font. Dòng không có số đo nét: cỡ cũ của slots."""
    h = (l.get("ink") or {}).get("h_cap")
    return float(h) / letter_em(font, text, upper) if h and not (l.get("path") or l.get("vertical")) else float(l["size"])


ROOM = 1.25      # khung chữ được nới tới 1,25 lần bề rộng nét nháp (chữ thật rộng hơn chữ giun ~11%: không phải co)


ROOM_DE = 15.0   # ... vào cột nền trống: >= ROOM_FREE điểm của cột cùng màu nền trong khung (ΔE <= 15, nền đã gỡ chữ) ...


ROOM_FREE = 0.97


ROOM_GAP = 0.3   # ... không vào vùng dòng khác (nới 0,3 cao dòng) -- không vượt vỏ / pill, không chạm chữ khác


ROOM_MARGIN = 0.2   # dừng vì chạm vật (vành, mép pill, dòng khác): chừa khe 0,2 cao dòng -- chữ không dính sát vành


def room(box: list, align: str, lab: np.ndarray, blocked: np.ndarray) -> list:
    """Khung nới NGANG vào nền trống cùng màu (người dùng chốt 15/10: cho nới, không vượt vùng an toàn). Lề trái: nới sang phải;
    lề phải: sang trái; giữa: đều hai bên. Dừng ở cột đầu tiên không phải nền (mép vỏ, hình, dòng khác, mép ảnh)."""
    H, W = blocked.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 2:
        return list(box)
    bg = np.median(lab[y0:y1, max(0, x0):min(W, x1)].reshape(-1, 3), axis=0)
    extra = int((ROOM - 1) * (x1 - x0))

    def free(x):
        if not 0 <= x < W or blocked[y0:y1, x].any():
            return False
        return float(np.mean(np.linalg.norm(lab[y0:y1, x] - bg, axis=1) <= ROOM_DE)) >= ROOM_FREE

    def reach(x, step, n):
        k = 0
        while k < n and free(x + step * (k + 1)):
            k += 1
        return k if k == n else max(0, k - int(ROOM_MARGIN * (y1 - y0)))
    if align == "left":
        return [box[0], box[1], box[2] + reach(x1 - 1, 1, extra), box[3]]
    if align == "right":
        return [box[0] - reach(x0, -1, extra), box[1], box[2], box[3]]
    k = min(reach(x1 - 1, 1, extra // 2), reach(x0, -1, extra // 2))   # giữa: nới đều để tâm chữ đứng yên
    return [box[0] - k, box[1], box[2] + k, box[3]]


def _rooms(T: list[dict], M: dict, base: np.ndarray, log: list[str]) -> None:
    """Nới khung các phần tử ngang (room) trên nền đã gỡ chữ; vùng cấm = vùng mọi dòng KHÁC của poster."""
    from ..color import lab as to_lab
    from ..erase.mask import line_region
    H, W = base.shape[:2]
    L = to_lab(base)
    reg = {l["id"]: line_region(l, H, W, ROOM_GAP) for l in M["L"]}
    allm = np.zeros((H, W), np.uint8)
    for r in reg.values():
        allm += r
    for u in T:
        own = np.zeros((H, W), np.uint8)
        for m in u["marks"]:
            own += reg[m]
        blocked = (allm - own) > 0
        for e in u.get("els") or []:
            if e["path"] or abs(e["angle"]) >= 2:
                continue
            nb = room(e["box"], u["align"], L, blocked)
            if nb[2] - nb[0] > e["box"][2] - e["box"][0] + 1:
                log.append(f"{e['id']}: khung nới {nb[2] - nb[0] - (e['box'][2] - e['box'][0]):.0f}px vào nền trống ({u['align']})")
                e["box"] = nb
