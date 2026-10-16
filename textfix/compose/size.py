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


def room(box: list, align: str, lab: np.ndarray, blocked: np.ndarray, want: float = 0.0) -> list:
    """Khung nới NGANG vào nền trống cùng màu (người dùng chốt 15/10: cho nới, không vượt vùng an toàn). Lề trái: nới sang phải;
    lề phải: sang trái; giữa: đều hai bên. Dừng ở cột đầu tiên không phải nền (mép vỏ, hình, dòng khác, mép ảnh). Nới tối đa
    (ROOM - 1) bề rộng khung, hoặc `want` px khi lớn hơn (tới bề rộng cột của tập, compose.size.column_want)."""
    H, W = blocked.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 2:
        return list(box)
    bg = np.median(lab[y0:y1, max(0, x0):min(W, x1)].reshape(-1, 3), axis=0)
    extra = int(max((ROOM - 1) * (x1 - x0), want))
    # vùng cấm xét ở NỬA GIỮA chiều cao dòng (thân chữ): vùng dòng khác nới dọc cho dấu thanh lấn vào mép trên / dưới của hàng
    # danh sách liền kề -- không phải vật chắn ngang; dòng khác CÙNG HÀNG (giá cạnh tên món) vẫn chặn
    q = (y1 - y0) // 4
    b0, b1 = y0 + q, max(y0 + q + 1, y1 - q)

    def free(x):
        if not 0 <= x < W or blocked[b0:b1, x].any():
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


def column_want(T: list[dict]) -> dict:
    """BỀ RỘNG CỘT CHUNG của tập (người dùng chốt 16/10: cùng danh sách cùng cỡ): các phần tử một dòng của tập thẳng mép theo lề
    tập (trái: mép trái lệch <= EDGE_TOL cỡ; phải: mép phải; giữa: tâm lệch <= CENTER_TOL cỡ) là một cột; mỗi phần tử được nới tới
    bề rộng phần tử rộng nhất cột (chữ giun FLUX vẽ dài ngắn tuỳ hứng: "Trá" 67px cạnh "Cồm sũron bí chà" 333px). -> {id phần tử:
    số px muốn nới}."""
    from .geometry import CENTER_TOL, EDGE_TOL
    out = {}
    for name in {u["set"] for u in T if u.get("set")}:
        us = [u for u in T if u.get("set") == name]
        al, s = us[0]["align"], us[0]["size"]
        E = [e for u in us for e in u.get("els") or [] if not e["path"] and abs(e["angle"]) < 2 and not e["multi"]]
        key = {"left": lambda b: b[0], "right": lambda b: b[2]}.get(al, lambda b: (b[0] + b[2]) / 2)
        tol = (CENTER_TOL if al == "center" else EDGE_TOL) * s
        for e in E:
            col = [f for f in E if abs(key(f["box"]) - key(e["box"])) <= tol]
            w = max(f["box"][2] - f["box"][0] for f in col)
            if w > e["box"][2] - e["box"][0] + 1:
                out[e["id"]] = w - (e["box"][2] - e["box"][0])
    return out


def _blocker(l: dict, H: int, W: int, line_region) -> np.ndarray:
    """Vùng cấm của một dòng khi nới khung dòng khác: dòng thẳng = khung dòng nới ROOM_GAP cao dòng sang ngang (vùng gỡ chữ
    erase.line_region nới dọc rộng cho dấu thanh, lấn sang hàng danh sách liền kề); dòng cong / nghiêng: vùng gỡ chữ."""
    if l.get("path") or abs(l.get("angle") or 0) >= 2:
        return line_region(l, H, W, ROOM_GAP)
    x0, y0, x1, y1 = l["box"]
    g = ROOM_GAP * (y1 - y0)
    m = np.zeros((H, W), np.uint8)
    m[max(0, int(y0)):min(H, int(np.ceil(y1))), max(0, int(x0 - g)):min(W, int(np.ceil(x1 + g)))] = 1
    return m


def _rooms(T: list[dict], M: dict, base: np.ndarray, log: list[str], gone: set = frozenset()) -> None:
    """Nới khung các phần tử ngang (room) trên nền đã gỡ chữ; vùng cấm = vùng mọi dòng KHÁC của poster còn chữ (dòng gỡ trống --
    `gone`: rác, hàng thừa -- không còn gì để chạm). Phần tử của tập nới tới bề rộng cột chung (column_want). Nới LẦN LƯỢT: phần
    đã nới của đơn vị trước thành vùng cấm của đơn vị sau (tên món và giá cùng hàng không nới vào chung một khoảng trống tới
    chạm nhau)."""
    from ..color import lab as to_lab
    from ..erase.mask import line_region
    H, W = base.shape[:2]
    L = to_lab(base)
    reg = {l["id"]: _blocker(l, H, W, line_region) for l in M["L"] if l["id"] not in gone}
    allm = np.zeros((H, W), np.uint8)
    for r in reg.values():
        allm += r
    want = column_want(T)
    grown = np.zeros((H, W), bool)   # phần khung đã nới của các đơn vị trước (nới thêm khe ROOM_GAP cao dòng)
    for u in T:
        own = np.zeros((H, W), np.uint8)
        for m in u["marks"]:
            if m in reg:
                own += reg[m]
        blocked = ((allm - own) > 0) | grown
        mine = []
        for e in u.get("els") or []:
            if e["path"] or abs(e["angle"]) >= 2:
                continue
            nb = room(e["box"], u["align"], L, blocked, want.get(e["id"], 0.0))
            if nb[2] - nb[0] > e["box"][2] - e["box"][0] + 1:
                log.append(f"{e['id']}: khung nới {nb[2] - nb[0] - (e['box'][2] - e['box'][0]):.0f}px vào nền trống ({u['align']})")
                e["box"] = nb
                mine.append(nb)
        for b in mine:
            g = ROOM_GAP * (b[3] - b[1])
            grown[max(0, int(b[1])):min(H, int(np.ceil(b[3]))), max(0, int(b[0] - g)):min(W, int(np.ceil(b[2] + g)))] = True
