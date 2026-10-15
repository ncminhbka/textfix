"""Ngắt dòng câu khách theo nháp (tỉ lệ số ký tự các dòng nháp) và nhận chữ viết hoa.
"""

from __future__ import annotations


def break_like(text: str, caps: list, meas=len) -> list[str]:
    """Chia câu thành len(caps) dòng theo SỨC CHỨA từng dòng nháp: caps = bề rộng nét / cỡ chữ (em), meas = bề rộng em của chữ
    theo font thật (fonts.advance); dòng cong: caps = số ký tự OCR, meas = len. Chọn cách ngắt cho dòng chật nhất đỡ chật nhất
    -- cả khối co theo dòng chật nhất, nên đây là cách giữ cỡ to nhất; hoà thì các dòng đều nhau hơn. (14/10 mix1: ngắt theo tỉ
    lệ số ký tự OCR của chữ giun ra "toàn / bộ / mỹ phẩm / chăm sóc da", "Thời gian của riêng / anh" -- chữ giun sai số ký tự,
    bề rộng nét thì không.)"""
    w = text.split()
    n = min(len(caps), len(w))
    if n <= 1:
        return [text]
    caps = caps[:n]
    # quy hoạch động: (từ đã dùng, số dòng) -> cách ngắt tốt nhất theo (chật nhất, tổng bình phương)
    best = {0: ((0.0, 0.0), [])}
    for j in range(n):
        nxt = {}
        for i, ((mx, sq), br) in best.items():
            for e in range(i + 1, len(w) - (n - 1 - j) + 1):
                r = meas(" ".join(w[i:e])) / max(1e-6, caps[j])
                v = ((max(mx, r), sq + r * r), br + [e])
                if e not in nxt or v[0] < nxt[e][0]:
                    nxt[e] = v
        best = nxt
    cuts = [0] + best[len(w)][1]
    return [" ".join(w[a:b]) for a, b in zip(cuts, cuts[1:])]


def _caps(u: dict) -> list[float]:
    """Sức chứa các dòng nháp của đơn vị: bề rộng nét / cỡ chữ (em)."""
    return [max(0.5, (b[2] - b[0]) / max(1.0, l["size"])) for b, l in zip(u["ib"], u["Ls"])]
