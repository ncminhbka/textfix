"""Ảnh debug các đơn vị chữ (màu theo số phận / theo tập).
"""

from __future__ import annotations

from .geometry import op_box
from .page import plain

KIND_COL = {"text": (230, 30, 30), "keep": (0, 160, 80), "skip": (170, 170, 170)}


SET_COLS = [(30, 110, 230), (200, 120, 0), (150, 40, 200), (0, 150, 150), (200, 40, 120), (90, 130, 20)]


def ops_image(img, M, P):
    """Nháp mờ + khung từng đơn vị chữ (compose.layout.public) + nhãn: vai / tập và chữ sẽ viết. Màu: mỗi TẬP cùng vai một màu (các mục
    cùng màu là được dựng giống hệt nhau); đơn vị lẻ đỏ; giữ xanh lá; xoá xám."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray((img * 0.55 + 115).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("arial.ttf", max(11, img.shape[1] // 90))
    except OSError:
        f = ImageFont.load_default()
    sets = sorted({op["set"] for op in P["ops"] if op.get("set")})
    for op in P["ops"]:
        b = op_box(op, M)
        if not b:
            continue
        col = SET_COLS[sets.index(op["set"]) % len(SET_COLS)] if op.get("set") else KIND_COL.get(op.get("kind"), (0, 0, 0))
        b = [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])]
        d.rectangle(b, outline=col, width=5 if op.get("set") else 3)
        tag = op.get("set") or op.get("role") or op.get("kind")
        lab = f"{tag}: {plain(op.get('html', ''))[:40]}"
        d.text((b[0] + 3, b[1] + 2), lab, fill=col, font=f)
    return im
