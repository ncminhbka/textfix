"""CONTACT SHEET ảnh ứng viên của kho tham chiếu (scripts/kho_fetch.py tải về): mỗi khái niệm một ảnh lưới có đánh số ô +
giấy phép, để chọn ảnh bằng mắt rồi ghi vào picks.json cho scripts/kho_index.py.

  python scripts/kho_sheet.py <thư mục tạm> [id ...]      -> <thư mục tạm>/_sheets/<id>.jpg
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL, COLS = 360, 4


def _font(size: int):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sheet(d: Path) -> Image.Image:
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    rows = (len(meta) + COLS - 1) // COLS
    out = Image.new("RGB", (COLS * CELL, max(1, rows) * (CELL + 40) + 40), "white")
    dr, f = ImageDraw.Draw(out), _font(18)
    dr.text((8, 8), d.name, fill="black", font=_font(24))
    for k, m in enumerate(meta):
        x, y = (k % COLS) * CELL, 40 + (k // COLS) * (CELL + 40)
        try:
            im = Image.open(d / m["file"]).convert("RGB")
            im.thumbnail((CELL - 8, CELL - 8))
            out.paste(im, (x + (CELL - im.width) // 2, y + (CELL - im.height) // 2))
        except Exception:
            dr.text((x + 8, y + 8), "lỗi ảnh", fill="red", font=f)
        dr.text((x + 6, y + CELL + 2), f"#{k} {m.get('license', '')[:28]}", fill="black", font=f)
        dr.text((x + 6, y + CELL + 20), m["title"].removeprefix("File:")[:34], fill="gray", font=_font(14))
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1])
    (root / "_sheets").mkdir(exist_ok=True)
    for d in sorted(p for p in root.iterdir() if (p / "meta.json").is_file() and (not sys.argv[2:] or p.name in sys.argv[2:])):
        sheet(d).save(root / "_sheets" / f"{d.name}.jpg", quality=85)
        print(d.name)
