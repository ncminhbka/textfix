"""Font theo khoá của danh mục (fonts.CATALOG) và độ đậm mặc định theo cỡ.
"""

from __future__ import annotations

import re


def font_key(k) -> str | None:
    """Khoá font VLM ghi -> khoá CATALOG: đúng khoá, tên họ font ("Dancing Script"), hoặc gần nhất ("bevietsans" -> bevietnam)."""
    from difflib import get_close_matches

    from .fonts import CATALOG
    if not isinstance(k, str) or not k.strip():
        return None
    if k in CATALOG:
        return k
    norm = lambda x: re.sub(r"[^a-z]", "", x.lower())   # noqa: E731
    names = {norm(v[0]): key for key, v in CATALOG.items()} | {norm(key): key for key in CATALOG}
    m = get_close_matches(norm(k), list(names), n=1, cutoff=0.6)
    return names[m[0]] if m else None


WEIGHT = lambda m: (300 if m < 0.07 else 400 if m < 0.11 else 700 if m < 0.16 else 900)   # noqa: E731  nét / cỡ chữ -> độ đậm (như compose.vlm._look)


HEAD = 1.4   # dòng cao >= 1.4 cỡ chữ trung vị của poster = tiêu đề -> mặc định style.display_font
