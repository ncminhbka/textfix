"""So chữ không phân biệt dấu / hoa thường: "TRƯỞNG" == "trương" == "TRUONG" (OCR của chữ méo ít khi đúng dấu)."""

from __future__ import annotations

import re


def _fold(s: str) -> str:
    """Bỏ dấu tiếng Việt, chữ thường: "TRƯỞNG" == "trương" == "TRUONG"."""
    import unicodedata
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _flat(s: str) -> str:
    return re.sub(r"\W", "", _fold(s))
