"""Đối chiếu câu LLM viết với lời khách: CHỮ (không phân biệt hoa / thường, bỏ dấu, gần đúng) và SỰ THẬT (số, liên kết, email).

Lời khách thường bẩn (không ngoặc kép, viết thường, gõ không dấu, lỗi gõ) nên không đòi khớp nguyên văn. Cam kết cần giữ là:
không có số / liên kết nào khách không viết, và câu ghi là "của khách" phải mang phần lớn chữ của khách.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import get_close_matches

WORD = re.compile(r"[^\W\d_]+")   # một chữ (chỉ chữ cái)
# liên kết dừng ở khoảng trắng / ngoặc / dấu nháy (khách viết link trong ngoặc kép: "www.ironfit.vn")
LINK = re.compile(r"(?:https?://|www\.)[^\s\"'“”()<>]+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
                  r"|\b[\w-]+(?:\.[\w-]+)*\.(?:vn|com|net|org|io|me)\b(?:/[^\s\"'“”()<>]*)?", re.I)
NUMBER = re.compile(r"\d(?:[\d.,]|\s(?=\d))*")   # "185.000", "1,5", "0909 123 456"
FUZZY = 0.8   # chữ >= 4 ký tự giống >= 80% một chữ của khách vẫn tính là của khách (khách gõ sai một ký tự)


def fold(s: str) -> str:
    """Dạng so sánh: chữ thường, bỏ dấu tiếng Việt."""
    s = unicodedata.normalize("NFD", s.casefold().replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def numbers(s: str) -> list[tuple[str, list[str]]]:
    """Các số trong câu (ngoài liên kết): (dãy chữ số liền, [các phần cách nhau khoảng trắng]).
    "0909 123 456" -> ("0909123456", ["0909", "123", "456"]); "185.000đ" -> ("185000", ["185000"])."""
    out = []
    for m in NUMBER.finditer(LINK.sub(" ", s)):
        g = m.group(0).rstrip(".,")
        parts = [d for d in (re.sub(r"\D", "", p) for p in g.split()) if d]
        out.append((re.sub(r"\D", "", g), parts))
    return out


def _link_key(s: str) -> str:
    return fold(s).rstrip("/.,")


class Evidence:
    """Tập chữ và tập sự thật của một hay nhiều đoạn văn (lời khách, hoặc các câu đã chốt)."""

    def __init__(self, passages: list[str]):
        text = " ".join(passages)
        self.passages = passages
        self._folded = [fold(p) for p in passages]
        self._words = {fold(w) for w in WORD.findall(text)}
        self._word_list = sorted(self._words)
        self._numbers = {d for whole, parts in numbers(text) for d in [whole, *parts]}
        self._links = {_link_key(x) for x in LINK.findall(text)}

    def facts(self, s: str) -> list[str]:
        """Số và liên kết của câu."""
        return [whole for whole, _ in numbers(s)] + LINK.findall(s)

    def unknown_facts(self, s: str) -> list[str]:
        """Số / liên kết của câu KHÔNG có ở đây. Số nhiều nhóm ("0909 123 456") khớp khi cả dãy hoặc từng nhóm có mặt."""
        bad = [x for x in LINK.findall(s) if _link_key(x) not in self._links]
        bad += [whole for whole, parts in numbers(s)
                if whole not in self._numbers and not (parts and all(p in self._numbers for p in parts))]
        return bad

    def coverage(self, s: str) -> tuple[float, list[str]]:
        """(tỉ lệ chữ của câu có ở đây, các chữ không có). Câu không có chữ cái (chỉ số) tính là đủ."""
        words = WORD.findall(LINK.sub(" ", s))
        extra = [w for w in words if not self.has_word(w)]
        return (1 - len(extra) / len(words)) if words else 1.0, extra

    def has_word(self, w: str) -> bool:
        f = fold(w)
        return f in self._words or (len(f) >= 4 and bool(get_close_matches(f, self._word_list, n=1, cutoff=FUZZY)))

    def holds(self, s: str, min_coverage: float) -> bool:
        """Câu s đã nằm ở đây: là một đoạn con, hoặc đủ chữ và mọi số / liên kết của nó có mặt."""
        if any(fold(s) in p for p in self._folded):
            return True
        return self.coverage(s)[0] >= min_coverage and not self.unknown_facts(s)
