"""Chữ in trên SẢN PHẨM khách tải: dòng của nháp có chữ OCR gần trùng chữ đọc được trên ảnh sản phẩm -> luôn giữ.
Cùng phép so (bỏ dấu, chữ thường) dùng để nhận chữ OCR giống câu khách.
"""

from __future__ import annotations

import re


def mark_printed(M: dict, product_ocr: list | None) -> list[str]:
    """CHỮ IN TRÊN SẢN PHẨM khách tải (13/10, người dùng: PHẢI GIỮ, không thay bằng bất cứ gì): dòng của nháp có chữ OCR gần
    trùng một dòng chữ đọc được trên ẢNH SẢN PHẨM (FLUX chép sản phẩm vào nháp rất sát) -> l["printed"] = True. Dòng đó luôn
    là "keep" (validate), VLM được báo trong bảng mô tả. -> id các dòng."""
    words = [{"text": w} for w in product_ocr or [] if w and w.strip()]
    out = []
    for l in M["L"] if words else []:
        ocr = l.get("ocr") or ""
        if _client_like(ocr, words) is not None and not _foreign_words(ocr, words):
            l["printed"] = True
            out.append(l["id"])
    return out


FOREIGN_SIM = 0.4   # một từ (>= 3 ký tự) của dòng nháp không giống từ nào trên ảnh sản phẩm (bỏ dấu, chữ thường, < 0,4) = chữ của
                    # poster: dòng CHỨA tên sản phẩm mà không phải chữ trên sản phẩm (final3 h17: tiêu đề "Bia Sapppro" cạnh "SAPPORO"
                    # trên lon). Đo trên 32 poster có ảnh sản phẩm: bắt 1 (h17), không bỏ nhầm dòng sản phẩm nào; 0,5 bỏ nhầm "CEW"


def _foreign_words(ocr: str, words: list[dict]) -> list[str]:
    """Các từ của dòng nháp không giống từ nào đọc được trên ảnh sản phẩm."""
    from difflib import SequenceMatcher
    pw = [_flat(x) for w in words for x in w["text"].split() if len(_flat(x)) >= 2]
    return [x for x in ocr.split() if len(_flat(x)) >= 3
            and max((SequenceMatcher(None, _flat(x), p).ratio() for p in pw), default=0.0) < FOREIGN_SIM]


CLIENT_LIKE = 0.6   # chữ OCR của ô giống câu khách >= 0.6 (bỏ dấu, chữ thường) = chữ poster chứa câu khách


def _client_like(ocr: str, texts: list[dict]) -> int | None:
    """Chữ OCR (méo) của ô gần trùng một câu khách (bỏ dấu, chữ thường, giống >= CLIENT_LIKE) -> chỉ số câu đó."""
    from difflib import SequenceMatcher
    f = _flat(ocr or "")
    if len(f) < 4:
        return None
    best = max(((SequenceMatcher(None, f, _flat(t["text"])).ratio(), i) for i, t in enumerate(texts)), default=(0, None))
    if best[0] >= CLIENT_LIKE:
        return best[1]
    # OCR chỉ đọc được MẨU của câu (chữ cong trên ruy băng: "TRIAN" của "QUÀ TẶNG TRI ÂN") -> mẩu nằm gọn trong câu
    for i, t in enumerate(texts):
        m = SequenceMatcher(None, f, _flat(t["text"]))
        # khớp LIỀN một đoạn (ký tự rải rác trong câu dài 150 ký tự không tính: "Noinlien Hic" trên thân nồi, dev3 p09)
        if len(f) >= 5 and m.find_longest_match(0, len(f), 0, len(_flat(t["text"]))).size >= 0.8 * len(f):
            return i
    return None


def _fold(s: str) -> str:
    """Bỏ dấu tiếng Việt, chữ thường: "TRƯỞNG" == "trương" == "TRUONG"."""
    import unicodedata
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _flat(s: str) -> str:
    return re.sub(r"\W", "", _fold(s))
