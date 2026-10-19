"""Hợp đồng của prompt FLUX: chữ trong ngoặc kép ĐÚNG BẰNG danh sách chữ đã chốt, mỗi câu một lần, và vừa giới hạn token.

FLUX vẽ mọi chuỗi trong ngoặc kép nó đọc được và chỉ những chuỗi đó, nên đây là chỗ duy nhất prompt bị sửa:
  1. thống nhất ngoặc ('...', “...” -> "...");
  2. áp các cặp thay thế từ bước chốt chữ (câu chép sai, mục danh sách, số sao);
  3. chuỗi không thuộc danh sách: từ tả tiếng Anh thì bỏ ngoặc; chữ lạ thì bỏ (cả câu nếu câu chỉ mang chữ lạ);
  4. câu nhắc lại lần hai bỏ ngoặc; bỏ tỉ lệ / độ phân giải;
  5. câu chưa có thì chèn sau câu đầu (FLUX cắt phần cuối khi quá dài, nên không nối vào cuối);
  6. ảnh tham chiếu kho (món / trang phục Việt) chưa được nhắc "image N" thì chèn câu nhắc sau câu đầu;
  7. quá giới hạn token thì bỏ dần câu tả cảnh ở cuối.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .texts import Notes

QUOTED = re.compile(r'"([^"]+)"')
OTHER_QUOTES = re.compile(r"“([^”\n]{1,120})”|‘([^’\n]{1,120})’|(?<![\w'])'(?=\S)([^'\n]{2,120}?)(?<=\S)'(?![\w'])")
IMAGE_REF = re.compile(r"\bimages? \d\b", re.I)   # câu nhắc ảnh tham chiếu ("the product from image 1")
RATIO = re.compile(r"\b(?:1:1|9:16|16:9|4:5|5:4|2:3|3:2|4:3|3:4|21:9)\b|\b\d{3,4}p\b|\b[248]k\b", re.I)
ORPHAN_PUNCT = re.compile(r"(?:\s*(?:,|\band\b|\bor\b))+\s*(?=[.,;:!?])|\s+(?=[.,;:!?])")
# FLUX.2 klein mã hoá prompt bằng Qwen3, tối đa 512 token kể cả khuôn chat (12 token) -- flux2/text_encoder.py
TOKEN_LIMIT = 512 - 12


def unify_quotes(pe: str) -> str:
    """Mọi kiểu ngoặc -> ngoặc kép. Dấu nháy trong từ tiếng Anh ("shop's") không phải ngoặc; cụm nhắc ảnh tham chiếu không
    phải chữ in nên bỏ ngoặc."""
    def other(m):
        g = next(g for g in m.groups() if g is not None)
        return g if IMAGE_REF.search(g) else f'"{g}"'
    pe = OTHER_QUOTES.sub(other, pe)
    return QUOTED.sub(lambda m: m.group(1) if IMAGE_REF.search(m.group(1)) else m.group(0), pe)


def compose(pe: str, texts: list[dict], rewrites: list[tuple[str, str]], refs: list[tuple[int, str]] = ()) -> tuple[str, Notes]:
    """Áp hợp đồng (xem đầu tệp) lên một prompt. refs: (số ảnh, tên món tiếng Anh) của ảnh kho. -> (prompt, nhật ký)."""
    notes = Notes()
    for old, new in rewrites:
        pe = pe.replace(old, new)
    pe = _match_case(pe, texts)
    pe = _fix_typos(pe, texts, notes)
    pe = _remove_foreign(pe, texts, notes)
    pe = _quote_once(pe)
    pe = _outside_quotes(pe, lambda s: ORPHAN_PUNCT.sub("", RATIO.sub("", s)))
    pe = _insert_missing(pe, texts, notes)
    pe = _cite_refs(pe, refs, notes)
    pe = _fit(" ".join(pe.split()), notes)
    return pe, notes


def _outside_quotes(pe: str, f) -> str:
    parts = pe.split('"')
    return '"'.join(f(p) if i % 2 == 0 else p for i, p in enumerate(parts))


def _match_case(pe: str, texts: list[dict]) -> str:
    """Chuỗi trùng một câu khi không phân biệt hoa / thường -> viết đúng như câu."""
    by = {t["text"].casefold(): t["text"] for t in texts}
    return QUOTED.sub(lambda m: '"' + by.get(" ".join(m.group(1).split()).casefold(), m.group(1)) + '"', pe)


TYPO = 0.85   # chuỗi trong ngoặc giống >= 85% một câu đã chốt chưa có trong prompt = LLM chép sai câu đó


def _fix_typos(pe: str, texts: list[dict], notes: Notes) -> str:
    """LLM chép sai vài ký tự của một câu khi viết prompt ("COMBO GIA ĐỊNH" cho "COMBO GIA ĐÌNH"): sửa TẠI CHỖ thành câu đúng,
    giữ vị trí của chữ trong câu tả (gỡ đi rồi chèn lại thì chữ mất chỗ của nó)."""
    from difflib import SequenceMatcher
    quoted = {" ".join(q.split()).casefold() for q in QUOTED.findall(pe)}
    free = [t["text"] for t in texts if t["text"].casefold() not in quoted]

    def one(m):
        q = m.group(1)
        if not free or approved(q, texts):
            return m.group(0)
        best = max(free, key=lambda t: SequenceMatcher(None, q.casefold(), t.casefold()).ratio())
        if SequenceMatcher(None, q.casefold(), best.casefold()).ratio() < TYPO:
            return m.group(0)
        free.remove(best)
        notes.add("typo_fixed", f"{q} -> {best}")
        return f'"{best}"'
    return QUOTED.sub(one, pe)


def approved(q: str, texts: list[dict]) -> bool:
    """Chuỗi là chữ đã chốt: trùng / nằm trong một câu, hoặc ghép từ các câu."""
    q = " ".join(q.split()).casefold()
    tx = [t["text"].casefold() for t in texts]
    parts = [t for t in tx if t in q]
    return any(q in t for t in tx) or (bool(parts) and sum(map(len, parts)) >= 0.8 * len(q))


def _remove_foreign(pe: str, texts: list[dict], notes: Notes) -> str:
    foreign = {q for q in QUOTED.findall(pe) if not approved(q, texts)}
    # từ tả tiếng Anh LLM để trong ngoặc ('tree pose', 'before/after'): không số, không dấu -> chỉ bỏ ngoặc
    descriptive = {q for q in foreign if q.isascii() and not re.search(r"\d", q)}
    pe = QUOTED.sub(lambda m: m.group(1) if m.group(1) in descriptive else m.group(0), pe)
    foreign -= descriptive
    if not foreign:
        return pe
    notes["quoted_removed"] = sorted(foreign)
    keep = []
    for i, s in enumerate(_sentences(pe)):
        qs = set(QUOTED.findall(s))
        if not qs & foreign:
            keep.append(s)
        elif i == 0 or IMAGE_REF.search(s) or qs - foreign:
            # câu đầu (chủ đề), câu nhắc ảnh tham chiếu, câu còn chữ đúng: chỉ bỏ chuỗi lạ
            keep.append(QUOTED.sub(lambda m: "" if m.group(1) in foreign else m.group(0), s))
        # còn lại: câu chỉ mang chữ lạ -> bỏ cả câu (không để vật trống chữ cho FLUX tự vẽ chữ giun)
    return "".join(keep)


def _quote_once(pe: str) -> str:
    seen = set()

    def once(m):
        k = " ".join(m.group(1).split()).casefold()
        if k in seen:
            return m.group(1)
        seen.add(k)
        return m.group(0)
    return QUOTED.sub(once, pe)


def _insert_missing(pe: str, texts: list[dict], notes: Notes) -> str:
    have = [" ".join(q.split()).casefold() for q in QUOTED.findall(pe)]
    miss = [t["text"] for t in texts if not any(t["text"].casefold() in h for h in have)]
    if not miss:
        return pe
    notes["quoted_added"] = miss
    s = _sentences(pe)
    add = " The poster also shows the texts " + ", ".join(f'"{t}"' for t in miss) + "."
    if not s:
        return add.strip()
    head = s[0].rstrip()
    return head + ("" if head[-1:] in ".!?" else ".") + add + "".join(s[1:])


def _cite_refs(pe: str, refs: list[tuple[int, str]], notes: Notes) -> str:
    """Ảnh kho FLUX nhận mà prompt không nhắc "image N" thì FLUX không biết ảnh đó là món nào: chèn câu nhắc sau câu đầu."""
    said = {int(m.group(0)[-1]) for m in IMAGE_REF.finditer(pe)}
    miss = [(n, name) for n, name in refs if n not in said]
    if not miss:
        return pe
    notes["refs_cited"] = [f"{name} = image {n}" for n, name in miss]
    add = "".join(f" The {name} looks exactly like the {name} from image {n}." for n, name in miss)
    s = _sentences(pe)
    if not s:
        return add.strip()
    head = s[0].rstrip()
    return head + ("" if head[-1:] in ".!?" else ".") + add + "".join(s[1:])


def _fit(pe: str, notes: Notes) -> str:
    """Bỏ dần câu tả cảnh ở cuối (không chứa chữ, không nhắc ảnh tham chiếu) cho tới khi vừa TOKEN_LIMIT."""
    cut = 0
    while count_tokens(pe) > TOKEN_LIMIT:
        s = _sentences(pe)
        i = next((j for j in range(len(s) - 1, 0, -1) if '"' not in s[j] and not IMAGE_REF.search(s[j])), None)
        if i is None:
            break
        pe = " ".join("".join(s[:i] + s[i + 1:]).split())
        cut += 1
    if cut:
        notes["trimmed_sentences"] = cut
    return pe


def _sentences(pe: str) -> list[str]:
    """Tách câu, giữ khoảng trắng; không tách trong ngoặc kép ("TP. HCM", "Mở cửa 7:00!")."""
    out, cur, inq = [], "", False
    for i, ch in enumerate(pe):
        cur += ch
        if ch == '"':
            inq = not inq
        elif ch in ".!?" and not inq and (i + 1 == len(pe) or pe[i + 1].isspace()):
            out.append(cur)
            cur = ""
    return out + ([cur] if cur.strip() else [])


_tokenizer = None


def count_tokens(s: str) -> int:
    """Số token của bộ mã hoá chữ FLUX. Tokenizer tìm ở máy (TEXT_ENCODER_PATH, thư mục tokenizer cạnh nó, cache HF), không
    tải mạng; không có thì ước 0,33 token / byte UTF-8 (đo trên prompt đã lưu: 0,20-0,31, nên ước này dư an toàn)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = False
        te = os.environ.get("TEXT_ENCODER_PATH")
        for spec in ([str(Path(te).parent / "tokenizer"), te] if te else []) + ["Qwen/Qwen3-4B"]:
            try:
                from transformers import AutoTokenizer
                _tokenizer = AutoTokenizer.from_pretrained(spec, local_files_only=True)
                break
            except Exception:
                continue
    return len(_tokenizer(s)["input_ids"]) if _tokenizer else int(len(s.encode()) * 0.33) + 1
