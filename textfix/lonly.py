"""CHẶNG B, phần VLM: ô dòng chữ để VLM làm việc (view), mô tả dòng (describe_l), lời dặn VLM (SYSTEM_L), gọi VLM (plan).

Người dùng chốt sau bộ accept1: không bỏ dòng nào nháp đã vẽ, thiếu câu khách thì VLM viết cho đủ; mọi thứ ngoài dòng chữ -- vỏ,
icon to, hoạ tiết, hàng sao, avatar -- là điểm ảnh của nháp. VLM chỉ nhận dòng chữ. Gỡ chữ (bước 3): textfix/erase.py.
"""

from __future__ import annotations

import numpy as np


def view(M: dict) -> dict:
    """Ô để VLM và bộ dựng làm việc: chỉ dòng chữ (plan.json cũ có thể còn S / I / shell: bỏ)."""
    L = [{k: v for k, v in l.items() if k != "shell"} for l in M["L"]]
    return {"L": L, "size": M["size"], "by": {l["id"]: l for l in L}, "geo_log": M.get("geo_log") or []}


COVER_MIN = 0.6   # VLM ghép < 60% số dòng = câu trả lời bị cụt (mix1 p02: 1 lệnh / 10 dòng) -> gọi lại một lần


def describe_l(M: dict) -> str:
    from . import slots
    note = lambda m: (slots._curve_note(m) if m.get("path") else f", drawn at {m['angle']:.0f} degrees" if abs(m.get("angle") or 0) >= 2   # noqa: E731
                      else "") + (", found by the stroke detector (big decorated lettering)" if m.get("missed") and not m.get("orphan") else "") \
        + (", its bullet dot was wiped with the letters (\"bullet\": true puts it back)" if m.get("dot") else "") \
        + (", STRAY LETTERING (letter shapes that may be junk, a logo or decoration)" if m.get("orphan") else "") \
        + (", PRINTED ON THE CLIENT'S PRODUCT: kind \"keep\"" if m.get("printed") else "")
    # CHỮ OCR GỬI LẠI làm NEO (người dùng 14/10 tối: "đưa OCR hint quay lại nếu thấy performance giảm"; mix1 v03: thiếu chữ OCR,
    # VLM ghép ngày vào mẩu 21px và xoá dòng OCR đọc đúng "01/12 - 07/12"). Chỉ để VLM biết L# nào là dòng nào -- code không
    # dùng chữ này để viết.
    li = "\n".join(f"{m['id']}: machine reading {(m.get('ocr') or '').strip()!r}, ~{m['size']:.0f}px, color {m['color']}{note(m)}"
                   f"{slots._look(m)}" for m in M["L"]) or "(none)"
    return f"Text lines:\n{li}"


SYSTEM_L = """You are a senior graphic designer finishing a poster. An image model drew the DRAFT from the PROMPT below. Everything in it \
is good except its lettering, which is garbled and misspelt. Code measured every TEXT LINE of the draft (red L# on SLOTS), \
wiped the old letters (BASE) and will set the CLIENT TEXTS there itself, word for word, at the size, place and alignment the \
draft shows. YOUR JOB IS TO READ, NOT TO WRITE: say which client text each drawn line was meant to be, and how each kind of \
text looks in the draft.

1. ASSIGN EVERY LINE. Look at the DRAFT and SLOTS and read what each marked line was meant to say (the drawn letters are \
garbled: "Hỏ mây ầm" is "Hồ nước ấm"). Every L# goes in exactly \
one "assign" entry. A client text the model broke over several stacked lines is ONE entry with all those lines (["L3","L4","L5"]). \
One drawn line holding two client texts (a dish name and its price) is one entry with both ("client": ["T5","T6"]), in \
reading order. Never assign a client text twice.
2. ONE TEXT, LINES OF CLEARLY DIFFERENT SIZE OR IN SEPARATE PLACES (a small "THANH TOÁN" over a huge "NHANH CHÓNG"; "NGÀY" and \
"HỘI" in two boxes): one entry with all those lines and "split": the client text cut into one piece per line, in line order \
-- the pieces joined must give the client text exactly.
3. EXTRA LINES (the model drew more lines than the client has texts, or repeated one): fill them in this order: (a) a \
repeated row of a list gets a client text OF THE SAME ROLE that has no line yet -- never move a text of another role into it; \
(b) "client": null with "text": a short phrase taken from another client text; (c) "client": null with "text": a short phrase \
of the same kind as the rows around it (another benefit beside benefits) that is true of that business, no fact, no number, \
no promise. Steps, name / price rows, dates and contacts are never invented. Give a filled line the "role" it plays (the role \
of the row it repeats).
4. KEEP / SKIP: kind "keep" ONLY for text printed on a product, package, screen or object in the photo (its pixels stay as \
drawn). "STRAY LETTERING": "keep" when it is a logo, seal, sign or product text, or decoration that \
looks fine; "skip" (wiped) only when it is garbled junk; when unsure "keep". Poster lettering, however decorated, is never kept.
5. Client texts the model drew no line for go in "missing". Never cram them into another line.
6. A line whose "bullet dot was wiped with the letters": set "bullet": true on its entry when the dot belongs to the design \
(the same on every row of that list). Ticks, icons, numbers and pictures before a line stay on the BASE: never mention them.

STYLES -- one entry per client-text ROLE (the role is written after each T#): how that kind of text looks in the DRAFT. Every \
text of the role is set exactly alike, so describe what the role looks like, not one line: "font" (a key from the catalog \
whose family matches the draft's letterforms -- condensed, rounded, geometric, serif, script, display; a condensed or \
narrow family only when the draft letters are clearly tall and narrow, ordinary lettering gets an ordinary-width family: the \
code narrows a tight line by itself), "weight" (100-900, as heavy as the draft: a heavy headline stays 800-900), \
"italic", "case" ("upper" when the draft shows capitals, else "none"), "tracking" (em, only when clearly spaced) and "effect" \
(CSS only for outlines, shadows, glows or gradient / metallic letters, as strong as the draft: text-shadow, -webkit-text-stroke, \
background-clip:text with a linear-gradient) and "color" (#hex fill of the letters as drawn -- white letters with a black \
outline are "#ffffff" with the outline in "effect"; a role drawn in different colors on different backgrounds: the most common one).
ACCENT (optional): a word or number the draft draws bigger or in another color inside one client text ("50%" in "Giảm 50% hôm \
nay"): {"client": "T#", "text": "<exact substring>", "scale": 1.6, "color": "#hex"}.

IMAGES: DRAFT = what the model drew. SLOTS = the draft with every text line marked (red L#). BASE = the draft with the old \
letters wiped. The list after the images gives each line's machine reading of the garbled letters (often wrong, but numbers, \
dates and short words usually come out right -- use it to tell which L# is which line), measured size, color and look.

OUTPUT: one JSON object
{"assign": [{"lines": ["L1"], "client": "T0" | ["T5","T6"] | null, "split": ["..."] | null, "text": "..." | null,
             "role": "<role, for a filled line>" | null, "kind": "text" | "keep" | "skip", "bullet": false, "why": "a few words"}],
 "styles": {"<role>": {"font": "<key>", "weight": 700, "italic": false, "case": "none", "tracking": null, "color": "#hex", "effect": ""}},
 "accent": [], "missing": ["T5"], "notes": "..."}

FONT CATALOG (key: family, class): {fonts}"""


def plan(draft: np.ndarray, M: dict, prompt: str, texts: list[dict], call, product: bool = False, base: np.ndarray | None = None,
         meta: dict | None = None) -> dict:
    from . import slots
    return slots.plan(draft, M, prompt, texts, call, slots._system(SYSTEM_L), describe_l(M), base, product=product, meta=meta)
