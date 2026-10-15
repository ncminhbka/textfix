"""BƯỚC 4b: VLM ĐỌC nháp -- dòng nào là câu khách nào, dòng nào giữ / xoá, mỗi vai trông thế nào.

  A = ask(draft, M, prompt, texts, call, base=clean)                  # một lời gọi cho mọi dòng
  A2 = reask(draft, M, prompt, texts, call, ["L5", "L7"], base=clean)  # chỉ các dòng câu trả lời đầu bỏ sót
VLM không đưa toạ độ, chỉ trỏ id dòng; mọi thứ ngoài dòng chữ (vỏ, icon, hoạ tiết) là điểm ảnh của nháp. Số phận cuối cùng của
mỗi dòng: compose/fate.py. Lời dặn có ba số phận: text (chữ poster), keep (chữ trên vật trong ảnh), skip (rác đứng riêng).
"""

from __future__ import annotations

import numpy as np

from .call import _system, plan
from .parse import parse


def view(M: dict) -> dict:
    """Ô để VLM và bộ dựng làm việc: chỉ dòng chữ (plan.json cũ có thể còn S / I / shell: bỏ)."""
    L = [{k: v for k, v in l.items() if k != "shell"} for l in M["L"]]
    return {"L": L, "size": M["size"], "by": {l["id"]: l for l in L}, "geo_log": M.get("geo_log") or []}


def describe(M: dict) -> str:
    note = lambda m: (_curve_note(m) if m.get("path") else f", drawn at {m['angle']:.0f} degrees" if abs(m.get("angle") or 0) >= 2   # noqa: E731
                      else "") + (", found by the stroke detector (big decorated lettering)" if m.get("missed") and not m.get("orphan") else "") \
        + (", STRAY LETTERING (letter shapes that may be junk, a logo or decoration)" if m.get("orphan") else "") \
        + (", PRINTED ON THE CLIENT'S PRODUCT: kind \"keep\"" if m.get("printed") else "")
    # CHỮ OCR GỬI LẠI làm NEO (người dùng 14/10 tối: "đưa OCR hint quay lại nếu thấy performance giảm"; mix1 v03: thiếu chữ OCR,
    # VLM ghép ngày vào mẩu 21px và xoá dòng OCR đọc đúng "01/12 - 07/12"). Chỉ để VLM biết L# nào là dòng nào -- code không
    # dùng chữ này để viết.
    li = "\n".join(f"{m['id']}: machine reading {(m.get('ocr') or '').strip()!r}, ~{m['size']:.0f}px, color {m['color']}{note(m)}"
                   f"{_look(m)}" for m in M["L"]) or "(none)"
    return f"Text lines:\n{li}"


SYSTEM = """You are a senior graphic designer finishing a poster. An image model drew the DRAFT from the PROMPT below. Everything in it \
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
(b) "client": null with "text": a short phrase of the same kind as the rows around it (another benefit beside benefits) that \
is true of that business, no fact, no number, no promise, never words of another client text. Steps, name / price rows, dates and contacts are never invented. Give a filled line the "role" it plays (the role \
of the row it repeats).
4. EVERY LINE GETS ONE KIND. "text": poster lettering -- titles, offers, buttons, badges, stickers, list rows, captions, also an extra line on a pill or beside its peers, and ANY line that says one of the client texts wherever it is drawn (on a sign a character holds, a board, a banner, a ribbon, a badge): it gets a client text or a filler as above. "keep": only the own lettering of a depicted product or object -- brand and label on a package, logo, screen interface, seal -- even when garbled: its pixels stay as drawn; never the poster's message. "skip": only junk standing alone, neither poster lettering nor on an object: it is wiped. Unsure between "keep" and "skip": "keep".
5. Client texts the model drew no line for go in "missing". Never cram them into another line.
6. Bullets, ticks, icons, numbers and pictures before a line stay on the BASE as drawn: never mention them.

STYLES -- one entry per client-text ROLE (the role is written after each T#): how that kind of text looks in the DRAFT. Every \
text of the role is set exactly alike, so describe what the role looks like, not one line: "font" (a key from the catalog \
whose family matches the draft's letterforms -- condensed, rounded, geometric, serif, script, display; a condensed or \
narrow family only when the draft letters are clearly tall and narrow, ordinary lettering gets an ordinary-width family: the \
code narrows a tight line by itself), "weight" (100-900, as heavy as the draft: a heavy headline stays 800-900), \
"italic", "case" ("upper" when the draft shows capitals, else "none"), "tracking" (em, only when clearly spaced) and "effect" \
(CSS only for outlines, shadows, glows or gradient / metallic letters, as strong as the draft: text-shadow, -webkit-text-stroke, \
background-clip:text with a linear-gradient) and "color" (#hex fill of the letters as drawn -- white letters with a black \
outline are "#ffffff" with the outline in "effect").
ACCENT (optional): a word or number the draft draws in another color inside one client text ("50%" in "Giảm 50% hôm nay"): \
{"client": "T#", "text": "<exact substring>", "color": "#hex"}. Sizes come from the draft: never give one.

IMAGES: DRAFT = what the model drew. SLOTS = the draft with every text line marked (red L#). BASE = the draft with the old \
letters wiped. The list after the images gives each line's machine reading of the garbled letters (often wrong, but numbers, \
dates and short words usually come out right -- use it to tell which L# is which line), measured size, color and look.

OUTPUT: one JSON object
{"assign": [{"lines": ["L1"], "client": "T0" | ["T5","T6"] | null, "split": ["..."] | null, "text": "..." | null,
             "role": "<role, for a filled line>" | null, "kind": "text" | "keep" | "skip", "why": "a few words"}],
 "styles": {"<role>": {"font": "<key>", "weight": 700, "italic": false, "case": "none", "tracking": null, "color": "#hex", "effect": ""}},
 "accent": [], "missing": ["T5"], "notes": "..."}

FONT CATALOG (key: family, class): {fonts}"""


def ask(draft: np.ndarray, M: dict, prompt: str, texts: list[dict], call, product: bool = False, base: np.ndarray | None = None,
        fresh: bool = False, meta: dict | None = None) -> dict:
    """Một lời gọi cho mọi dòng -> câu trả lời đã đọc (parse). fresh: bỏ qua bộ đệm (gọi lại sau câu trả lời rỗng)."""
    return parse(plan(draft, M, prompt, texts, call, _system(SYSTEM), describe(M), base, product=product,
                                 meta=meta, fresh=fresh))


REASK = ("YOUR PREVIOUS ANSWER LEFT OUT LINES {ids}. Every other line is done. Answer with \"assign\" entries for exactly these "
         "lines -- each gets a kind (text / keep / skip) as in rule 4; \"styles\", \"accent\" and \"missing\" may stay empty.")
REFILL = ("LINES {ids} ARE POSTER LETTERING WITH NO WORDS LEFT: the client text you gave them is already set on another line. "
          "Answer with \"assign\" entries for exactly these lines: \"client\": a client text of the same role that has no line yet, "
          "else \"client\": null with \"text\": a short filler of the same kind as the lettering around it (rule 3b); \"skip\" only "
          "when nothing fits. \"styles\", \"accent\" and \"missing\" may stay empty.")


def reask(draft: np.ndarray, M: dict, prompt: str, texts: list[dict], call, lines: list[str], product: bool = False,
          base: np.ndarray | None = None, ask: str = REASK) -> dict:
    """Hỏi lại RIÊNG một số dòng (fresh): REASK -- dòng câu trả lời đầu bỏ sót (người dùng chốt 15/10: bỏ sót -> hỏi lại; vẫn
    không -> giữ); REFILL -- dòng chữ poster không còn chữ (câu VLM ghép đã có chỗ khác): xin câu cùng vai còn thiếu / chữ lấp
    trước khi gỡ trống. -> câu trả lời đã đọc, chỉ còn mục chạm các dòng đó."""
    desc = describe(M) + "\n\n" + ask.format(ids=", ".join(lines))
    A = parse(plan(draft, M, prompt, texts, call, _system(SYSTEM), desc, base, product=product, fresh=True))
    want = set(lines)
    A["assign"] = [{**u, "marks": [m for m in u["marks"] if m in want]} for u in A["assign"] if want & set(u["marks"])]
    return A


def _look(m: dict) -> str:
    """Kiểu chữ đo trên nháp: độ đậm (độ dày nét / cỡ chữ), nét dày-mảnh (chữ có chân / viết tay), viết hoa, bóng / viền."""
    out = []
    if m.get("stroke") and m.get("size"):
        r = m["stroke"] / m["size"]   # độ dày nét / em
        out.append("weight " + ("light" if r < 0.07 else "regular" if r < 0.11 else "bold" if r < 0.16 else "black")
                   + f" (stroke ~{m['stroke']:.0f}px)")
    if (m.get("stroke_contrast") or 0) >= 2.2:
        out.append("thick-thin strokes (serif / script)")
    if m.get("caps"):
        out.append("all caps")
    e = m.get("effect")
    if e:
        if abs(e.get("dx", 0)) + abs(e.get("dy", 0)) >= 1:
            out.append(f"drop shadow {e.get('color')} offset ({e['dx']:.0f},{e['dy']:.0f})px")
        else:
            out.append(f"outline / glow {e.get('color')}")
    return (", " + ", ".join(out)) if out else ""


def _curve_note(m: dict) -> str:
    """Ô chữ chạy theo đường cong: vòm lên hay võng xuống, dài bao nhiêu (để VLM viết vừa một dòng)."""
    pt = np.asarray(m["path"], np.float64)
    ln = float(np.sum(np.hypot(*np.diff(pt, axis=0).T)))
    mid, chord = pt[len(pt) // 2], (pt[0] + pt[-1]) / 2
    n = max(2, int(round(ln / max(1.0, 0.55 * (m.get("size") or 1.0)))))
    return (f", CURVED LINE: the letters run along an {'arch (bulging up)' if mid[1] < chord[1] else 'arc bulging down (a smile)'}, "
            f"room for about {n} characters")
