"""CHỈ DÒNG CHỮ -- ĐƯỜNG DUY NHẤT của nhánh main: poster = bản NHÁP với chữ viết lại cho rõ.

Người dùng chốt sau bộ accept1: (1) không bỏ dòng nào nháp đã vẽ, thiếu câu khách thì VLM viết cho đủ; (2) icon nhỏ trước mỗi
dòng -> vẽ lại; (3) mọi thứ còn lại -- vỏ, icon to, hoạ tiết, hàng sao, avatar -- là điểm ảnh của nháp. VLM chỉ nhận dòng chữ.

14/10: KHÔNG CÒN BẢN XOÁ FLUX (đường đó: nhánh git `legacy-flux-erase`). Chữ nằm ở đâu do Hi-SAM trả lời (hisam.strokes: mặt nạ
nét chữ của nháp), dưới chữ là gì do LaMa vẽ lại:

  ink = hisam.strokes(draft)                  # nét chữ (trạm vẽ)
  M   = view(slots.build(draft, ink))         # chỉ còn L: dòng OCR + cụm nét chưa dòng nào nhận ("orphan": VLM quyết giữ / xoá)
  P   = fix_ops(plan(...), M)                 # lệnh của VLM: chỉ ô L, không nền / viền riêng
  bg, mask = erase(draft, ink, M, P)          # nền dựng = nháp, LaMa lấp đúng mặt nạ nét chữ của các dòng không giữ
"""

from __future__ import annotations

import os
import re

import numpy as np

LONLY = True   # không còn cờ: main chỉ có đường này (đường cũ ô vỏ / ô chi tiết: nhánh legacy-slots-si)

INK_SAME = 25.0   # điểm cùng màu mực của dòng: cách màu mực <= 25 ΔE
INK_IN = 0.7      # mảng cùng màu mực là nét chữ khi >= 70% của nó nằm trong vùng dòng (vòng con dấu, viền nút: phần lớn ở ngoài)
GLYPH_MAX = 2.0   # ... và cạnh dài <= 2 cao dòng (một chữ cái / chữ số); dài hơn là hình khác cùng màu mực


def view(M: dict) -> dict:
    """Ô để VLM và bộ dựng làm việc: chỉ dòng chữ."""
    L = [{**l, "shell": None} for l in M["L"]]
    return {**M, "L": L, "S": [], "I": [], "shells": {}, "by": {l["id"]: l for l in L}, "lonly": True}


LOOSE = 0.4       # ghép dòng bị bỏ với câu khách còn thừa: chữ nháp giống >= 0.4
COVER_MIN = 0.6   # VLM trả lệnh cho < 60% số dòng = câu trả lời bị cụt (mix1 p02: 1 lệnh / 10 dòng) -> gọi lại một lần


def covered(P: dict, M: dict) -> set:
    """Các dòng đã có lệnh VIẾT (có chữ) hoặc GIỮ."""
    return {m for op in (P or {}).get("ops") or [] if op.get("kind") == "keep" or (op.get("kind") == "text" and op.get("html"))
            for m in op.get("marks") or op.get("slots") or [] if m in M["by"]}


ICON_LEAD = re.compile(r"^(\s*(?:<(?!i-icon|svg)[^>]+>\s*)*)(?:<i-icon\b[^>]*>(?:\s*</i-icon>)?|<svg\b.*?</svg>)\s*", re.I | re.S)
DOT_LEAD = re.compile(r"^(\s*(?:<(?!i-icon|svg)[^>]+>\s*)*)[•●○▪■◆✓✔☑★→·]\s*")
BR = re.compile(r"<br\s*/?>", re.I)


def marks_rule(P: dict, M: dict) -> tuple[dict, list[str]]:
    """TEXTFIX CHỈ SỬA CHỮ (người dùng chốt 14/10): icon / tick / hình nhỏ trước mỗi dòng là điểm ảnh của nháp, không gỡ, không vẽ
    lại -> lệnh chữ không được mở đầu bằng icon (gỡ, nếu không sẽ có hai dấu cạnh nhau). CHẤM TRÒN đầu dòng thì khác: OCR gộp nó
    vào khung dòng và Hi-SAM coi nó là nét chữ (đo accept2 r04: 3 / 3), tức nó bị gỡ cùng chữ -- nó là một KÝ TỰ của dòng. Dòng
    mã đo thấy có chấm (slots._lead_dots: l["dot"]) thì dòng chữ mới mở đầu bằng "• "; dòng không có thì không.
    Lệnh nhiều dòng: xét từng dòng khi số dòng html khớp số ô, không thì chỉ dòng đầu."""
    log, ops = [], []
    for op in P.get("ops") or []:
        mk = [m for m in op.get("marks") or [] if m in M["by"]]
        if op.get("kind") != "text" or not op.get("html") or not mk:
            ops.append(op)
            continue
        Ls = sorted((M["by"][m] for m in mk), key=lambda l: (l["box"][1], l["box"][0]))
        parts = BR.split(op["html"])
        rows = list(zip(parts, Ls)) if len(parts) == len(Ls) else [(parts[0], Ls[0])]
        out, did = [], []
        for h, l in rows:
            h2 = ICON_LEAD.sub(lambda m: m.group(1), h)
            if h2 != h:
                did.append(f"{l['id']} bỏ icon đầu dòng")
            had = bool(DOT_LEAD.match(h2))
            if had and not l.get("dot"):
                h2 = DOT_LEAD.sub(lambda m: m.group(1), h2)
                did.append(f"{l['id']} bỏ chấm đầu dòng (nháp không có)")
            elif l.get("dot") and not had:
                h2 = "• " + h2.lstrip()
                did.append(f"{l['id']} thêm chấm đầu dòng (nháp có, đã gỡ cùng chữ)")
            out.append(h2)
        if did:
            log.append(f"{op.get('id')}: " + "; ".join(did))
            op = {**op, "html": "<br>".join(out + parts[len(out):])}
        ops.append(op)
    return {**P, "ops": ops}, log


STACK_GAP = 1.0   # hai dòng xếp chồng: khe dọc <= 1 cao chữ và có phần chồng ngang
ROW_GAP = 2.0     # hai mẩu của một dòng: cùng hàng (chồng dọc >= nửa cao), khe ngang <= 2 cao chữ


def split_far(P: dict, M: dict) -> tuple[dict, list[str]]:
    """LỆNH GỘP CÁC DÒNG KHÔNG XẾP CHỒNG (ca người dùng gửi 14/10: VLM gộp năm dòng của bốn thẻ vào một lệnh -> khung dựng là
    hình bao cả bốn thẻ, chữ trôi khỏi thẻ). Các ô của một lệnh chia thành cụm liền nhau (xếp chồng sát / cùng hàng); hơn một
    cụm thì tách thành mỗi cụm một lệnh, chữ chia theo <br>: số dòng html = số ô -> mỗi ô một dòng; = số cụm -> mỗi cụm một
    dòng; không khớp -> phát lần lượt, cụm hết chữ thành dòng bị bỏ (fill_gaps lo). 63 lệnh gộp của accept2 + mix1 (đo 14/10)
    đều là dòng xếp chồng khe <= 0,6: luật này không đụng lệnh nào ở đó."""
    log, ops = [], []
    for op in P.get("ops") or []:
        mk = [m for m in op.get("marks") or [] if m in M["by"]]
        if op.get("kind") != "text" or not op.get("html") or len(mk) < 2 or any(M["by"][m].get("path") for m in mk):
            ops.append(op)
            continue
        Ls = sorted((M["by"][m] for m in mk), key=lambda l: (l["box"][1], l["box"][0]))
        par = list(range(len(Ls)))

        def root(i):
            while par[i] != i:
                i = par[i]
            return i
        for i, a in enumerate(Ls):
            for j in range(i + 1, len(Ls)):
                b = Ls[j]
                A, B = a["box"], b["box"]
                sz = max(float(a.get("size") or A[3] - A[1]), float(b.get("size") or B[3] - B[1]))
                vgap, hov = max(B[1] - A[3], A[1] - B[3]), min(A[2], B[2]) - max(A[0], B[0])
                vov, hgap = min(A[3], B[3]) - max(A[1], B[1]), max(B[0] - A[2], A[0] - B[2])
                if (vgap <= STACK_GAP * sz and hov > 0) or (vov >= 0.5 * min(A[3] - A[1], B[3] - B[1]) and hgap <= ROW_GAP * sz):
                    par[root(i)] = root(j)
        groups: dict = {}
        for i, l in enumerate(Ls):
            groups.setdefault(root(i), []).append(l)
        G = sorted(groups.values(), key=lambda g: (min(l["box"][1] for l in g), min(l["box"][0] for l in g)))
        if len(G) < 2:
            ops.append(op)
            continue
        parts = [x for x in BR.split(op["html"])]
        take = [len(g) for g in G] if len(parts) == len(Ls) else [1] * len(G) if len(parts) == len(G) else None
        k, made = 0, []
        for n, g in enumerate(G):
            cnt = take[n] if take else min(len(g), max(0, len(parts) - k))
            html = "<br>".join(parts[k:k + cnt])
            k += cnt
            if not re.sub(r"<[^>]+>", "", html).strip():
                continue
            ids = [l["id"] for l in g]
            # không tự khai nhóm cho các lệnh vừa tách: nhóm "list" kéo theo luật dồn hàng / xếp lại thứ tự của validate
            made.append({**op, "id": f"{op.get('id')}_{n + 1}", "marks": ids, "slots": ids, "html": html})
        ops += made
        log.append(f"{op.get('id')}: lệnh gộp {len(Ls)} dòng không xếp chồng ({[[l['id'] for l in g] for g in G]}) -> tách {len(made)} lệnh")
    return {**P, "ops": ops}, log


def fill_gaps(P: dict, M: dict, texts: list[dict], product: bool = False) -> tuple[dict, list[str]]:
    """KHÔNG ĐỂ TRỐNG DÒNG NÀO nháp đã vẽ (13/10, người dùng: "cố gắng bịa vào, đừng bỏ trống" -- mix1: mỗi dòng bị bỏ là một
    nút / huy hiệu trống). Dòng chưa có lệnh viết / giữ (VLM bỏ sót, VLM ghi skip, luật chặn đổi thành skip):
      - chữ in trên sản phẩm khách tải -> GIỮ (mix1 p02: VLM bỏ sót, nhãn trên gói bị xoá);
      - chữ nháp ở đó gần trùng một câu khách -> viết câu khách đó;
      - không thì GIỮ nguyên chữ nháp của dòng (có thể sai chính tả, nhưng vỏ không trống)."""
    import html as _html
    from . import slots
    have, log, ops = covered(P, M), [], []
    # CỤM NÉT CHƯA DÒNG NÀO NHẬN ("orphan"): VLM quyết (người dùng chốt 14/10) -- nó ghi skip thì xoá, không nhắc tới thì GIỮ
    wiped = {m for op in P.get("ops") or [] if op.get("kind") == "skip" for m in op.get("marks") or op.get("slots") or []}
    gaps = [l for l in M["L"] if l["id"] not in have and not (l.get("orphan") and l["id"] in wiped)]
    if not gaps:
        return P, log
    gone = {l["id"] for l in gaps}
    # câu khách CHƯA lệnh nào viết: ghép một - một với các dòng bị bỏ theo độ giống chữ nháp, ngưỡng thấp (LOOSE) -- chữ nháp méo
    # nặng ("Tio shard" = "Tiêu chuẩn") không qua được ngưỡng chung, nhưng trong số câu còn thừa thì nó vẫn là câu giống nhất
    from difflib import SequenceMatcher
    used = {slots._tid(op.get("client")) for op in P.get("ops") or [] if op.get("kind") == "text" and op.get("html")}
    wrote = " ".join(slots._flat(re.sub(r"<[^>]+>", " ", op.get("html") or "")) for op in P.get("ops") or [] if op.get("kind") == "text")
    free = [k for k, t in enumerate(texts) if k not in used and slots._flat(t["text"]) not in wrote]
    pairs = sorted(((SequenceMatcher(None, slots._flat(l.get("ocr") or ""), slots._flat(texts[k]["text"])).ratio(), l["id"], k)
                    for l in gaps if not l.get("printed") and not l.get("orphan") and len(slots._flat(l.get("ocr") or "")) >= 3 for k in free), reverse=True)
    loose = {}
    for sim, lid, k in pairs:
        if sim >= LOOSE and lid not in loose and k not in loose.values():
            loose[lid] = k
    for op in P.get("ops") or []:   # gỡ các dòng này khỏi lệnh skip cũ
        mk = [m for m in op.get("marks") or op.get("slots") or [] if m not in gone]
        if op.get("kind") == "skip" and not mk:
            continue
        ops.append({**op, "marks": mk, "slots": mk} if op.get("kind") == "skip" else op)
    for l in gaps:
        if l.get("orphan"):
            ops.append({"id": f"fill_{l['id']}", "kind": "keep", "marks": [l["id"]], "slots": [l["id"]], "why": "fill: nét OCR không đọc, VLM không nhắc"})
            log.append(f"{l['id']}: cụm nét OCR không đọc ({l.get('ocr')!r}), VLM không nhắc tới -> giữ nguyên")
            continue
        k = None if l.get("printed") else loose.get(l["id"], slots._client_like(l.get("ocr") or "", texts))
        if k is not None:
            ops.append({"id": f"fill_{l['id']}", "kind": "text", "marks": [l["id"]], "slots": [l["id"]], "client": f"T{k}",
                        "html": _html.escape(texts[k]["text"]), "box_style": "", "why": "fill: dòng bị bỏ, chữ nháp giống câu khách"})
            log.append(f"{l['id']}: dòng bị bỏ ({l.get('ocr')!r}) -> viết câu khách T{k} {texts[k]['text']!r}")
        elif l.get("printed"):
            ops.append({"id": f"fill_{l['id']}", "kind": "keep", "marks": [l["id"]], "slots": [l["id"]], "why": "fill: chữ in trên sản phẩm"})
            log.append(f"{l['id']}: dòng bị bỏ ({l.get('ocr')!r}) -> giữ nguyên (chữ in trên sản phẩm)")
        elif product:
            # KHÁCH TẢI ẢNH SẢN PHẨM (accept2 q03, 14/10: logo Coca-Cola trên lon bị OCR đọc thành dòng "eo.mg", VLM không nhắc tới,
            # bộ đọc ảnh sản phẩm không khớp được -> dòng bị xoá trống = mất logo). Chữ trên sản phẩm PHẢI GIỮ: ở poster có ảnh
            # sản phẩm, dòng không ai viết gì thì giữ nguyên điểm ảnh nháp -- sai về phía này chỉ còn lại một mẩu chữ giun
            ops.append({"id": f"fill_{l['id']}", "kind": "keep", "marks": [l["id"]], "slots": [l["id"]], "why": "fill: poster có ảnh sản phẩm, không ai viết dòng này"})
            log.append(f"{l['id']}: dòng bị bỏ ({l.get('ocr')!r}) -> giữ nguyên (poster có ảnh sản phẩm: có thể là chữ trên sản phẩm)")
        else:   # không còn câu nào để viết (dòng nối của một câu đã viết ở dòng trên): XOÁ TRỐNG -- người dùng chốt 14/10, chữ
            # nháp giữ lại là chữ giun (mix1 "SoundBean Air 3")
            ops.append({"id": f"fill_{l['id']}", "kind": "skip", "marks": [l["id"]], "slots": [l["id"]], "html": "",
                        "why": "fill: không còn câu nào để viết"})
            log.append(f"{l['id']}: dòng bị bỏ ({l.get('ocr')!r}) -> xoá trống (không còn câu nào để viết)")
    return {**P, "ops": ops}, log


def describe_l(M: dict) -> str:
    from . import slots
    note = lambda m: (slots._curve_note(m) if m.get("path") else f", drawn at {m['angle']:.0f} degrees" if abs(m.get("angle") or 0) >= 2   # noqa: E731
                      else "") + (", OCR missed this line: the letters read here are a guess" if m.get("missed") and not m.get("orphan") else "") \
        + (", starts with a bullet dot that was wiped with the letters: begin with \"• \"" if m.get("dot") else "") \
        + (", STRAY LETTERING the reader could not read" if m.get("orphan") else "") \
        + (", PRINTED ON THE CLIENT'S PRODUCT: kind \"keep\"" if m.get("printed") else "")
    li = "\n".join(f"{m['id']}: \"{m['ocr']}\" (text ~{m['size']:.0f}px, ~{len(m['ocr'])} characters, color {m['color']}"
                   f"{note(m)}{slots._look(m)})" for m in M["L"]) or "(none)"
    return f"Text lines:\n{li}"


SYSTEM_L = """You are a senior graphic designer finishing a poster. An image model drew the DRAFT from the PROMPT below. Everything in it \
is good except its lettering, which is garbled and misspelt. The renderer keeps the draft exactly as it is -- pictures, cards, \
pills, badges, big icons, ornaments -- and only replaces the lettering: code measured every TEXT LINE of the draft as a slot \
L#, wiped the old letters (the BASE image shows the result) and now draws what you write exactly inside each slot. The final \
poster is the draft with clean, correct text.

THE RULES THAT MATTER MOST
1. The LOOK comes from the draft, the WORDS come from the client texts. Never take spelling from the image or the OCR \
(the model draws TRƯỞNG for TRƯƠNG); copy client words character for character with every Vietnamese diacritic, and never \
swap a word for a synonym (nghìn stays "nghìn").
2. EVERY LINE THE MODEL DREW IS WRITTEN. Never leave an L# out and never skip one (only "STRAY LETTERING" may be skipped): an empty spot next to its icon or inside \
its card looks broken. When the draft has more lines than the client has texts (the model repeated a row, or drew an extra \
one), fill the extra lines in this order: (a) a client text that has no line yet; (b) a short phrase taken from another \
client text (from the description: "Cơm chín đều"); (c) a short natural phrase of your own that fits that spot and that \
business, carrying no fact and no promise. Steps of a guide, name / price rows, dates and contacts are never invented: there \
an extra line repeats nothing and gets (a) or (b) only.
3. FIT. The room is what the model drew, never what the client wrote. Each line lists how many characters the model wrote \
there: that is the most the line holds (real letters are wider than the model's thin ones); fewer is fine. A client text \
longer than its room is REWRITTEN SHORT in the client's own words until it fits at the drawn size: keep the key words, drop \
the rest. Example: "Hỗ trợ nấu cơm chín đều, giữ độ mềm dẻo và giúp việc chuẩn bị bữa ăn hằng ngày trở nên thuận tiện" drawn \
as 2 lines of about 22 characters -> "Cơm chín đều, mềm dẻo<br>mỗi bữa ăn". Never the full sentence at a smaller size, never \
more lines than the model drew.
4. FACTS -- names, prices, numbers, dates, phones, addresses, emails, links -- are never changed, abbreviated or invented: \
exact, with their unit (35.000đ, 30 phút, 50%, 15-25 triệu; "Quận 3", never "Q.3"), or left out.
5. Each client text is written ONCE, in the line that fits it best (the biggest one for a headline). Where the model repeated \
it, the copy gets other words by rule 2.
6. The rows of one list look exactly alike and keep the client's order. Client items the model drew no row for are listed \
in "missing", never crammed into another row.

IMAGES: the message names each one. DRAFT = what the model drew. SLOTS = the draft with every text line marked (red L#). \
BASE = the draft with the old letters wiped: what will really be behind each line. Cards, pills, buttons and icons are already \
there: never draw a background, border, box or shadow box of your own, and never write a big icon -- only text.

TEXT LINES, as listed after the images: the garbled OCR letters, the drawn text size in px, the number of characters drawn \
(its room), the color. L# numbers follow the reading order block by block: a column is numbered top to bottom before the \
column beside it starts.

ONE OP PER LINE (or per group of stacked lines)
- kind "text", slots ["L5"], html = the text. One op may take several CONSECUTIVE lines of ONE block stacked one under the \
other (["L5","L6"], <br> between the lines), never lines standing side by side (only two pieces of one drawn line that OCR \
cut apart go together). Every slot id appears in exactly ONE op.
- kind "keep" ONLY for text printed on a product, package, screen, dial or object in the photo: its pixels stay as drawn. \
Poster lettering, however decorated (on a ribbon, 3D, metallic, in a badge), is always rewritten as text.
- CURVED LINE (the list says so; SLOTS shows its curve): ONE text op with that single slot, one line, no <br>: the renderer \
lays your letters along the measured curve.
- A line "drawn at N degrees" is tilted and the renderer rotates it itself: never write transform rotate / skew. Tilted \
lines stacked in one sticker / ribbon are ONE op.
- "OCR missed this line": real lettering of the draft (usually a big decorated headline). Look at the DRAFT to see which \
client text it is and write it there.
- ICONS ARE NOT YOURS. Every icon, check mark, tick, leaf, arrow or little picture in front of a line is still on the BASE, \
exactly as the model drew it: never write an icon, never an <i-icon>, never an svg. The one exception is a plain BULLET DOT: \
when the list says a line "starts with a bullet dot", the dot was wiped with the letters -- start that line with "• ". No \
other line starts with a dot.
- "STRAY LETTERING the reader could not read": letter shapes the stroke detector found where OCR read nothing (the listed \
letters may be empty). Look at the DRAFT and decide: kind "keep" when it is lettering on a product, logo, seal, sign or \
object in the picture, or decoration that looks fine; kind "text" when it is clearly one of the client texts drawn there; \
kind "skip" (it is wiped, the spot stays empty) only when it is garbled junk that would look wrong on the poster. When \
unsure: "keep".
- Step numbers, big icons and pictures beside a line are still on the BASE: never repeat them in the text ("1.", "Bước 2").

TEXT
- Write each line at the size measured for its slot, with the model's line breaks; fine print and footers too. Set \
"client": "T<i>" on every op writing (part of) a client text, and null on a line you filled by rule 2 (b) or (c).
- One client text drawn over several stacked lines (MENU / TRÀ SỮA) is one op with all those slots. Read the garbled OCR to \
find them: a short line under a line that ends its sentence is that text's second line.
- Client texts marked OURS are filler we wrote (headline, tagline, button, section label): shorten or reword them freely.
- Star ratings are pictures and stay on the BASE: never write "5 sao" / "5.0" or star characters.
- html: inline spans only (color / weight / italic / size for emphasis), inline CSS, no <img>, URLs or scripts; sizes in poster \
px. The slot is a flex box centered both ways; left-aligned text needs box_style "justify-content:flex-start; text-align:left".

GROUPS -- you declare, the renderer obeys. Ops that belong together carry the same "group" name ("benefits", "menu-names", \
"menu-prices", "badges", "headline"), the same "group_kind" and "align":
  "list" = separate items, one per row: drawn on ONE margin at ONE size, font, weight and color.
  "flow" = ONE sentence the draft broke over several lines that you did not merge into one op: all lines drawn alike.
  "align" = "left" | "center" | "right", as the draft aligned them.
  Write the ops of a group identically yourself (same font key, weight, size, color, case, letter-spacing; only the words \
change). Every list and every multi-line sentence that is not one merged op MUST be a group. Lines the draft drew at clearly \
different sizes (a small "GIẢM" beside a huge "50%", a big headline over a small tagline) are NOT a group: "group": null.

STYLE -- the poster should look like the draft with clean text, not like another design
- One type system: style.display_font for headlines, style.text_font for the rest (keys from the catalog); <i-font key="..." \
weight="700">...</i-font> only where a line needs another font. Pick fonts whose family matches the draft's letterforms \
(condensed, rounded, geometric, serif, script, display) and whose letter width is within about 0.05em of the width listed for \
the line (caps width for all-caps lines): a wider font gets shrunk, a narrower one looks lost.
- Match each line's weight, case, italic and tracking: a heavy headline stays heavy (800-900), a script stays a script, \
slanted stays italic, a wide headline is never set condensed. The measured weight / strokes / shadow are hints; the image \
wins. Effects with inline CSS, as strong as the draft: -webkit-text-stroke or text-shadow for outlines, text-shadow for \
shadows and glows, background-clip:text with a linear-gradient for gradient or metallic letters.
- Colors: put the draft's 3-5 main colors in style.palette and take every text color from it or from the measured colors; \
no new hues. Same role, same color and font. Every text must read clearly on what is behind it on the BASE; on a busy \
background add a subtle text-shadow.
- Hierarchy and alignment as the draft shows them; a centered block stays centered, a left column stays left.

OUTPUT: one JSON object
{"style": {"display_font": "<key>", "text_font": "<key>", "palette": ["#hex"], "notes": "..."},
 "ops": [{"id": "o1", "slots": ["L1"], "kind": "text|keep|skip", "client": "T0" | null, "html": "...", "box_style": "...",
          "group": "name" | null, "group_kind": "list" | "flow" | null, "align": "left" | "center" | "right" | null,
          "why": "a few words"}],
 "missing": ["T5"], "notes": "..."}

FONT CATALOG (key: family, class): {fonts}"""


def plan(draft: np.ndarray, M: dict, prompt: str, texts: list[dict], call, product: bool = False, base: np.ndarray | None = None,
         meta: dict | None = None) -> dict:
    from . import slots
    return slots.plan(draft, M, prompt, texts, call, product=product, base=base, meta=meta, system=slots._system(SYSTEM_L),
                      desc=describe_l(M), base_label="BASE (the draft with the old letters wiped; your text is drawn on it):")


BOX = re.compile(r"(?:^|;)\s*(?:background(?:-color|-image)?|border(?:-(?:top|right|bottom|left|radius|width|color|style))?|"
                 r"box-shadow|padding(?:-(?:top|right|bottom|left))?|outline)\s*:[^;]*", re.I)


def fix_ops(P: dict, M: dict) -> tuple[dict, list[str]]:
    """Lệnh của VLM trong chế độ chỉ dòng chữ: bỏ ô không phải L; lệnh chữ không có nền / viền / đệm riêng (vỏ đã nằm trên nền,
    vẽ thêm là vỏ lồng vỏ) -- trừ dải màu tô CHỮ (background-clip:text)."""
    log, ops = [], []
    for op in P.get("ops") or []:
        mk = [m for m in op.get("marks") or [] if m in M["by"]]
        if not mk:
            continue
        op = {**op, "marks": mk, "slots": mk}
        bs = op.get("box_style") or ""
        if op.get("kind") == "text" and bs and not re.search(r"background-clip\s*:\s*text", bs, re.I):
            bs2 = BOX.sub("", bs).strip("; ")
            if bs2 != bs.strip("; "):
                log.append(f"{op.get('id')}: bỏ nền / viền riêng của lệnh chữ (vỏ là điểm ảnh nháp)")
                op["box_style"] = bs2
        if op.get("kind") not in ("text", "keep", "skip"):
            log.append(f"{op.get('id')}: kind {op.get('kind')!r} trên ô dòng chữ -> skip")
            op = {**op, "kind": "skip", "html": ""}
        if op.get("kind") == "skip":
            log.append(f"{op.get('id')}: VLM bỏ dòng {mk} -- chữ cũ được xoá, chỗ đó để trống")
        ops.append(op)
    return {**P, "ops": ops}, log


def _region(l: dict, H: int, W: int, grow: float) -> np.ndarray:
    """Vùng của một dòng chữ (bool cỡ ảnh): đa giác OCR (dòng cong: dải dọc đường chân) nới `grow` cao dòng."""
    import cv2
    m = np.zeros((H, W), np.uint8)
    if l.get("path"):
        sz = float(l.get("size") or 1.0)
        for (ax, ay), (bx, by) in zip(l["path"][:-1], l["path"][1:]):
            ln = float(np.hypot(bx - ax, by - ay)) or 1.0
            nx, ny = (by - ay) / ln, -(bx - ax) / ln
            cv2.fillConvexPoly(m, np.int32([[ax - nx * 0.3 * sz, ay - ny * 0.3 * sz], [bx - nx * 0.3 * sz, by - ny * 0.3 * sz],
                                            [bx + nx * 1.2 * sz, by + ny * 1.2 * sz], [ax + nx * 1.2 * sz, ay + ny * 1.2 * sz]]), 1)
        h = sz
    else:
        b = l["box"]
        poly = l.get("poly") or [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        cv2.fillPoly(m, [np.asarray(poly, np.int32)], 1)
        h = float(min(b[3] - b[1], b[2] - b[0])) if abs(l.get("angle") or 0) < 2 else float(l.get("size") or (b[3] - b[1]))
    k = max(1, int(round(grow * h)))
    # dấu thanh tiếng Việt nhô lên khỏi khung OCR: nới dọc nhiều hơn ngang
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * int(1.6 * k) + 1))).astype(bool)


LAMA_PATH = os.path.expanduser(os.environ.get("TEXTFIX_LAMA", "~/persistent-data/lama/big-lama.pt"))
MASK_GROW = float(os.environ.get("TEXTFIX_MASK_GROW", "0.12"))   # nới mặt nạ quanh nét, theo cao dòng (viền khử răng cưa: còn sót
                                                                 # viền là LaMa vẽ lại hình chữ -- đo 13/10, nới 0,05 chưa đủ)
WHOLE = 0.5         # mảng nét Hi-SAM chạm lõi dòng thuộc trọn về dòng khi >= nửa diện tích nằm trong vùng dòng
WEAK = 0.08         # Hi-SAM phủ < 8% lõi dòng = thấy quá ít (chữ rất nhỏ: accept2 e02 chân trang 6%; dòng thường 21-30%) -> thêm màu mực
HALO_REACH = 0.35   # QUẦNG (viền dày, bóng đổ, quầng sáng): tìm trong dải sát nét, rộng 0,35 cao dòng ...
HALO_DE = 14.0      # ... điểm trong dải khác NỀN CỦA DÒNG > 14 ΔE là quầng. Nền của dòng ước từ vành ngay ngoài dải (vẫn trong
                    # vùng dòng): chữ nằm trong nút / thẻ thì vành đó là lòng nút, nên lòng nút không bị coi là quầng
HALO_LO = 7.0       # nền của dòng PHẲNG (vành lệch trung vị <= HALO_FLAT): quầng lan tiếp qua các điểm > 7 ΔE liền với nó, ra tới
HALO_FLAT = 8.0     # mép vùng dòng -- đuôi bóng mờ dần (accept2 e04 chữ 3D đỏ: đuôi < 14 lọt ngoài mặt nạ thì LaMa kéo bóng vào,
                    # hiện lại hình chữ mờ). Nền có vân / ảnh chụp: không lan (mọi thứ đều "khác nền"). Ngưỡng 4 thì nền giấy
                    # của chính e04 lúc qua lúc không (nén JPG khác nhau); 8: 65 nháp mặt nạ 14,9 -> 15,4% ảnh, 14 poster đổi > 1 điểm
_LAMA: dict = {}


def lama_ready() -> bool:
    return os.path.exists(LAMA_PATH)


def _lama(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """LaMa (big-lama.pt, TorchScript): vẽ lại phần trong mặt nạ theo nền xung quanh. Một lượt, không khuếch tán."""
    import torch
    if "net" not in _LAMA:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _LAMA["net"], _LAMA["dev"] = torch.jit.load(LAMA_PATH, map_location=dev).eval(), dev
    H, W = img.shape[:2]
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    im = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    mk = np.pad(mask.astype(np.float32), ((0, ph), (0, pw)))
    with torch.no_grad():
        o = _LAMA["net"](torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255).to(_LAMA["dev"]),
                         torch.from_numpy(mk)[None, None].to(_LAMA["dev"]))
    return (o[0].permute(1, 2, 0).float().cpu().numpy()[:H, :W] * 255).round().clip(0, 255).astype(np.uint8)


def erase_mask(draft: np.ndarray, ink: np.ndarray, M: dict, kept: set, log: list | None = None) -> np.ndarray:
    """MẶT NẠ GỠ CHỮ của mọi dòng không giữ -- thứ duy nhất quyết định chất lượng gỡ chữ (LaMa sạch khi mặt nạ ôm sát nét, hỏng
    khi mặt nạ trùm cả nút). ink = nét chữ của Hi-SAM. Mỗi dòng, cùng một cách:
      (1) NÉT: điểm nét nằm trong vùng dòng + trọn các mảng nét chạm lõi dòng mà quá nửa nằm trong vùng dòng (đuôi chữ, dấu
          thanh thò khỏi khung OCR). Hi-SAM đã trả lời "đây có phải chữ không" nên vòng con dấu, viền nút, cầu vồng dưới tiêu đề
          không còn bị kéo vào (các luật màu cũ kéo nhầm).
      (2) Hi-SAM thấy quá ít ở dòng đó (chữ rất nhỏ): thêm các mảng cùng MÀU MỰC cỡ một chữ cái nằm trong vùng dòng; vẫn
          không có gì: lõi dòng.
      (3) QUẦNG: Hi-SAM chỉ thấy thân nét. Trong dải sát nét, điểm khác nền của dòng là viền dày / bóng đổ / quầng sáng -> gộp
          vào; nền của dòng phẳng thì lan tiếp theo đuôi mờ của quầng. Dấu câu / gạch nối nằm gọn trong lõi dòng cũng gỡ.
          ĐÃ THỬ VÀ BỎ (14/10): nền ước từ vành NGOÀI vùng dòng + lan mọi thứ "khác nền liền với nét" -- lòng nút, vòng con dấu,
          lá cây dưới chữ đều thành "quầng" (mix1 f02 nút đỏ bị lau, c01 mất cả con dấu, mặt nạ 15 -> 30% ảnh).
    Nét của dòng GIỮ (chữ in trên vật) không bao giờ vào mặt nạ. Icon / dấu đầu dòng không phải nét chữ: để nguyên."""
    import cv2
    from . import slots
    H, W = draft.shape[:2]
    A = slots._lab(draft)
    ink = np.asarray(ink, bool)
    n_cc, cc, st, _c = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    area = st[:, cv2.CC_STAT_AREA]
    mask, hold = np.zeros((H, W), bool), np.zeros((H, W), bool)
    regs = {l["id"]: _region(l, H, W, 0.3) for l in M["L"]}
    count = np.zeros((H, W), np.uint8)
    for l in M["L"]:
        count += regs[l["id"]]
        if l["id"] in kept:
            hold |= regs[l["id"]]
    n = {"nét": 0, "màu mực": 0, "lõi dòng": 0}
    n_halo = 0
    for l in M["L"]:
        if l["id"] in kept:
            continue
        R2 = _region(l, H, W, 0.6)
        ys, xs = np.nonzero(R2)
        if not len(ys):
            continue
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        r2, r, r0 = R2[y0:y1, x0:x1], regs[l["id"]][y0:y1, x0:x1], _region(l, H, W, 0.1)[y0:y1, x0:x1]
        hgt = float(l.get("size") or (l["box"][3] - l["box"][1]))
        ik, lab, c_ = ink[y0:y1, x0:x1], A[y0:y1, x0:x1], cc[y0:y1, x0:x1]
        # CỠ DÒNG KHÔNG VƯỢT 2,2 LẦN CAO NÉT (accept2 q02: khung OCR của một dòng trùm hai hàng, "cỡ" 127 px cho chữ cao 20 px
        # -> dải quầng rộng 44 px nuốt icon của cả ba hàng).
        ids0 = [i for i in np.unique(c_[ik & regs[l["id"]][y0:y1, x0:x1]]) if i > 0 and st[i, cv2.CC_STAT_HEIGHT] >= 4]
        if len(ids0) >= 3:   # mảng CAO (phân vị 90: thân chữ hoa / nét lên), không phải trung vị -- dấu thanh rời kéo trung vị xuống,
            # tiêu đề chữ hoa có dấu bị ép cỡ nhỏ lại, dải quầng hẹp đi, bóng chữ lộ lại (accept2 e04)
            hgt = min(hgt, 2.2 * float(np.percentile(st[ids0, cv2.CC_STAT_HEIGHT], 90)))
        got = ik & r
        inside = np.bincount(c_[r], minlength=n_cc)
        whole = [i for i in np.unique(c_[ik & r0]) if i > 0 and inside[i] >= WHOLE * area[i]]
        if whole:
            got |= np.isin(c_, whole) & r2
        src = "nét"
        core = max(1.0, float(r0.sum()))
        if float((got & r0).sum()) < WEAK * core:
            if l.get("color"):
                cand = r2 & (np.linalg.norm(lab - slots._hex_lab(l["color"]), axis=2) <= INK_SAME)
                k, c2, s2, _x = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
                touch, ins = np.bincount(c2[r0], minlength=k), np.bincount(c2[r], minlength=k)
                ok = [i for i in range(1, k) if touch[i] and ins[i] >= INK_IN * s2[i, cv2.CC_STAT_AREA]
                      and max(s2[i, cv2.CC_STAT_WIDTH], s2[i, cv2.CC_STAT_HEIGHT]) <= GLYPH_MAX * hgt]
                if ok:
                    got |= np.isin(c2, ok)
                    src = "màu mực"
            if float((got & r0).sum()) < WEAK * core:
                got |= r0
                src = "lõi dòng"
        n[src] += 1
        got &= ~hold[y0:y1, x0:x1]
        # (3) QUẦNG: nền của dòng = trung bình có trọng số của vành ngay ngoài dải sát nét (trong vùng dòng), trừ mọi nét chữ
        k = 2 * max(3, int(round(MASK_GROW * hgt))) + 1
        grown = cv2.dilate(got.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool) & r2
        kh = 2 * max(4, int(round(HALO_REACH * hgt))) + 1
        near = cv2.dilate(got.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kh, kh))).astype(bool) & r2
        ring = (r2 & ~near & ~ik).astype(np.float32)
        halo = np.zeros(got.shape, bool)
        if float(ring.sum()) > 50:
            sg = max(6.0, 0.6 * hgt)
            den = cv2.GaussianBlur(ring, (0, 0), sg)
            bg = cv2.GaussianBlur(lab.astype(np.float32) * ring[..., None], (0, 0), sg) / np.maximum(den[..., None], 1e-3)
            d = np.where(den > 0.02, np.linalg.norm(lab - bg, axis=2), 0.0)
            halo = near & ~got & (d > HALO_DE)
            halo = cv2.morphologyEx(halo.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
            # QUẦNG DÍNH VÀO NÉT: viền, bóng, quầng sáng luôn chạm chữ; icon đứng trước dòng cách chữ một khe nền (accept2 q02: ba
            # chiếc lá cách chữ ~8 px lọt vào dải sát nét, bị gặm mất nửa). Chỉ giữ mảng quầng chạm phần nét đã nới.
            _k, c5 = cv2.connectedComponents((halo | grown).astype(np.uint8), connectivity=8)
            touch = np.unique(c5[grown])
            halo &= np.isin(c5, touch[touch > 0])
            if halo.any() and float(np.median(d[ring > 0])) <= HALO_FLAT:   # nền phẳng: lan theo đuôi mờ của quầng
                # chỉ lan qua điểm NHẠT (7-14): đuôi bóng. Điểm đậm ngoài dải sát nét là hình khác (lòng nút nhỏ, hạt cà phê giữa con
                # dấu, cầu vồng dưới tiêu đề -- thử lan cả điểm đậm 14/10: mix1 f02 nút đỏ bị lau, c01 hạt cà phê nhoè, x04 mất đầu cầu vồng)
                lo = ((d > HALO_LO) & (d <= HALO_DE) & r2 & ~got) | halo
                _k, c3 = cv2.connectedComponents(lo.astype(np.uint8), connectivity=8)
                keep = np.unique(c3[halo])
                halo = np.isin(c3, keep[keep > 0])
            # dấu câu / gạch nối nằm GỌN trong lõi dòng mà Hi-SAM không coi là nét ("300 Mbps - 245.000đ": gạch còn lại thành "~")
            # CHỈ ở giữa nét đầu và nét cuối của dòng: thứ đứng TRƯỚC chữ là icon đầu dòng của nháp (accept2 q02: khung OCR ôm cả
            # chiếc lá trước dòng -> lá bị gỡ như "dấu câu", còn lại một mẩu vụn)
            n4, c4, s4, _x = cv2.connectedComponentsWithStats(((d > HALO_DE) & r0 & ~grown).astype(np.uint8), 8)
            rim = r0 & (cv2.erode(r0.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
            edge = np.bincount(c4[rim], minlength=n4)
            gx = np.nonzero(got.any(axis=0))[0]
            lo_x, hi_x = (int(gx.min()), int(gx.max())) if len(gx) else (0, -1)
            small = [i for i in range(1, n4) if not edge[i] and max(s4[i, cv2.CC_STAT_WIDTH], s4[i, cv2.CC_STAT_HEIGHT]) <= hgt
                     and s4[i, cv2.CC_STAT_LEFT] >= lo_x and s4[i, cv2.CC_STAT_LEFT] + s4[i, cv2.CC_STAT_WIDTH] - 1 <= hi_x]
            if small:
                halo |= np.isin(c4, small)
            # quầng của dòng này không lấn vùng của DÒNG KHÁC (nút nhỏ nằm sát dưới một tiêu đề to: mix1 f02 nút đỏ bị tiêu đề
            # coi là "quầng" và lau mất) -- dòng kia tự lo phần của nó
            theirs = ((count[y0:y1, x0:x1] - r.astype(np.uint8)) > 0) & ~r0
            halo = cv2.dilate((halo & ~theirs).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & r2 & ~theirs & ~hold[y0:y1, x0:x1]
            n_halo += bool(halo.sum() > 0.25 * max(1.0, float(grown.sum())))
        mask[y0:y1, x0:x1] |= grown | halo
    mask &= ~(ink & hold)
    if log is not None:
        free = ink & ~mask & ~hold
        log.append("nền = nháp trừ chữ (Hi-SAM + LaMa): mặt nạ %.1f%% ảnh; dòng theo %s; %d dòng có quầng; %d dòng giữ nguyên; nét "
                   "không thuộc dòng nào %.2f%% ảnh" % (100 * mask.mean(), ", ".join(f"{k} {v}" for k, v in n.items() if v), n_halo,
                                                       len(kept), 100 * free.mean()))
    return mask


def kept_of(P: dict | None) -> set:
    """Các dòng lệnh "keep" giữ nguyên điểm ảnh nháp."""
    return {m for op in (P or {}).get("ops") or [] if op.get("kind") == "keep" for m in op.get("marks") or op.get("slots") or []}


def erase(draft: np.ndarray, ink: np.ndarray, M: dict, log: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """GỠ MỌI DÒNG, MỘT LẦN LaMa cho cả poster: LaMa lấp mặt nạ gỡ chữ (erase_mask, không dòng nào giữ) trên nháp; ngoài mặt nạ là
    nguyên điểm ảnh nháp. -> (nền sạch chữ, mặt nạ). Nền này VLM nhìn, và settle() lấy lại từ nó cho mọi lượt dựng.
    Thiếu big-lama.pt: báo lỗi và dừng (không còn đường dự phòng: bản xoá FLUX đã rời nhánh main)."""
    import cv2
    if not lama_ready():
        raise RuntimeError(f"thiếu tệp LaMa {LAMA_PATH} (docs/SERVER.md, mục LaMa; đổi chỗ: export TEXTFIX_LAMA=<đường dẫn>)")
    mask = erase_mask(draft, ink, M, set(), log)
    if not mask.any():
        return draft.copy(), mask
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)[..., None]
    return (_lama(draft, mask) * a + draft * (1 - a)).round().clip(0, 255).astype(np.uint8), mask


def settle(draft: np.ndarray, clean: np.ndarray, ink: np.ndarray, M: dict, P: dict | None = None,
           log: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """NỀN DỰNG của một kế hoạch: nền sạch chữ (erase) ở trong mặt nạ của các dòng KHÔNG giữ, nháp gốc ở mọi chỗ khác -- dòng VLM
    chọn "keep" tự trở về đúng điểm ảnh nháp. Chỉ tính lại mặt nạ, không gọi LaMa (người dùng chỉ ra 14/10: không cần gỡ hai lần).
    Chỗ hai mặt nạ chồng nhau thuộc về dòng bị gỡ. -> (nền, mặt nạ)."""
    import cv2
    kept = kept_of(P)
    if not kept:
        mask = erase_mask(draft, ink, M, set(), log)
        return clean, mask
    mask = erase_mask(draft, ink, M, kept, log)
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)[..., None]
    return (clean * a + draft * (1 - a)).round().clip(0, 255).astype(np.uint8), mask


def draft_base(draft: np.ndarray, ink: np.ndarray, M: dict, P: dict | None = None, log: list | None = None) -> np.ndarray:
    """Một lần gọi cho script đo / dựng lại: gỡ rồi trả dòng giữ."""
    clean, _m = erase(draft, ink, M)
    return settle(draft, clean, ink, M, P, log)[0]


def hisam_image(draft: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Ảnh xem đầu ra THÔ của Hi-SAM: nháp nhạt đi, mọi điểm Hi-SAM coi là nét chữ tô HỒNG."""
    vis = draft.astype(np.float32) * 0.55 + 255 * 0.45
    ink = np.asarray(ink, bool)
    vis[ink] = vis[ink] * 0.15 + np.asarray((235, 0, 200), np.float32) * 0.85
    return vis.round().clip(0, 255).astype(np.uint8)


def ink_image(draft: np.ndarray, ink: np.ndarray, mask: np.ndarray, M: dict, kept: set) -> np.ndarray:
    """Ảnh xem bước gỡ chữ: nét bị gỡ XANH LÁ, phần nới / quầng quanh nét VÀNG, nét của dòng giữ XANH DƯƠNG, nét không thuộc dòng
    nào (để nguyên) HỒNG."""
    H, W = draft.shape[:2]
    hold = np.zeros((H, W), bool)
    for l in M["L"]:
        if l["id"] in kept:
            hold |= _region(l, H, W, 0.3)
    ink = np.asarray(ink, bool)
    vis = draft.astype(np.float32) * 0.55 + 255 * 0.45
    for sel, col, a in ((mask & ~ink, (255, 210, 0), 0.55), (mask & ink, (0, 170, 60), 0.85), (ink & ~mask & hold, (30, 110, 240), 0.85),
                        (ink & ~mask & ~hold, (235, 0, 200), 0.85)):
        vis[sel] = vis[sel] * (1 - a) + np.asarray(col, np.float32) * a
    return vis.round().clip(0, 255).astype(np.uint8)
