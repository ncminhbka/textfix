"""CHỈ DÒNG CHỮ (13/10, `TEXTFIX_LONLY=on`, mặc định TẮT): poster = bản NHÁP với chữ viết lại cho rõ.

Người dùng chốt sau bộ accept1: (1) không bỏ dòng nào nháp đã vẽ, thiếu câu khách thì VLM viết cho đủ; (2) icon nhỏ trước mỗi
dòng -> vẽ lại; (3) mọi thứ còn lại -- vỏ, icon to, hoạ tiết, hàng sao, avatar -- là điểm ảnh của nháp. Không còn ô vỏ S, ô chi
tiết I: VLM chỉ nhận dòng chữ. Bản xoá của FLUX chỉ còn một việc: cho biết DƯỚI NÉT CHỮ CŨ là gì.

  M  = view(slots.build(draft, plate))        # chỉ còn L
  P  = fix_ops(plan(...), M)                  # lệnh của VLM: chỉ ô L, không nền / viền riêng
  bg = draft_base(draft, plate, M, P)         # nền dựng = nháp, trừ chỗ là chữ
"""

from __future__ import annotations

import os
import re

import numpy as np

LONLY = os.environ.get("TEXTFIX_LONLY", "off") == "on"

SAME_DE = 8.0     # vành quanh dòng: bản xoá cách nháp <= 8 ΔE (trung vị) = nền / vỏ ở đó còn nguyên (nền không vỏ 1-4, vỏ nhạt bị gỡ 10-18)
DIFF_GONE = 20.0  # nháp cách bản xoá > 20 ΔE = bước xoá đã gỡ thứ ở đó (cùng số với overlay.DIFF)
MISS_FRAC = 0.25  # vùng chữ của bộ dò có >= 25% diện tích nằm ngoài mọi ô dòng = có chữ OCR bỏ sót ở đó
FLAT_DE = 12.0    # điểm thuộc nền của dòng: cách màu nền <= 12 ΔE
FLAT_MIN = 0.3    # ... và nền chiếm >= 30% vùng dòng; ít hơn = không có nền phẳng (chữ trên ảnh chụp)


def view(M: dict) -> dict:
    """Ô để VLM và bộ dựng làm việc: chỉ dòng chữ (dấu nhỏ đầu dòng đã nằm trong ô dòng, slots._row_marks)."""
    L = [{**l, "shell": None} for l in M["L"]]
    return {**M, "L": L, "S": [], "I": [], "shells": {}, "by": {l["id"]: l for l in L}}


def describe_l(M: dict) -> str:
    from . import slots
    note = lambda m: (slots._curve_note(m) if m.get("path") else f", drawn at {m['angle']:.0f} degrees" if abs(m.get("angle") or 0) >= 2   # noqa: E731
                      else "") + (", OCR missed this line: the letters read here are a guess" if m.get("missed") else "") \
        + (", row mark in front (inside this slot)" if m.get("mark") else "") \
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
2. EVERY LINE THE MODEL DREW IS WRITTEN. Never leave an L# out and never skip one: an empty spot next to its icon or inside \
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
- "row mark in front": a small bullet / check / icon stood at the left INSIDE that line's slot and was wiped with the \
letters. Start the row's html with the mark and a space -- "• " for a dot, <i-icon name="check" color="#hex"></i-icon> for \
a check or a small icon (drawn at text size) -- the same mark on every row of the list. Never a ring or circle for a dot.
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
 "ops": [{"id": "o1", "slots": ["L1"], "kind": "text|keep", "client": "T0" | null, "html": "...", "box_style": "...",
          "group": "name" | null, "group_kind": "list" | "flow" | null, "align": "left" | "center" | "right" | null,
          "why": "a few words"}],
 "missing": ["T5"], "notes": "..."}

FONT CATALOG (key: family, class): {fonts}

SMALL MARK ICONS (Lucide names for row marks): check, circle-check, square-check, badge-check, arrow-right, chevron-right, \
star, heart, sparkles, phone, mail, map-pin, globe, clock, calendar"""


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
        if l.get("mark"):   # ô đã nới sang trái ôm dấu đầu dòng: lấy khung thẳng của ô
            poly = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        cv2.fillPoly(m, [np.asarray(poly, np.int32)], 1)
        h = float(min(b[3] - b[1], b[2] - b[0])) if abs(l.get("angle") or 0) < 2 else float(l.get("size") or (b[3] - b[1]))
    k = max(1, int(round(grow * h)))
    # dấu thanh tiếng Việt nhô lên khỏi khung OCR: nới dọc nhiều hơn ngang
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * int(1.6 * k) + 1))).astype(bool)


def draft_base(draft: np.ndarray, plate: np.ndarray, M: dict, P: dict | None = None, log: list | None = None) -> np.ndarray:
    """NỀN DỰNG = NHÁP, trừ chỗ là chữ. Với mỗi dòng không giữ (dòng "keep" = chữ in trên vật: nguyên điểm ảnh nháp):
      - vành sát quanh dòng ở bản xoá gần giống nháp (nền / vỏ còn nguyên) -> trong vùng dòng lấy điểm ảnh BẢN XOÁ (FLUX đã vẽ
        lại phần dưới chữ, khít với xung quanh);
      - vành khác hẳn (bước xoá gỡ cả vỏ) và vành trên nháp phẳng màu -> giữ vỏ của nháp, LẤP nét chữ cũ (điểm khác màu vành)
        bằng màu quanh nét; vành không phẳng (ảnh chụp bị vẽ lại khác) -> lấy bản xoá.
    Vùng chữ của bộ dò mà OCR không trả dòng nào (chữ méo OCR bỏ sót) -> bản xoá, để chữ sai không hiện lại.
    Dấu nhỏ đầu dòng nằm trong ô dòng: xoá cùng chữ (VLM vẽ lại trong lệnh chữ)."""
    import cv2
    from . import ocr, slots
    H, W = draft.shape[:2]
    A, B = slots._lab(draft), slots._lab(plate)
    e = np.linalg.norm(A - B, axis=2)
    kept = {m for op in (P or {}).get("ops") or [] if op.get("kind") == "keep" for m in op.get("marks") or []}
    regs = {l["id"]: _region(l, H, W, 0.3) for l in M["L"]}
    every = np.zeros((H, W), bool)
    for r in regs.values():
        every |= r
    use_plate, fill = np.zeros((H, W), bool), np.zeros((H, W), bool)
    n_plate = n_fill = 0
    marks = []   # (khung dấu đầu dòng, màu nền quanh dòng): bước xoá có khi GIỮ dấu -- lau nốt, VLM vẽ dấu mới trong lệnh chữ
    for l in M["L"]:
        if l["id"] in kept:
            continue
        R = regs[l["id"]]
        # NỀN CỦA DÒNG = màu chiếm nhiều nhất trong vùng dòng trên nháp (lòng vỏ, hoặc nền phẳng; nét chữ là phần còn lại).
        # Không lấy vành ngoài dòng: pill ôm sát chữ thì vành đã nằm ngoài pill.
        q = (draft[R] // 16).astype(np.int32)
        key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
        c = np.median(A[R][key == np.bincount(key).argmax()], axis=0)
        near = R & (np.linalg.norm(A - c, axis=2) <= FLAT_DE)
        if float(near.sum()) < FLAT_MIN * float(R.sum()):   # không có màu nền rõ: chữ nằm trên ảnh chụp -> bản xoá
            use_plate |= R
            n_plate += 1
            continue
        if l.get("mark"):
            marks.append(([int(v) for v in l["mark"]["box"]], c))
        if float(np.median(e[near])) <= SAME_DE:   # nền của dòng còn nguyên ở bản xoá (vỏ được giữ, hoặc không có vỏ)
            use_plate |= R
            n_plate += 1
        else:                                       # bước xoá gỡ cả vỏ: giữ vỏ của nháp, lấp nét chữ cũ
            fill |= R & (np.linalg.norm(A - c, axis=2) >= slots.INK_IN_SHELL)
            n_fill += 1
    # CHỮ OCR BỎ SÓT (13/10 dựng lại accept1: b03 chữ số "2" to của tiêu đề, g04 vòm chữ thứ hai -- không thuộc ô dòng nào nên
    # hiện lại nguyên dạng méo cạnh chữ mới): vùng chữ của bộ dò mà một phần đáng kể (>= MISS_FRAC) nằm ngoài mọi ô dòng -> phần
    # ngoài đó lấy bản xoá. Vùng nới gấp đôi `pad`: chữ to có viền / bóng rộng hơn vùng bộ dò khoanh.
    n_miss, cc = 0, None
    for Rg in ocr.regions(draft):
        k = max(1, int(round(2 * Rg["pad"])))
        near = cv2.dilate(Rg["mask"].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
        out_ = near & ~every
        if float(out_.sum()) < MISS_FRAC * float(near.sum()):
            continue
        # ... kèm trọn các mảng bước xoá đã gỡ chạm vùng đó (nét viền / bóng của chữ to vượt ra ngoài vùng bộ dò khoanh)
        if cc is None:
            _n, cc = cv2.connectedComponents(((e > DIFF_GONE) & ~every).astype(np.uint8), connectivity=8)
        ids = np.unique(cc[out_])
        ids = ids[ids > 0]
        if len(ids):
            out_ = out_ | np.isin(cc, ids)
        use_plate |= out_
        n_miss += 1
    a = cv2.GaussianBlur(use_plate.astype(np.float32), (0, 0), 2.0)[..., None]
    out = (plate * a + draft * (1 - a)).round().clip(0, 255).astype(np.uint8)
    if marks:
        Lo = slots._lab(out)
        for (x0, y0, x1, y1), c in marks:
            x0, y0, x1, y1 = max(0, x0 - 2), max(0, y0 - 2), min(W, x1 + 2), min(H, y1 + 2)
            fill[y0:y1, x0:x1] |= np.linalg.norm(Lo[y0:y1, x0:x1] - c, axis=2) >= slots.INK_IN_SHELL
    if fill.any():
        k = 2 * slots.KEEP_HALO + 1
        out = cv2.inpaint(out, cv2.dilate(fill.astype(np.uint8), np.ones((k, k), np.uint8)), 3, cv2.INPAINT_TELEA)
    if log is not None:
        log.append(f"nền = nháp trừ chữ: {n_plate} dòng lấy bản xoá, {n_fill} dòng lấp nét trong vỏ bị gỡ, {n_miss} vùng chữ OCR sót "
                   f"lấy bản xoá, {len(kept)} dòng giữ nguyên")
    return out
