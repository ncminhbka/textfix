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

DIFF_GONE = 20.0  # nháp cách bản xoá > 20 ΔE = bước xoá đã gỡ thứ ở đó (cùng số với overlay.DIFF)
FLAT_DE = 12.0    # điểm thuộc nền của dòng: cách màu nền <= 12 ΔE
FLAT_MIN = 0.3    # ... và nền chiếm >= 30% vùng dòng; ít hơn = không có nền phẳng (chữ trên ảnh chụp)
FIT_DE = 6.0      # điểm khớp MẶT MÀU NỀN (mặt bậc hai khớp từ các điểm nền của dòng): cách mặt <= 6 ΔE
FIT_MIN = 0.9     # nền phẳng / chuyển màu đều: >= 90% số điểm cách mặt <= 12 thì cách mặt <= 6 (đo 13/10: chữ trong nút, thẻ,
                  # trên nền trơn 0,92-0,99; chữ trên gối, trên sàn gỗ có quầng tối 0,83-0,86)
INK_CORE = 0.3    # mảng khác màu nền là NÉT CHỮ khi >= 30% của nó nằm trong lõi dòng (viền nút, icon cạnh dòng: phần lớn ở ngoài)
SAME_DE = 8.0     # bản xoá cách nháp <= 8 ΔE (trung vị) = thứ ở đó còn nguyên
INK_SAME = 25.0   # điểm khác nền cùng màu mực: cách màu mực của dòng <= 25 ΔE (khác hơn = viền vỏ, bóng, icon cạnh dòng)
INK_SMALL = 0.5   # ... hoặc mảng nhỏ (dấu thanh, chấm, mẩu nét) cạnh < 0,5 cao dòng nằm trong vùng dòng


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


def _quad(ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(xs)), xs, ys, xs * xs, xs * ys, ys * ys]


def draft_base(draft: np.ndarray, plate: np.ndarray, M: dict, P: dict | None = None, log: list | None = None) -> np.ndarray:
    """NỀN DỰNG = NHÁP, trừ chỗ là chữ. Với mỗi dòng không giữ (dòng "keep" = chữ in trên vật: nguyên điểm ảnh nháp):
      - dòng nằm trên NỀN PHẲNG / chuyển màu đều (lòng nút, thẻ, nền trơn): khớp một mặt màu nền từ chính nháp, TÔ ĐÈ đúng nét
        chữ cũ bằng mặt đó. Không dán gì từ bản xoá -> không có mép miếng dán, vỏ của nháp còn nguyên dù bước xoá gỡ hay giữ.
        (13/10 người dùng xem lượt đầu: "viền gai mờ" -- miếng bản xoá lệch tông lộ khung chữ nhật, lấp nét bằng inpaint để
        lại bóng chữ lốm đốm.)
      - dòng nằm trên ảnh chụp / nền có vân / chữ có quầng: lấy vùng dòng từ BẢN XOÁ (FLUX đã vẽ lại phần dưới chữ).
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
    use_plate = np.zeros((H, W), bool)
    paint = []   # (mặt nạ nét chữ, khung, màu nền tại khung) của các dòng trên nền phẳng
    n_plate = n_paint = 0
    why = []   # mỗi dòng lấy bản xoá: vì sao (để truy khi nút / thẻ bị khoét)
    for l in M["L"]:
        if l["id"] in kept:
            continue
        R, R0, R2 = regs[l["id"]], _region(l, H, W, 0.1), _region(l, H, W, 0.6)
        # NỀN CỦA DÒNG = màu chiếm nhiều nhất trong LÕI dòng trên nháp (lòng vỏ, hoặc nền phẳng; nét chữ là phần còn lại).
        # Không lấy vành ngoài dòng, cũng không lấy cả vùng nới: nút ôm sát chữ thì phần nới đã nằm ngoài nút (b01 nút "Đặt
        # lịch ngay": màu nhiều nhất của vùng nới là nền kem ngoài nút -> cả nút bị coi là nét chữ).
        q = (draft[R0] // 16).astype(np.int32)
        key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
        c = np.median(A[R0][key == np.bincount(key).argmax()], axis=0)
        ys, xs = np.nonzero(R2)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        sc = float(max(y1 - y0, x1 - x0))
        X = _quad((ys - y0) / sc, (xs - x0) / sc)
        inR = R[ys, xs]
        near = inR & (np.linalg.norm(A[ys, xs] - c, axis=1) <= FLAT_DE)
        if float(near.sum()) < FLAT_MIN * float(inR.sum()):   # không có màu nền rõ: chữ nằm trên ảnh chụp -> bản xoá
            use_plate |= R
            n_plate += 1
            why.append(f"{l['id']} không nền phẳng {float(near.sum()) / float(inR.sum()):.2f}")
            continue
        # MÀU NHIỀU NHẤT LÀ NÉT CHỮ, không phải nền (chữ đậm trắng trên vòm cầu vồng, g03): ở bản xoá các điểm màu đó đã đổi
        # hẳn còn phần còn lại của lõi dòng thì nguyên -> nền là phần còn lại, không phẳng -> bản xoá
        dom = R0 & (np.linalg.norm(A - c, axis=2) <= FLAT_DE)
        rest = R0 & ~dom
        # ... và màu đó RỜI THÀNH NHIỀU MẢNG (các chữ cái). Lòng nút thì liền một mảng: chữ trắng trong nút màu trên nền trắng cũng
        # cho đúng dấu hiệu trên khi bước xoá gỡ nút (d27 "-36%", "Mua ngay"; f13 nút xanh) -- dán bản xoá là lau trắng giữa nút
        _n, _cc, _st, _ct = cv2.connectedComponentsWithStats((R & (np.linalg.norm(A - c, axis=2) <= FLAT_DE)).astype(np.uint8), 8)
        one = _n > 1 and float(_st[1:, cv2.CC_STAT_AREA].max()) >= 0.6 * float(_st[1:, cv2.CC_STAT_AREA].sum())
        if (dom.any() and not one and float(rest.sum()) >= 0.15 * float(R0.sum()) and float(np.median(e[dom])) > DIFF_GONE
                and float(np.median(e[rest])) <= SAME_DE):
            use_plate |= R
            n_plate += 1
            why.append(f"{l['id']} màu nhiều nhất là nét chữ")
            continue
        res = np.linalg.norm(A[ys, xs] - X @ np.linalg.lstsq(X[near], A[ys, xs][near], rcond=None)[0], axis=1)
        bg = res <= FIT_DE
        flat = float((bg & inR).sum()) >= FIT_MIN * float(((res <= 2 * FIT_DE) & inR).sum())
        if not flat and float(np.median(e[ys, xs][near])) > SAME_DE:
            # VỎ CÓ ÁNH / CHUYỂN MÀU mà bước xoá đã GỠ (a03 nút vàng kim: dán bản xoá là khoét một lỗ thấy mặt nước trong nút):
            # nới gấp đôi -- nền = cách mặt <= 12, và nét chữ phải TÁCH HẲN khỏi nền (ít điểm lưng chừng 12-24; chữ có quầng
            # mờ trên ảnh chụp thì lưng chừng nhiều -> vẫn lấy bản xoá). Phần lệch khỏi mặt được bù cục bộ khi tô.
            bg = res <= 2 * FIT_DE
            flat = float((bg & inR).sum()) >= FIT_MIN * float(((res <= 4 * FIT_DE) & inR).sum())
        if not flat or float((bg & inR).sum()) < FLAT_MIN * float(inR.sum()):
            use_plate |= R   # nền có vân, hoặc chữ có quầng mờ rộng: mặt màu không tả được nền
            n_plate += 1
            why.append(f"{l['id']} nền có vân {float((bg & inR).sum()) / max(1.0, float(((res <= 2 * FIT_DE) & inR).sum())):.2f}")
            continue
        ink = np.zeros((y1 - y0, x1 - x0), np.uint8)
        ink[ys[~bg] - y0, xs[~bg] - x0] = 1
        hgt = float(l.get("size") or (l["box"][3] - l["box"][1]))
        r0, r2 = R0[y0:y1, x0:x1], R2[y0:y1, x0:x1].astype(np.uint8)
        rim = (r2 > 0) & (cv2.erode(r2, np.ones((3, 3), np.uint8), borderValue=0) == 0)   # mép vùng nới rộng
        # mảng nhỏ phải nằm GỌN trong vùng dòng: mẩu nền ngoài nút bị mép vùng dòng cắt ra (b01, dưới góc nút) không phải nét chữ
        gut = cv2.erode(R[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

        def pick(mask: np.ndarray, strict: bool) -> np.ndarray:
            """Các mảng của `mask` là nét chữ: phần lớn nằm trong lõi dòng; hoặc chạm lõi và nằm trọn trong vùng nới rộng (chữ
            số to lệch khỏi khung OCR) -- mảng chạm mép vùng nới rộng là thứ kéo dài ra ngoài dòng (viền nút, nền ngoài nút);
            hoặc mảng nhỏ nằm gọn trong vùng dòng (dấu thanh, chấm đầu dòng). strict: chỉ luật giữa."""
            n, cc, st, _c = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
            core, edge, inside = (np.bincount(cc[z], minlength=n) for z in (r0, rim, gut))
            ok = [i for i in range(1, n) if (core[i] > 0 and edge[i] == 0) or (not strict and (
                core[i] >= INK_CORE * st[i, cv2.CC_STAT_AREA]
                or (max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) < INK_SMALL * hgt and inside[i] == st[i, cv2.CC_STAT_AREA])))]
            return np.isin(cc, ok)

        got = pick(ink > 0, False)
        # NÉT CHỮ DÍNH LIỀN VIỀN VỎ khác màu (d07 chữ số "5" chạm viền bóng của vòng tròn): cả mảng bị coi là viền. Trong phần
        # chạm lõi mà chưa lấy, tách riêng các điểm CÙNG MÀU MỰC rồi xét lại.
        left = (ink > 0) & ~got
        k0 = left & r0
        if int(k0.sum()) >= 20:
            pa, pd = A[y0:y1, x0:x1], draft[y0:y1, x0:x1]
            src = (got & r0) if int((got & r0).sum()) >= 20 else k0
            q = (pd[src] // 16).astype(np.int32)
            key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
            ci = np.median(pa[src][key == np.bincount(key).argmax()], axis=0)
            got |= pick(left & (np.linalg.norm(pa - ci, axis=2) <= INK_SAME), True)
        take = bool(got.any())
        if take:
            k = 2 * max(3, int(round(0.04 * hgt))) + 1   # viền khử răng cưa + bóng mờ sát nét
            m = cv2.dilate(got.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
            col = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
            okb = bg & inR
            col[ys - y0, xs - x0] = X @ np.linalg.lstsq(X[okb], draft[ys, xs][okb].astype(np.float64), rcond=None)[0]
            # mặt bậc hai không tả hết nền (nút có vệt sáng, nền tối có quầng đèn): chỗ tô lệch nền vài ΔE, nhìn ra hình chữ mờ
            # ("lem", a02 nút cam). Bù phần lệch: trung bình có trọng số của (nháp - mặt) trên các điểm NỀN quanh nét.
            wb = np.zeros((y1 - y0, x1 - x0), np.float32)
            wb[ys[okb] - y0, xs[okb] - x0] = 1.0
            wb[cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)] = 0.0
            sg = max(4.0, 0.35 * hgt)
            den = cv2.GaussianBlur(wb, (0, 0), sg)
            num = cv2.GaussianBlur((draft[y0:y1, x0:x1].astype(np.float32) - col) * wb[..., None], (0, 0), sg)
            col += np.where(den[..., None] > 0.02, num / np.maximum(den[..., None], 1e-3), 0.0)
            paint.append((m & R2[y0:y1, x0:x1], (y0, y1, x0, x1), col))
        n_paint += 1
    # CHỮ OCR BỎ SÓT (13/10 dựng lại accept1: b03 chữ số "2" to của tiêu đề, g04 vòm chữ thứ hai -- không thuộc ô dòng nào nên
    # hiện lại nguyên dạng méo cạnh chữ mới): MẢNG BỊ GỠ (nháp cách bản xoá > DIFF_GONE) mà (a) chạm lõi một vùng chữ của bộ dò,
    # (b) phần lớn nằm ngoài mọi ô dòng, (c) không phải vỏ của một dòng (không trùm tâm dòng nào), (d) bộ đọc đọc ra chữ / số
    # -> lấy bản xoá. Icon bộ dò nhận nhầm là chữ thì bộ đọc không ra chữ: giữ nguyên của nháp.
    n_miss = 0
    gone = cv2.morphologyEx((e > DIFF_GONE).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _n, cc, st, _c = cv2.connectedComponentsWithStats(gone, 8)
    anchors = [slots._anchor(l) for l in M["L"]]
    done = set()
    for Rg in ocr.regions(draft):
        for i_ in (int(v) for v in np.unique(cc[Rg["mask"]]) if v > 0):
            if i_ in done:
                continue
            done.add(i_)
            x, y, w, h, area = (int(v) for v in st[i_])
            blob = cc[y:y + h, x:x + w] == i_
            cnts, _h = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            solid = cv2.drawContours(np.zeros(blob.shape, np.uint8), cnts, -1, 1, cv2.FILLED).astype(bool)
            # vỏ / nét của một dòng đã có ô: tâm dòng nằm TRONG mảng (lấp kín); không xét theo khung -- khung của một vòm chữ
            # trùm cả dòng nằm dưới vòm
            if any(x <= ax < x + w and y <= ay < y + h and solid[int(ay - y), int(ax - x)] for ax, ay in anchors):
                continue
            if float(every[y:y + h, x:x + w][blob].mean()) >= 0.5:
                continue
            ys_, xs_ = np.nonzero(blob)
            text = ocr.read_region(draft, cv2.minAreaRect(np.c_[xs_ + x, ys_ + y].astype(np.float32)))
            if not re.search(r"[A-Za-z0-9À-ỹ]", text):
                continue
            use_plate[y:y + h, x:x + w] |= cv2.dilate(blob.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            n_miss += 1
    a = cv2.GaussianBlur(use_plate.astype(np.float32), (0, 0), 2.0)[..., None]
    out = plate * a + draft * (1 - a)
    for m, (y0, y1, x0, x1), col in paint:
        al = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)[..., None]
        out[y0:y1, x0:x1] = col * al + out[y0:y1, x0:x1] * (1 - al)
    out = out.round().clip(0, 255).astype(np.uint8)
    if log is not None:
        log.append(f"nền = nháp trừ chữ: {n_paint} dòng tô đè nét chữ bằng màu nền của nháp, {n_plate} dòng lấy bản xoá, {n_miss} vùng chữ "
                   f"OCR sót lấy bản xoá, {len(kept)} dòng giữ nguyên" + (" | bản xoá vì: " + "; ".join(why) if why else ""))
    return out
