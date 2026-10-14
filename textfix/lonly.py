"""CHỈ DÒNG CHỮ -- ĐƯỜNG DUY NHẤT của nhánh main: poster = bản NHÁP với chữ viết lại cho rõ.

Người dùng chốt sau bộ accept1: (1) không bỏ dòng nào nháp đã vẽ, thiếu câu khách thì VLM viết cho đủ; (2) icon nhỏ trước mỗi
dòng -> vẽ lại; (3) mọi thứ còn lại -- vỏ, icon to, hoạ tiết, hàng sao, avatar -- là điểm ảnh của nháp. VLM chỉ nhận dòng chữ.

14/10: KHÔNG CÒN BẢN XOÁ FLUX (đường đó: nhánh git `legacy-flux-erase`). Chữ nằm ở đâu do Hi-SAM trả lời (hisam.strokes: mặt nạ
nét chữ của nháp), dưới chữ là gì do LaMa vẽ lại:

  ink = hisam.strokes(draft)                  # nét chữ (trạm vẽ)
  M   = view(slots.build(draft, ink))         # chỉ còn L: dòng OCR + cụm nét chưa dòng nào nhận ("orphan": VLM quyết giữ / xoá)
  clean, mask = erase(draft, ink, M)         # LaMa MỘT lần: gỡ mọi dòng (nền VLM nhìn)
  A   = fill.parse(plan(...))                 # VLM GHÉP câu khách <-> dòng + kiểu theo vai (chặng B: không viết html)
  base, mask = settle(draft, clean, ink, M, {"ops": U})   # dòng "keep" trả về điểm ảnh nháp
"""

from __future__ import annotations

import os

import numpy as np

INK_SAME = 25.0   # điểm cùng màu mực của dòng: cách màu mực <= 25 ΔE
INK_IN = 0.7      # mảng cùng màu mực là nét chữ khi >= 70% của nó nằm trong vùng dòng (vòng con dấu, viền nút: phần lớn ở ngoài)
GLYPH_MAX = 2.0   # ... và cạnh dài <= 2 cao dòng (một chữ cái / chữ số); dài hơn là hình khác cùng màu mực


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
