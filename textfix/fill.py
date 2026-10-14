"""ĐIỀN CHỮ (chặng B, 14/10): VLM ghép câu khách <-> dòng nháp, code chép NGUYÊN VĂN và dựng theo nháp.

  A = parse(vlm_json)                       # {"assign": [...], "styles": {...}, "accent": [...], "missing": [...]}
  U, log, missing = resolve(A, M, texts, product)   # đơn vị: một câu (hoặc vài câu một dòng) + các dòng nháp của nó
  U, log2 = layout(U, M, ink, A["styles"], A["accent"], texts)   # khung nét, cỡ, màu, lề, tập cùng vai, phần tử dựng
  render.render_units(page, base, U, M)     # dựng

VLM chỉ hiểu NGHĨA (dòng nào là câu nào, mỗi vai trông ra sao); code làm phần chính xác:
  1. chữ   = nguyên văn câu khách ("split": các mảnh nối lại phải đúng câu); dòng lấp (client null): chữ VLM, chặn số bịa;
  2. chỗ   = hình bao NÉT Hi-SAM của dòng (ít nét: khung OCR); dòng nghiêng / cong giữ cách dựng của render.op_geom / path_of;
  3. cỡ    = trung vị cỡ đo (glyph) các dòng của câu; màu = màu đo của dòng; kiểu (font, đậm, nghiêng, hoa) = theo VAI;
  4. lề    = mép nét nháp; ngắt dòng = theo tỉ lệ số ký tự OCR các dòng nháp;
  5. TẬP   = cùng vai, cỡ nháp đôi một lệch <= SET_RATIO: một cỡ, một màu, một lề, một mức nén (render);
  6. co    = nén ngang <= render.CONDENSE rồi co cỡ, theo đơn vị (dòng / khối / tập);
  7. các BƯỚC (câu "ordered" của brief) luôn theo thứ tự khách, đọc hàng trên trái -> phải rồi hàng dưới;
  8. một câu ở nhiều chỗ / dòng chưa ai ghép: xem resolve.
Người dùng chốt 14/10: nhóm lấy từ vai câu khách (brief), không đoán từ ảnh; không cắt chữ, chỉ co; nháp thiếu câu thì chịu."""

from __future__ import annotations

import html as _html
import re
from collections import Counter

import numpy as np

SET_RATIO = 1.25    # tập cùng vai: cỡ nháp đôi một lệch <= 1,25 lần (probe 394b: trung vị +-1,4 gộp nhầm nhãn 22px với website 32px)
SPLIT_RATIO = 1.4   # các dòng của MỘT câu lệch cỡ quá 1,4 lần ("Giảm" nhỏ / "15%" to): mỗi dòng giữ cỡ của nó
TALL = 1.7          # một khung OCR cao > 1,7 cỡ chữ = ôm nhiều dòng chữ (huy hiệu ba dòng 582e)
STACK_GAP = 1.0     # các dòng của một khối: khe dọc <= 1 cỡ chữ, chồng ngang
ROW_GAP = 2.0       # mẩu cùng hàng của một câu (OCR cắt "NGÀY" / "• HỘI"): chồng dọc >= nửa cao, khe ngang <= 2 cỡ -> một phần tử
CENTER_TOL = 0.3    # tâm các dòng lệch <= 0,3 cỡ: căn giữa
EDGE_TOL = 0.6      # mép lệch <= 0,6 cỡ: thẳng mép đó
PIECE_SIM = 0.8     # các mẩu chữ VLM nối lại giống câu khách >= 0,8 = một câu chia nhiều chỗ (VLM viết lệch dấu: "TRƢƠNG", i03)
QUOTES = "\"“”«»'‘’"
DIFF_RGB = 120.0    # hai màu đo cách nhau > 120 (khoảng cách RGB) = khác hẳn (đỏ / trắng ~ 255, đen / xám đậm ~ 60)
EFFECT_KEYS = re.compile(r"^\s*(-webkit-)?(text-shadow|text-stroke(-width|-color)?|background(-image|-clip|-color)?|"
                         r"text-fill-color|filter|paint-order)\s*:", re.I)


def _s():
    from . import slots
    return slots


# ---------------------------------------------------------------------------------------------------- câu trả lời VLM
def parse(raw) -> dict:
    """Câu trả lời VLM -> {"assign": [đơn vị thô], "styles", "accent", "missing", "notes"}. Mục sai khuôn bị bỏ, ghi "dropped"."""
    S = _s()
    P = S._obj(raw)
    out, bad = [], []
    for i, a in enumerate(P.get("assign") or []):
        a = S._obj(a) if isinstance(a, str) else a
        if not isinstance(a, dict):
            bad.append(str(a)[:80])
            continue
        ln = a.get("lines") or a.get("slots") or []
        ln = [x for x in (ln if isinstance(ln, list) else [ln]) if isinstance(x, str)]
        cl = a.get("client")
        cl = cl if isinstance(cl, list) else [] if cl is None else [cl]
        sp = a.get("split")
        sp = [str(x).strip() for x in sp] if isinstance(sp, list) and sp and all(isinstance(x, (str, int, float)) for x in sp) else None
        out.append({"id": f"u{i + 1}", "kind": a.get("kind") if a.get("kind") in ("keep", "skip") else "text", "marks": ln,
                    "clients": [k for k in (S._tid(c) for c in cl) if k is not None], "split": sp,
                    "text_vlm": str(a.get("text") or "").strip(), "role_vlm": str(a.get("role") or "").strip() or None,
                    "bullet": bool(a.get("bullet")), "why": str(a.get("why") or "")[:160]})
    st = P.get("styles") if isinstance(P.get("styles"), dict) else {}
    return {"assign": out, "styles": {str(k): v for k, v in st.items() if isinstance(v, dict)},
            "accent": [x for x in P.get("accent") or [] if isinstance(x, dict) and str(x.get("text") or "").strip()],
            "missing": [k for k in (S._tid(x) for x in P.get("missing") or []) if k is not None],
            "notes": P.get("notes"), **({"dropped": bad} if bad else {})}


def covered(A: dict, M: dict) -> set:
    """Các dòng đã được ghép (viết câu / chữ lấp / giữ)."""
    return {m for u in A.get("assign") or [] if u["kind"] == "keep" or u["clients"] or u["text_vlm"] for m in u["marks"] if m in M["by"]}


def from_ops(P: dict, texts: list[dict]) -> dict:
    """plan.json CŨ (lệnh html của VLM) -> khuôn mới, để dựng lại tại máy không gọi VLM (scripts/lonly_replay.py, probe 14/10):
    lệnh chữ khai câu khách -> câu đó; câu khách nằm NGUYÊN VĂN trong chữ lệnh khác cũng ghép vào đó (OCR gộp tên + giá);
    nhiều lệnh cùng một câu mà chữ nối lại đúng câu -> một đơn vị chia mẩu (split); kiểu theo vai rút từ html / CSS của lệnh."""
    S = _s()
    from .render import font_key
    # lệnh "fill_*" / "fix_*" do luật CŨ của mã sinh (giữ chữ nháp, skip bỏ sót), không phải VLM chọn: bỏ, để resolve quyết theo luật mới
    ops = [o for o in P.get("ops") or [] if isinstance(o, dict) and not str(o.get("id") or "").startswith(("fill_", "fix_"))]
    A, used = [], set()
    plain = lambda h: " ".join(_html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<br\s*/?>", " ", h or "", flags=re.I))).split())  # noqa: E731
    for i, o in enumerate(ops):
        mk = [m for m in (o.get("slots") or o.get("marks") or []) if isinstance(m, str)]
        if o.get("kind") in ("keep", "skip") or not o.get("html"):
            A.append({"id": f"u{i + 1}", "kind": "keep" if o.get("kind") == "keep" else "skip", "marks": mk, "clients": [], "split": None,
                      "text_vlm": "", "role_vlm": None, "bullet": False, "why": o.get("why") or "", "_op": o})
            continue
        k = S._tid(o.get("client"))
        A.append({"id": f"u{i + 1}", "kind": "text", "marks": mk, "clients": [k] if k is not None and k < len(texts) else [],
                  "split": None, "text_vlm": plain(o["html"]), "role_vlm": None, "bullet": bool(re.match(r"\s*(<[^>]+>\s*)*[•·●]", o["html"])),
                  "why": o.get("why") or "", "_op": o})
        used |= set(A[-1]["clients"])
    for u in A:
        fv = S._flat(u["text_vlm"])
        extra = [(fv.find(S._flat(t["text"])), j) for j, t in enumerate(texts)
                 if j not in used and len(S._flat(t["text"])) >= 3 and S._flat(t["text"]) in fv]
        if extra:
            u["clients"] = [k for _, k in sorted([(fv.find(S._flat(texts[k]["text"])), k) for k in u["clients"]] + extra)]
            used |= {j for _, j in extra}
    # một câu chia nhiều lệnh (tiêu đề "NGÀY / HỘI / SIÊU / SALE" mỗi mẩu một lệnh): gộp thành một đơn vị có split
    from difflib import SequenceMatcher
    by_k = {}
    for u in A:
        if u["kind"] == "text" and len(u["clients"]) == 1:
            by_k.setdefault(u["clients"][0], []).append(u)
    for k, us in by_k.items():
        if len(us) < 2:
            continue
        joined = " ".join(u["text_vlm"] for u in us)
        if SequenceMatcher(None, S._flat(joined), S._flat(texts[k]["text"])).ratio() >= PIECE_SIM:
            head = us[0]
            head["marks"] = [m for u in us for m in u["marks"]]
            head["split"] = [u["text_vlm"] for u in us] if all(len(u["marks"]) == 1 for u in us) else None
            for u in us[1:]:
                A.remove(u)
    # kiểu theo vai: lệnh đầu tiên của vai có ghi font / độ đậm / hoa / hiệu ứng
    st, styles = P.get("style") or {}, {}
    for u in A:
        o, ks = u.get("_op") or {}, u["clients"]
        if u["kind"] != "text" or not ks:
            continue
        role = texts[ks[0]]["role"]
        css = (o.get("box_style") or "") + ";" + " ".join(re.findall(r"style=[\"']([^\"']*)", o.get("html") or ""))
        fk = re.search(r"<i-font\b[^>]*key=[\"']([^\"']+)", o.get("html") or "") or re.search(r"font-family\s*:\s*['\"]?([^;'\",]+)", css)
        wt = re.search(r"<i-font\b[^>]*weight=[\"'](\d+)", o.get("html") or "") or re.search(r"font-weight\s*:\s*(\d+)", css)
        s = styles.setdefault(role, {})
        if fk and "font" not in s and font_key(fk.group(1)):
            s["font"] = font_key(fk.group(1))
        if wt and "weight" not in s:
            s["weight"] = int(wt.group(1))
        if "case" not in s and re.search(r"text-transform\s*:\s*uppercase", css):
            s["case"] = "upper"
        if "italic" not in s and re.search(r"font-style\s*:\s*italic|<i>|<em>", css + (o.get("html") or "")):
            s["italic"] = True
        col = re.search(r"(?<![-\w])color\s*:\s*(#[0-9a-fA-F]{6})\b", css) or re.search(r"<i-font\b[^>]*color=[\"'](#[0-9a-fA-F]{6})", o.get("html") or "")
        if col and "color" not in s:
            s["color"] = col.group(1)
        eff = [x for x in css.split(";") if EFFECT_KEYS.match(x)]
        if eff and "effect" not in s:
            s["effect"] = ";".join(eff)
    for role in {texts[k]["role"] for u in A for k in u["clients"]}:
        s = styles.setdefault(role, {})
        if "font" not in s and font_key(st.get("text_font")):
            s["font_default"] = font_key(st.get("text_font"))
            s["font_display"] = font_key(st.get("display_font")) or font_key(st.get("text_font"))
    for u in A:
        u.pop("_op", None)
    return {"assign": A, "styles": styles, "accent": [], "missing": [S._tid(x) for x in P.get("missing") or [] if S._tid(x) is not None],
            "notes": P.get("notes")}


# ---------------------------------------------------------------------------------------------------- ghép
def resolve(A: dict, M: dict, texts: list[dict], product: bool = False) -> tuple[list[dict], list[str], list[int]]:
    """Đơn vị dựng từ phần ghép của VLM. -> (đơn vị, nhật ký, câu khách không có trên poster)."""
    S = _s()
    by, log = M["by"], []
    from difflib import SequenceMatcher
    sim = lambda a, b: SequenceMatcher(None, S._flat(a), S._flat(b)).ratio()  # noqa: E731
    U = []
    for u in A.get("assign") or []:
        u = {**u, "marks": [m for m in u["marks"] if m in by], "clients": [k for k in u["clients"] if 0 <= k < len(texts)]}
        if u["marks"]:
            U.append(u)
    # MỘT DÒNG MỘT ĐƠN VỊ: viết > giữ > xoá; ngang nhau: đơn vị đứng trước (VLM ghi thêm keep cho dòng đã có chữ -> hai lớp chữ)
    rank = lambda u: 0 if u["kind"] == "text" and (u["clients"] or u["text_vlm"]) else 1 if u["kind"] == "keep" else 2  # noqa: E731
    owner = {}
    for i, u in sorted(enumerate(U), key=lambda t: (rank(t[1]), t[0])):
        for m in u["marks"]:
            owner.setdefault(m, i)
    for i, u in enumerate(U):
        lost = [m for m in u["marks"] if owner[m] != i]
        if lost:
            log.append(f"{u['id']}: dòng {lost} đã thuộc đơn vị khác -> bỏ khỏi đơn vị này")
            u["marks"] = [m for m in u["marks"] if owner[m] == i]
    U = [u for u in U if u["marks"]]
    # CHỮ IN TRÊN SẢN PHẨM KHÁCH TẢI (mark_printed: chữ OCR trùng chữ đọc trên ảnh sản phẩm) giữ nguyên điểm ảnh -- trừ khi VLM
    # ghép dòng đó vào một CÂU KHÁCH: VLM nhìn ảnh thấy đó là chữ poster (14/10 dev2 d39 / d14: tiêu đề "Cà phê hoà tan Classic"
    # vẽ hai dòng, dòng "Classic" trùng chữ trên gói -> bị giữ dạng giun, câu dồn vào một dòng)
    for u in list(U):
        pr = [m for m in u["marks"] if by[m].get("printed")]
        if pr and u["kind"] != "keep" and not u["clients"]:
            log.append(f"{u['id']}: {pr} là chữ in trên sản phẩm khách tải -> giữ nguyên")
            U.append({**u, "id": u["id"] + "p", "kind": "keep", "marks": pr, "clients": [], "split": None, "text_vlm": ""})
            u["marks"] = [m for m in u["marks"] if m not in pr]
    U = [u for u in U if u["marks"]]
    # MỘT CÂU CHIA Ở NHIỀU ĐƠN VỊ (VLM ghi mỗi mẩu một mục: "Du lịch Mây Ngàn -" | "0283 777 8899", 14/10 mix1 m04): các mẩu nối
    # lại theo thứ tự đọc ra đúng câu -> gộp thành một đơn vị nhiều dòng, mỗi dòng một mẩu
    for k in {u["clients"][0] for u in U if len(u["clients"]) == 1 and u["split"]}:
        us = [u for u in U if u["kind"] == "text" and u["clients"] == [k] and u["split"] and len(u["split"]) == len(u["marks"])]
        if len(us) < 2:
            continue
        us = [us[i] for i in _reading([by[u["marks"][0]]["box"] for u in us])]
        if sim("".join(p for u in us for p in u["split"]), texts[k]["text"]) < PIECE_SIM:
            continue
        us[0].update(marks=[m for u in us for m in u["marks"]], split=[p for u in us for p in u["split"]], split_ok=True)
        U = [u for u in U if u not in us[1:]]
        log.append(f"{us[0]['id']}: câu T{k} chia ở {[u['id'] for u in us]} -> một đơn vị {us[0]['marks']}")
    # SPLIT trong MỘT đơn vị: chỉ dùng mảnh rỗng (VLM nói dòng đó không thuộc câu -- 14/10 mix1 m01: dòng thứ ba là tên máy FLUX
    # tự vẽ) -> dòng rời đơn vị, xoá trống (VLM đã nhìn dòng và nói không có chữ khách nào ở đó). Ngắt câu vào các dòng là việc
    # của code (bề rộng nét, layout); split chỉ còn cho chữ cong và phải nối lại đúng câu khách.
    for u in list(U):
        if u["split"] and len(u["split"]) == len(u["marks"]) and any(not p.strip() for p in u["split"]):
            out = [m for m, p in zip(u["marks"], u["split"]) if not p.strip()]
            u["marks"], u["split"] = [m for m, p in zip(u["marks"], u["split"]) if p.strip()], [p for p in u["split"] if p.strip()]
            U.append({**u, "id": u["id"] + "x", "kind": "skip", "marks": out, "clients": [], "split": None, "text_vlm": "", "vlm_out": True})
            log.append(f"{u['id']}: split để trống dòng {out} -> dòng đó rời đơn vị, xoá trống")
        if u["split"] and not u.get("split_ok") and (len(u["clients"]) != 1
                                                   or S._flat("".join(u["split"])) != S._flat(texts[u["clients"][0]]["text"])):
            u["split"] = None
    U = [u for u in U if u["marks"]]
    # MỘT CÂU Ở NHIỀU ĐƠN VỊ: đơn vị có chữ OCR giống câu nhất nhận câu; đơn vị kia: còn câu CÙNG VAI chưa có chỗ thì nhận câu đó
    # (FLUX vẽ lặp một hàng: 5f35 "Gội dưỡng" hai lần, "Đính đá" chưa có chỗ), không thì thành dòng lấp (chữ VLM nếu có)
    used = Counter(k for u in U for k in u["clients"])
    ocr = lambda u: " ".join(by[m].get("ocr") or "" for m in u["marks"])  # noqa: E731
    for k, n in list(used.items()):
        if n < 2:
            continue
        us = [u for u in U if k in u["clients"]]
        # các chỗ của câu XẾP CHỒNG liền nhau, cùng cỡ (FLUX vẽ một câu thành hai dòng, VLM ghi mỗi dòng một mục -- 14/10 dev2
        # d31: dòng dưới giữ chữ giun ngay dưới câu đã viết) -> một đơn vị nhiều dòng
        # (một hàng list FLUX vẽ lặp mà còn câu cùng vai chưa có chỗ thì không gộp: hàng lặp nhận câu đó, như dưới)
        if all(u["clients"] == [k] and u["kind"] == "text" for u in us) and not any(
                not used[j] and texts[j]["role"] == texts[k]["role"] for j in range(len(texts))):
            Ls = sorted((by[m] for u in us for m in u["marks"]), key=lambda l: (l["box"][1], l["box"][0]))
            if _stacked(Ls) and max(l["size"] for l in Ls) / min(l["size"] for l in Ls) <= SPLIT_RATIO:
                us[0].update(marks=[l["id"] for l in Ls], split=None)
                U = [u for u in U if not any(u is x for x in us[1:])]
                used[k] = 1
                log.append(f"câu T{k}: {[u['id'] for u in us]} xếp chồng liền nhau -> một đơn vị {us[0]['marks']}")
                continue
        best = max(us, key=lambda u: (sim(u["text_vlm"], texts[k]["text"]), sim(ocr(u), texts[k]["text"])))
        for u in us:
            if u is best:
                continue
            free = [j for j in range(len(texts)) if not used[j] and texts[j]["role"] == texts[k]["role"]]
            u["clients"] = [j for j in u["clients"] if j != k]
            used[k] -= 1
            if free and not u["clients"]:
                u["clients"] = [free[0]]
                used[free[0]] += 1
                log.append(f"{u['id']}: câu T{k} đã ở {best['id']} -> nhận câu cùng vai chưa có chỗ T{free[0]}")
            else:
                u["role_vlm"] = u["role_vlm"] or texts[k]["role"]
                log.append(f"{u['id']}: câu T{k} đã ở {best['id']} -> dòng lấp" + (f" {u['text_vlm']!r}" if u["text_vlm"] else ""))
    # DÒNG LẤP (VLM tự viết, không câu khách): số không có trong câu khách nào là dữ kiện bịa -> không viết
    tnums = {n for t in texts for n in re.findall(r"\d+", t["text"])}
    for u in U:
        if u["kind"] == "text" and not u["clients"] and u["text_vlm"]:
            bad = [n for n in re.findall(r"\d+", u["text_vlm"]) if n not in tnums]
            if bad:
                log.append(f"{u['id']}: chữ lấp {u['text_vlm']!r} có số không có trong câu khách {bad} -> bỏ")
                u["text_vlm"] = ""
    # DÒNG CHƯA AI GHÉP (VLM bỏ sót / ghi text rỗng): cụm nét lạ, chữ trên sản phẩm, poster có ảnh sản phẩm -> giữ (logo Coca-Cola
    # đọc thành "eo.mg", 14/10 q03); còn lại -> xoá trống (chữ giun). KHÔNG đem câu khách còn thiếu ép vào (14/10: FLUX vẽ thiếu
    # câu thì chịu; ghép lỏng theo chữ OCR đặt "Tiết kiệm điện" -- một ưu điểm -- vào dòng tên máy dưới tiêu đề, m01).
    # VLM chỉ được XOÁ cụm nét lạ (orphan): dòng chữ thường VLM ghi skip cũng là dòng chưa ai ghép (không bỏ dòng nào nháp đã vẽ)
    empty = [u for u in U if u["kind"] == "text" and not u["clients"] and not u["text_vlm"]]
    U = [u for u in U if u not in empty]
    for u in U:
        if u["kind"] == "skip" and not u.get("vlm_out"):
            u["marks"] = [m for m in u["marks"] if by[m].get("orphan")]
    U = [u for u in U if u["marks"]]
    have = {m for u in U for m in u["marks"]}
    for l in (l for l in M["L"] if l["id"] not in have):
        base = {"id": f"g_{l['id']}", "marks": [l["id"]], "clients": [], "split": None, "text_vlm": "", "role_vlm": None, "bullet": False}
        if l.get("orphan") or l.get("printed") or product:
            U.append({**base, "kind": "keep", "why": "chưa ai ghép: cụm nét lạ / chữ sản phẩm -> giữ"})
            log.append(f"{l['id']}: chưa ai ghép ({l.get('ocr')!r}) -> giữ nguyên")
        else:
            U.append({**base, "kind": "skip", "why": "chưa ai ghép"})
            log.append(f"{l['id']}: chưa ai ghép ({l.get('ocr')!r}) -> xoá trống")
    log += order(U, M, texts)
    log += _markers(U, M, texts)
    missing =[j for j in range(len(texts)) if not used[j] and texts[j].get("source") != "ai"]
    return U, log, missing


def _markers(U: list[dict], M: dict, texts: list[dict]) -> list[str]:
    """SỐ THỨ TỰ FLUX VẼ TRƯỚC CÁC HÀNG (số trong vòng tròn / ô vuông đứng riêng một dòng, 14/10 dev2 d20 / d19): chữ giun số
    bị xoá thì để lại vòng tròn trống, giữ thì số sai (1, 4, 2, 3, 4). -> viết lại số: dòng chỉ 1-2 chữ số, ngay bên trái một
    hàng chữ đã viết câu khách (cùng hàng, khe <= 3 cỡ). Số = thứ tự câu khách nếu là các bước (brief "ordered"), không thì thứ
    tự đọc các hàng có số. Không phải các dòng số đứng riêng khác (giá, ngày)."""
    by = M["by"]
    rows = [u for u in U if u["kind"] == "text" and u["clients"]]
    found = []
    for u, m in [(u, m) for u in U if not (u["kind"] == "text" and (u["clients"] or u["text_vlm"])) for m in u["marks"]]:
        l = by[m]
        if not re.fullmatch(r"\s*\d{1,2}\s*[.)]?\s*", l.get("ocr") or ""):
            continue
        b, s = l["box"], l["size"]
        cy = (b[1] + b[3]) / 2
        nxt = [r for r in rows for m in r["marks"][:1] if by[m]["box"][1] <= cy <= by[m]["box"][3]
               and 0 <= by[m]["box"][0] - b[2] <= 3 * max(s, by[m]["size"])]
        if nxt:
            found.append((u, m, nxt[0]))
    if len(found) < 2:   # một số lẻ loi không phải đánh số hàng
        return []
    ordered = sorted(j for j, t in enumerate(texts) if t.get("ordered"))
    found.sort(key=lambda p: (by[p[1]]["box"][1], by[p[1]]["box"][0]))
    out = []
    for i, (u, m, r) in enumerate(found):
        k = r["clients"][0]
        n = 1 + ordered.index(k) if k in ordered else i + 1
        u["marks"] = [x for x in u["marks"] if x != m]
        U.append({"id": f"n_{m}", "kind": "text", "marks": [m], "clients": [], "split": None, "text_vlm": str(n),
                  "role_vlm": "step_no", "bullet": False, "why": "số thứ tự trước hàng"})
        out.append(f"{m}: số FLUX vẽ trước hàng {r['id']} -> viết lại '{n}'")
    U[:] = [u for u in U if u["marks"]]
    return out


def _reading(boxes: list) -> list[int]:
    """Thứ tự đọc: hàng (tâm dọc lệch < nửa cao) từ trên xuống, trong hàng trái -> phải."""
    idx = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    rows = []
    for i in idx:
        b = boxes[i]
        cy, h = (b[1] + b[3]) / 2, b[3] - b[1]
        if rows and abs(cy - rows[-1]["cy"]) < 0.5 * max(h, rows[-1]["h"]):
            rows[-1]["ids"].append(i)
        else:
            rows.append({"cy": cy, "h": h, "ids": [i]})
    return [i for r in rows for i in sorted(r["ids"], key=lambda i: boxes[i][0])]


def order(U: list[dict], M: dict, texts: list[dict]) -> list[str]:
    """CÁC BƯỚC theo thứ tự khách (người dùng chốt 14/10, kể cả lưới 2x2 và khi lệch hình minh hoạ FLUX vẽ): đơn vị viết đúng một
    câu "ordered" (brief: bước của poster hướng dẫn), xếp theo thứ tự đọc -> hoán CÂU giữa các đơn vị cho tăng dần. Khung, cỡ,
    kiểu của đơn vị giữ nguyên. Danh sách khác (ưu điểm, nút) để VLM gán theo ảnh."""
    by = M["by"]
    us = [u for u in U if u["kind"] == "text" and len(u["clients"]) == 1 and texts[u["clients"][0]].get("ordered")]
    if len(us) < 2:
        return []
    box = lambda u: [min(by[m]["box"][0] for m in u["marks"]), min(by[m]["box"][1] for m in u["marks"]),  # noqa: E731
                     max(by[m]["box"][2] for m in u["marks"]), max(by[m]["box"][3] for m in u["marks"])]
    seq = [us[i] for i in _reading([box(u) for u in us])]
    cur = [u["clients"][0] for u in seq]
    if cur == sorted(cur):
        return []
    for u, k in zip(seq, sorted(cur)):
        if u["clients"][0] != k:
            u["clients"], u["split"] = [k], None
    return [f"các bước: thứ tự đọc {['T%d' % k for k in cur]} -> theo thứ tự khách {['T%d' % k for k in sorted(cur)]}"]


# ---------------------------------------------------------------------------------------------------- hình
def ink_box(ink: np.ndarray, l: dict) -> list | None:
    """Hình bao NÉT Hi-SAM của dòng: mảng nét chạm vùng dòng (khung nới 0,15 cao) và >= nửa nằm trong khung; ít nét -> None."""
    import cv2
    H, W = ink.shape[:2]
    x0, y0, x1, y1 = l["box"]
    h = y1 - y0
    g = 0.15 * h
    X0, Y0, X1, Y1 = int(max(0, x0 - g)), int(max(0, y0 - g)), int(min(W, x1 + g)), int(min(H, y1 + g))
    sub = ink[Y0:Y1, X0:X1].astype(np.uint8)
    if X1 <= X0 or Y1 <= Y0 or sub.sum() < 0.03 * max(1.0, (x1 - x0) * h):
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    core = lab[max(0, int(y0) - Y0):int(y1) - Y0, max(0, int(x0) - X0):int(x1) - X0]
    inside = np.bincount(core.ravel(), minlength=n)
    keep = [k for k in range(1, n) if inside[k] >= 0.5 * st[k, cv2.CC_STAT_AREA]]
    if not keep:
        return None
    sel = st[keep]
    return [X0 + float(sel[:, 0].min()), Y0 + float(sel[:, 1].min()), X0 + float((sel[:, 0] + sel[:, 2]).max()),
            Y0 + float((sel[:, 1] + sel[:, 3]).max())]


def align_of(boxes: list, size: float) -> str | None:
    """Lề của các khung xếp chồng: tâm thẳng (<= CENTER_TOL cỡ) -> giữa; không thì mép lệch ít nhất (<= EDGE_TOL cỡ); một khung: None."""
    if len(boxes) < 2:
        return None
    sp = {"left": np.ptp([b[0] for b in boxes]), "right": np.ptp([b[2] for b in boxes]), "center": np.ptp([(b[0] + b[2]) / 2 for b in boxes])}
    # các dòng dài bằng nhau: thẳng tâm lẫn thẳng mép -- không nói được lề (14/10 mix1 t02: "Tối 2-4-6" / "Tối 3-5-7" kéo cả tập
    # tên khoá học căn trái thành căn giữa; m01 tiêu đề hai dòng gần bằng nhau căn trái thành giữa) -> None, để chỗ khác quyết
    if sp["center"] <= CENTER_TOL * size and min(sp["left"], sp["right"]) <= CENTER_TOL * size:
        return None
    if sp["center"] <= CENTER_TOL * size:
        return "center"
    best = min(sp, key=sp.get)
    return best if sp[best] <= EDGE_TOL * size else "center"


def lone_align(b: list, others: list, size: float) -> str:
    """Lề của MỘT dòng: thẳng mép trái / phải với một dòng ngay trên / dưới (khe <= 2 cỡ, chồng ngang) mà tâm lệch -> mép đó; không: giữa."""
    for o in others:
        if max(o[1] - b[3], b[1] - o[3]) > 2 * size or min(o[2], b[2]) - max(o[0], b[0]) <= 0:
            continue
        off = abs((o[0] + o[2]) / 2 - (b[0] + b[2]) / 2) > CENTER_TOL * size
        if off and abs(o[0] - b[0]) <= CENTER_TOL * size:
            return "left"
        if off and abs(o[2] - b[2]) <= CENTER_TOL * size:
            return "right"
    return "center"


def set_align(B: list, size: float) -> str:
    """Lề của TẬP: theo các cặp xếp chồng cùng cột; không có cặp nào (một hàng, lưới thưa, toả tròn) -> giữa."""
    pairs = [(a, b) for i, a in enumerate(B) for b in B[i + 1:] if min(a[2], b[2]) - max(a[0], b[0]) > 0.3 * min(a[2] - a[0], b[2] - b[0])]
    votes = Counter(x for x in (align_of([a, b], size) for a, b in pairs) if x)
    return votes.most_common(1)[0][0] if votes else "center"


def break_like(text: str, caps: list, meas=len) -> list[str]:
    """Chia câu thành len(caps) dòng theo SỨC CHỨA từng dòng nháp: caps = bề rộng nét / cỡ chữ (em), meas = bề rộng em của chữ
    theo font thật (fonts.advance); dòng cong: caps = số ký tự OCR, meas = len. Chọn cách ngắt cho dòng chật nhất đỡ chật nhất
    -- cả khối co theo dòng chật nhất, nên đây là cách giữ cỡ to nhất; hoà thì các dòng đều nhau hơn. (14/10 mix1: ngắt theo tỉ
    lệ số ký tự OCR của chữ giun ra "toàn / bộ / mỹ phẩm / chăm sóc da", "Thời gian của riêng / anh" -- chữ giun sai số ký tự,
    bề rộng nét thì không.)"""
    w = text.split()
    n = min(len(caps), len(w))
    if n <= 1:
        return [text]
    caps = caps[:n]
    # quy hoạch động: (từ đã dùng, số dòng) -> cách ngắt tốt nhất theo (chật nhất, tổng bình phương)
    best = {0: ((0.0, 0.0), [])}
    for j in range(n):
        nxt = {}
        for i, ((mx, sq), br) in best.items():
            for e in range(i + 1, len(w) - (n - 1 - j) + 1):
                r = meas(" ".join(w[i:e])) / max(1e-6, caps[j])
                v = ((max(mx, r), sq + r * r), br + [e])
                if e not in nxt or v[0] < nxt[e][0]:
                    nxt[e] = v
        best = nxt
    cuts = [0] + best[len(w)][1]
    return [" ".join(w[a:b]) for a, b in zip(cuts, cuts[1:])]


def _caps(u: dict) -> list[float]:
    """Sức chứa các dòng nháp của đơn vị: bề rộng nét / cỡ chữ (em)."""
    return [max(0.5, (b[2] - b[0]) / max(1.0, l["size"])) for b, l in zip(u["ib"], u["Ls"])]


def _gaps(ink: np.ndarray, b: list, parts: list[str], size: float) -> list[float] | None:
    """Ranh giới giữa các câu trên MỘT dòng nét: không ngưỡng -- kết hợp CHỮ (ranh giới dự kiến theo tỉ lệ số ký tự các câu) với NÉT
    Hi-SAM (các khe cột không nét): mỗi ranh giới chọn khe có điểm = bề rộng khe / (1 + khoảng cách tới chỗ dự kiến tính theo cỡ chữ)
    cao nhất -> tâm các khe, trái -> phải; dòng không có khe nào -> None."""
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    col = ink[max(0, y0):y1, max(0, x0):x1].any(axis=0)
    runs, i = [], 0
    while i < len(col):
        if not col[i]:
            j = i
            while j < len(col) and not col[j]:
                j += 1
            if i > 0 and j < len(col):
                runs.append((j - i, x0 + (i + j) / 2))
            i = j
        else:
            i += 1
    n = [len(p) + 1 for p in parts]
    want = [x0 + (x1 - x0) * sum(n[:k + 1]) / sum(n) for k in range(len(parts) - 1)]
    out = []
    for w in want:
        cand = [r for r in runs if r[1] not in out and (not out or r[1] > out[-1])]
        if not cand:
            return None
        out.append(max(cand, key=lambda r: r[0] / (1 + abs(r[1] - w) / max(1.0, size)))[1])
    return out


def _stacked(Ls: list) -> bool:
    """Các dòng (đã xếp theo thứ tự đọc) xếp chồng thành một khối: khe dọc <= STACK_GAP cỡ và chồng ngang, liên tiếp."""
    for a, b in zip(Ls, Ls[1:]):
        A, B = a["box"], b["box"]
        sz = max(a["size"], b["size"])
        if not (B[1] - A[3] <= STACK_GAP * sz and min(A[2], B[2]) - max(A[0], B[0]) > 0):
            return False
    return True


def _rows(Ls: list, boxes: list) -> list[list[int]]:
    """Gộp các dòng cùng hàng, sát nhau (mẩu OCR cắt của một dòng chữ) -> các nhóm chỉ số, theo thứ tự đọc."""
    groups = []
    for i in range(len(Ls)):
        b = boxes[i]
        for g in groups:
            c = boxes[g[-1]]
            sz = max(Ls[i]["size"], Ls[g[-1]]["size"])
            if min(b[3], c[3]) - max(b[1], c[1]) >= 0.5 * min(b[3] - b[1], c[3] - c[1]) and max(b[0] - c[2], c[0] - b[2]) <= ROW_GAP * sz:
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def _rgb(hx) -> tuple | None:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(hx or ""))
    return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)) if m else None


def _sanitize(css: str) -> tuple[str, str]:
    """Hiệu ứng VLM ghi cho một vai (CSS) -> (phần đặt ở khung, phần đặt ở span ôm chữ). Chỉ nhận bóng / viền / dải màu; bỏ url()."""
    keep = [x.strip() for x in str(css or "").split(";") if EFFECT_KEYS.match(x) and "url(" not in x.lower() and "expression" not in x.lower()]
    grad = [x for x in keep if re.match(r"\s*(-webkit-)?(background|text-fill-color)", x, re.I)]
    clip = any("clip" in x.lower() and "text" in x.lower() for x in grad)
    # CHỮ TÔ DẢI MÀU (background-clip:text) phải ở span ôm chữ: ở khung thì dải màu chỉ phủ trong khung, nét tràn ra trong suốt
    # (09/10 dev3 g14, probe b21a "An khang thịnh vượng" mất hai đầu)
    inner = ";".join(grad) if clip else ""
    outer = ";".join(x for x in keep if not (clip and x in grad))
    return outer, inner


def layout(U: list[dict], M: dict, ink: np.ndarray, styles: dict, accent: list, texts: list[dict]) -> tuple[list[dict], list[str]]:
    """Đơn vị chữ -> số đo + phần tử dựng (u["els"]): khung nét, cỡ, màu, lề, tập, chữ (nguyên văn, ngắt theo nháp). Sửa U tại chỗ."""
    from .fonts import advance
    from .render import HEAD, WEIGHT, font_key, op_geom, path_of
    by, log = M["by"], []
    med = float(np.median([l["size"] for l in M["L"]])) if M.get("L") else 0.0
    T = [u for u in U if u["kind"] == "text"]
    for u in T:
        u["Ls"] = sorted((by[m] for m in u["marks"]), key=lambda l: (l["box"][1], l["box"][0]))
        u["role"] = texts[u["clients"][0]]["role"] if u["clients"] else (u.get("role_vlm") or "fill")
        u["text"] = [texts[k]["text"] for k in u["clients"]] or [u["text_vlm"]]
        u["size"] = float(np.median([l["size"] for l in u["Ls"]]))
        u["bent"] = any(l.get("path") or abs(l.get("angle") or 0) >= 2 or l.get("vertical") for l in u["Ls"])
        u["ib"] = []
        for l in u["Ls"]:
            b = None if u["bent"] else ink_box(ink, l)
            u["ib"].append([b[0], min(b[1], l["box"][1]), b[2], max(b[3], l["box"][3])] if b else list(l["box"]))
        u["box"] = [min(b[0] for b in u["ib"]), min(b[1] for b in u["ib"]), max(b[2] for b in u["ib"]), max(b[3] for b in u["ib"])]
        u["color"] = next((l["color"] for l in u["Ls"] if l.get("color")), None)
        u["color_draft"] = u["color"]   # màu đo trên nét nháp (bộ kiểm tương phản: đúng màu nháp thì giữ, thêm quầng)
        szs = [l["size"] for l in u["Ls"]]
        # CÁCH DỰNG: "lines" = mỗi hàng nháp một phần tử (câu khác cỡ / có split / các dòng không chồng thành khối);
        # "block" = khối nhiều dòng (tự ngắt theo nháp); "line" = một dòng
        if u["bent"]:
            u["mode"] = "bent"
        elif len(u["Ls"]) > 1 and (u["split"] or max(szs) / min(szs) > SPLIT_RATIO or not _stacked(u["Ls"])):
            u["mode"] = "lines"
        elif len(u["Ls"]) > 1 or (u["Ls"][0]["box"][3] - u["Ls"][0]["box"][1]) > TALL * u["size"]:
            u["mode"] = "block"
        else:
            u["mode"] = "line"
    # MỘT DÒNG NHIỀU CÂU (OCR gộp tên món + giá vào một dòng, 14/10 h02): cắt khung nét tại các khe trống rộng nhất giữa các cụm nét
    # -> mỗi câu một đơn vị theo vai của nó (giá vào tập giá, tên vào tập tên); dòng không có khe thì giữ một phần tử
    for u in list(T):
        if u["mode"] != "line" or len(u["clients"]) < 2:
            continue
        cuts = _gaps(ink, u["ib"][0], [texts[k]["text"] for k in u["clients"]], u["size"])
        if not cuts:
            continue
        b, xs = u["ib"][0], [u["ib"][0][0]] + cuts + [u["ib"][0][2]]
        subs = []
        for i, k in enumerate(u["clients"]):
            # khung mẩu ôm đúng nét của nó (không ôm nửa khe): lề / cỡ của mẩu theo chữ nháp (14/10 mix1 t02: tên khoá học căn trái
            # thành căn giữa vì khung mẩu kéo tới giữa khe)
            col = np.flatnonzero(ink[max(0, int(b[1])):int(b[3]), max(0, int(xs[i])):int(xs[i + 1])].any(axis=0))
            o = max(0, int(xs[i]))
            sx0, sx1 = (float(o + col[0]), float(o + col[-1] + 1)) if len(col) else (xs[i], xs[i + 1])
            v = {**u, "id": f"{u['id']}c{i}", "clients": [k], "text": [texts[k]["text"]], "role": texts[k]["role"],
                 "ib": [[sx0, b[1], sx1, b[3]]], "split": None}
            v["box"] = v["ib"][0]
            subs.append(v)
        U[U.index(u):U.index(u) + 1] = subs
        T[T.index(u):T.index(u) + 1] = subs
        log.append(f"{u['id']}: một dòng {len(subs)} câu -> cắt tại khe nét x={[round(c) for c in cuts]}")
    # TẬP CÙNG VAI: cỡ nháp đôi một lệch <= SET_RATIO (gom theo cỡ tăng dần); dòng lấp có vai VLM ghi thì cũng vào tập
    sets = {}
    for role in {u["role"] for u in T if u["role"] != "fill"}:
        us = sorted((u for u in T if u["role"] == role and u["mode"] in ("line", "block")), key=lambda u: u["size"])
        grp = []
        for u in us:
            if grp and u["size"] / grp[0]["size"] <= SET_RATIO:
                grp.append(u)
            else:
                if len(grp) >= 2:
                    sets[f"{role}#{len(sets)}"] = grp
                grp = [u]
        if len(grp) >= 2:
            sets[f"{role}#{len(sets)}"] = grp
    for name, us in sets.items():
        s = float(np.median([l["size"] for u in us for l in u["Ls"]]))
        col = Counter(u["color"] for u in us if u["color"]).most_common(1)
        al = set_align([u["box"] for u in us], s)
        for u in us:
            u.update(set=name, size=s, align=al, **({"color": col[0][0]} if col else {}))
        log.append(f"tập '{name}': {[u['id'] for u in us]} cỡ {s:.0f}px, lề {al}")
    allb = [(u["id"], b) for u in T for b in u["ib"]]
    for u in T:
        if "align" not in u:
            u["align"] = align_of(u["ib"], u["size"]) or lone_align(u["box"], [b for i, b in allb if i != u["id"]], u["size"])
    # KIỂU THEO VAI (VLM không ghi font / độ đậm: font theo cỡ, độ đậm theo độ dày nét đo được); trong một TẬP: font / độ đậm dùng
    # nhiều nhất (h02: giá cắt ra từ dòng gộp tên + giá đo nét cả dòng -> đậm hơn các giá khác)
    for u in T:
        st = styles.get(u["role"]) or {}
        u["font"] = font_key(st.get("font")) or (st.get("font_display") if med and u["size"] >= HEAD * med else st.get("font_default")) or "bevietnam"
        sw = [l["stroke"] / l["size"] for l in u["Ls"] if l.get("stroke") and l.get("size")]
        u["weight"] = int(st.get("weight") if isinstance(st.get("weight"), (int, float)) else WEIGHT(float(np.median(sw))) if sw else 400)
    for us in sets.values():
        fk, wt = Counter((u["font"], u["weight"]) for u in us).most_common(1)[0][0]
        for u in us:
            u["font"], u["weight"] = fk, wt
    # MÀU: VLM ghi màu theo vai (nó thấy chữ trắng viền đen; phép đo trên chữ giun có viền dày ra màu VIỀN -- probe k05). Các dòng
    # cùng vai mà màu đo khác hẳn nhau (cùng vai trên hai nền: chữ đỏ trên trắng, chữ trắng trên đỏ -- probe ac10) -> màu đo từng dòng
    for role in {u["role"] for u in T}:
        us = [u for u in T if u["role"] == role]
        vc = str((styles.get(role) or {}).get("color") or "")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", vc):
            continue
        rgb = [_rgb(u["color"]) for u in us if _rgb(u["color"])]
        if len(rgb) >= 2 and max(float(np.linalg.norm(np.subtract(a, b))) for a in rgb for b in rgb) > DIFF_RGB:
            log.append(f"vai '{role}': màu đo các dòng khác hẳn nhau -> màu theo từng dòng, không theo vai")
            continue
        for u in us:
            u["color"] = vc
    for u in T:
        st = styles.get(u["role"]) or {}
        wt = u["weight"]
        case = {"upper": "uppercase", "lower": "lowercase", "title": "capitalize"}.get(str(st.get("case") or "").lower(), "none")
        tr = st.get("tracking")
        outer, inner = _sanitize(st.get("effect"))
        u["css"] = (f"font-weight:{int(wt)};font-style:{'italic' if st.get('italic') else 'normal'};text-transform:{case};"
                    + (f"letter-spacing:{float(tr):.3f}em;" if isinstance(tr, (int, float)) else "") + outer)
        u["inner"] = inner
        # bề rộng chữ theo font thật của đơn vị (ngắt dòng): chữ in hoa đo bản in hoa
        u["meas"] = lambda t, k=u["font"], up=case == "uppercase": advance(k, t.upper() if up else t)  # noqa: E731
        # chữ: nguyên văn; ngoặc kép FLUX vẽ quanh câu trích (OCR thấy dấu ngoặc ở đầu / cuối) -> giữ ngoặc (probe dc0d)
        first, last = (u["Ls"][0].get("ocr") or "").strip(), (u["Ls"][-1].get("ocr") or "").strip()
        if len(u["text"]) == 1 and u["role"] == "review_quote":   # chỉ câu trích (probe k03: mép cuộn ruy băng đọc ra dấu ngoặc)
            t = u["text"][0]
            if first[:1] in QUOTES and t[:1] not in QUOTES:
                t = "“" + t
            if last[-1:] in QUOTES and t[-1:] not in QUOTES:
                t = t + "”"
            u["text"] = [t]
        # SỐ BƯỚC FLUX vẽ trong dòng ("1. Tráng phin", mix1 i02): OCR gộp số vào khung, số bị gỡ cùng chữ -> đặt lại số theo THỨ TỰ
        # CÂU KHÁCH (FLUX hay đánh sai: 1, 3, 4), chỉ cho câu là bước (brief "ordered")
        if len(u["clients"]) == 1 and texts[u["clients"][0]].get("ordered") and re.match(r"\s*\d{1,2}\s*[.)]", first):
            n = 1 + sorted(j for j, t in enumerate(texts) if t.get("ordered")).index(u["clients"][0])
            u["text"] = [f"{n}. {u['text'][0]}"]
        u["accent"] = [a for a in accent if any(str(a.get("text")) in t for t in u["text"])]
    # PHẦN TỬ DỰNG
    for u in T:
        lead = "• " if u.get("bullet") and u["Ls"][0].get("dot") else ""
        if u["mode"] == "bent":
            g = op_geom({"kind": "text", "marks": u["marks"]}, M)
            if g is None:
                u["els"] = []
                continue
            pieces = u["split"] or (break_like(" ".join(u["text"]), [max(1, len(l.get("ocr") or "")) for l in u["Ls"]])
                                   if len(u["Ls"]) > 1 and not path_of({"kind": "text", "marks": u["marks"]}, M) else [" ".join(u["text"])])
            u["els"] = [{"id": u["id"], "box": g["rect"], "angle": g["angle"], "path": path_of({"kind": "text", "marks": u["marks"]}, M),
                         "lines": pieces, "size": u["size"], "multi": len(pieces) > 1}]
        elif u["mode"] == "lines":
            pieces = break_like(" ".join(u["text"]), _caps(u), u["meas"])
            rank = list(range(len(u["Ls"])))   # thứ tự trong câu của mẩu ở mỗi dòng
            # một câu VLM ghi thành nhiều mục rồi được gộp (m04 "Du lịch Mây Ngàn -" | "0283 777 8899"): mỗi dòng giữ mẩu của nó
            if u.get("split_ok") and len(u["split"]) == len(u["Ls"]):
                rank = [u["marks"].index(l["id"]) for l in u["Ls"]]
                pieces = [u["split"][i] for i in rank]
            if len(pieces) < len(u["Ls"]):   # câu ít chữ hơn số dòng: dồn vào các dòng đầu
                pieces += [""] * (len(u["Ls"]) - len(pieces))
            u["els"] = []
            szs = [l["size"] for l in u["Ls"]]
            even = max(szs) / min(szs) <= SPLIT_RATIO   # các hàng nháp vẽ gần cùng cỡ: một cỡ chung; khác cỡ hẳn: mỗi hàng cỡ của nó
            for gi, g in enumerate(_rows(u["Ls"], u["ib"])):
                txt = " ".join(p for p in (pieces[i] for i in sorted(g, key=lambda i: rank[i])) if p)
                if not txt:
                    continue
                b = [min(u["ib"][i][0] for i in g), min(u["ib"][i][1] for i in g), max(u["ib"][i][2] for i in g), max(u["ib"][i][3] for i in g)]
                sz = u["size"] if even else float(np.median([u["Ls"][i]["size"] for i in g]))
                # hàng mà khung OCR ôm nhiều dòng chữ (huy chương "vỗng / 2025" chung một khung, 14/10 e03): ngắt theo chiều cao
                n = int(round((b[3] - b[1]) / (1.25 * sz))) if b[3] - b[1] > TALL * sz else 1
                parts = break_like(txt, [1.0] * n, u["meas"]) if n > 1 else [txt]
                u["els"].append({"id": f"{u['id']}#{gi}", "box": b, "angle": 0.0, "path": None, "lines": parts, "size": sz,
                                 "multi": len(parts) > 1})
        elif u["mode"] == "block":
            if len(u["text"]) > 1:
                pieces = u["text"]
            elif len(u["Ls"]) > 1:
                pieces = break_like(u["text"][0], _caps(u), u["meas"])
            else:   # một khung OCR ôm nhiều dòng chữ: số dòng ước theo chiều cao
                n = max(1, int(round((u["box"][3] - u["box"][1]) / (1.25 * u["size"]))))
                pieces = break_like(u["text"][0], [1.0] * n, u["meas"])
            u["els"] = [{"id": u["id"], "box": u["box"], "angle": 0.0, "path": None, "lines": pieces, "size": u["size"], "multi": True}]
        else:
            u["els"] = [{"id": u["id"], "box": u["box"], "angle": 0.0, "path": None, "lines": [" ".join(u["text"]) if len(u["text"]) == 1
                         else " ".join(u["text"])], "size": u["size"], "multi": False}]
        for e in u["els"]:
            if lead and e is u["els"][0]:
                e["lines"][0] = lead + e["lines"][0]
            e["html"] = "<br>".join(_markup(x, u, texts) for x in e["lines"])
        u.pop("meas", None)
    return U, log


def _markup(text: str, u: dict, texts: list[dict]) -> str:
    """Một dòng chữ -> html: escape; từ nhấn (accent của VLM) -> span cỡ / màu / đậm; chữ tô dải màu -> span ôm chữ."""
    h = _html.escape(text)
    for a in u.get("accent") or []:
        t = _html.escape(str(a["text"]))
        if t not in h:
            continue
        sc = a.get("scale")
        st = (f"font-size:{float(sc):.2f}em;" if isinstance(sc, (int, float)) and 0.5 <= sc <= 3 else "") + \
             (f"color:{a['color']};" if re.fullmatch(r"#[0-9a-fA-F]{3,8}", str(a.get("color") or "")) else "") + \
             (f"font-weight:{int(a['weight'])};" if isinstance(a.get("weight"), (int, float)) else "")
        if st:
            h = h.replace(t, f'<span style="{st}">{t}</span>', 1)
    if u.get("inner"):
        h = f'<span style="display:inline-block;{_html.escape(u["inner"], quote=True)}">{h}</span>'
    return h


def public(U: list[dict]) -> list[dict]:
    """Đơn vị -> "ops" của plan.json (UI debug đọc id / kind / marks / html như lệnh cũ) + số đo dựng."""
    out = []
    for u in U:
        o = {"id": u["id"], "kind": u["kind"], "marks": u["marks"], "client": [f"T{k}" for k in u["clients"]] or None, "why": u.get("why") or ""}
        if u["kind"] == "text":
            o.update(html="<br>".join(_html.escape(x) for e in u.get("els") or [] for x in e["lines"]), role=u.get("role"),
                     set=u.get("set"), align=u.get("align"), size=round(u.get("size") or 0, 1), mode=u.get("mode"),
                     font=u.get("font"), weight=u.get("weight"), **({"split": u["split"]} if u.get("split") else {}))
        out.append(o)
    return out
