"""LLM ĐẦU LUỒNG (một lời gọi): form / prompt người dùng -> prompt tiếng Anh cho FLUX.2 + CÂU KHÁCH.

  prompt_en  theo hướng dẫn nâng prompt chính thức của BFL (flux2/system_messages.py, SYSTEM_MESSAGE_UPSAMPLING_T2I): giữ chủ đề,
             thêm chi tiết hình, MỌI chữ trong ngoặc kép đúng nguyên văn tiếng Việt; không bịa giá / % / khuyến mãi / ngày / số
             điện thoại / địa chỉ / bảo hành; không ghi tỉ lệ / độ phân giải.
  texts      chữ poster phải có: CHỈ chữ của người dùng, nguyên văn, kèm vai trò (designer dùng để hiểu từng chỗ).
Code giữ cam kết: câu phải có NGUYÊN VĂN trong form / prompt (sai -> bỏ); chuỗi trong ngoặc kép của prompt người dùng luôn có mặt;
câu chưa nằm trong ngoặc kép của prompt_en thì nối vào cuối prompt_en (model chỉ vẽ được chữ nó thấy).
"""

from __future__ import annotations

import json
import re

ROLES = ["headline", "subheadline", "body", "price", "old_price", "offer", "date", "info", "contact", "cta", "list_item",
         "review_quote", "review_name", "label"]
TECH_KEYS = {"image_base64", "seed", "num_images", "aspect_ratio", "primary_color", "style_pref", "has_product_image"}
QUOTED = re.compile(r'"([^"]+)"')
STAR_TEXT = re.compile(r"^\d+(?:[.,]\d+)?\s*(?:sao|stars?)$", re.I)   # câu CHỈ ghi số sao ("5 sao")
CURLY = re.compile(r"“([^”]+)”")   # ngoặc kép cong (gõ trên điện thoại / Word)
# vai mặc định của trường form (thêm lại trường LLM bỏ sót)
FIELD_ROLE = {"title": "headline", "discount": "offer", "applied_product": "subheadline", "date_start": "date", "date_end": "date",
              "product_name": "headline", "product_desc": "subheadline", "highlights": "list_item", "price": "price",
              "opening_date": "date", "opening_promo": "offer", "booking_contact": "contact", "feedback_target": "label",
              "feedback_quote": "review_quote", "customer_name": "review_name", "feedback_highlights": "list_item",
              "special_offer": "offer", "job_position": "headline", "job_desc": "body", "apply_deadline": "date",
              "apply_method": "contact", "guide_steps": "list_item", "store_name": "info", "phone": "contact",
              "address": "contact", "website_link": "contact"}
# trường form là DANH SÁCH các ý cách nhau dấu phẩy (ví dụ mẫu của form: "Pin 40 giờ, Bluetooth 5.3, Nặng 180g") -> mã tách
# từng ý thành một câu list_item, không phụ thuộc LLM (12/10: LLM lúc tách lúc không; không tách thì VLM chia sai các hàng)
LIST_FIELDS = ("highlights", "feedback_highlights")
# CHỮ PHỤ TRỢ do LLM viết (source "ai", 12/10): người dùng điền form (sự thật: ưu đãi, giá, ngày, liên hệ) nhưng không viết
# tiêu đề / khẩu hiệu / nút kêu gọi / nhãn mục -> poster thiếu thành phần, FLUX tự vẽ chữ vô nghĩa cho đầy. LLM được viết các
# dòng đó, mã chặn mọi thứ là sự thật / cam kết thương mại:
AI_MAX = 4           # tối đa 4 câu phụ trợ một poster
AI_MAX_CHARS = 48    # mỗi câu <= 48 ký tự (tiêu đề ~7 chữ)
AI_BAN = re.compile(r"\d|[%@₫$€£]|https?:|www\.|\.(?:vn|com|net|org)\b", re.I)   # số (giá, %, ngày, SĐT, số liệu), tiền, liên kết
AI_BAN_WORDS = ("giảm", "tặng", "miễn phí", "khuyến mãi", "khuyến mại", "bảo hành", "cam kết", "hoàn tiền", "trả góp", "đổi trả",
                "free", "sale", "voucher", "số một", "tốt nhất", "rẻ nhất", "hàng đầu", "chính hãng", "uy tín nhất")
# khung ảnh (tỉ lệ người dùng chọn) bằng lời: LLM mặc định "vertical poster" cả khi khung ngang (09/10 dev2 d16, d32 16:9)
ORIENT = {"1:1": "a square canvas", "4:5": "a portrait (slightly tall) canvas", "3:4": "a portrait canvas",
          "9:16": "a tall vertical canvas (phone story)", "16:9": "a wide horizontal landscape canvas (web banner)",
          "4:3": "a landscape canvas", "2:3": "a tall portrait canvas"}

SYSTEM = """You are the copy planner and prompt engineer of a Vietnamese poster generator built on FLUX.2 by Black Forest Labs.
Input: the user's poster form (category + filled fields, JSON) and/or a free prompt. Output ONE JSON object with:

0. intent -- "poster" when the user wants words on the image: any quoted string, any filled form field, or a poster / banner / flyer / ad / menu / sign whose content (name, offer, price, date, slogan, ...) the user gave. "image" when the user only wants a picture (photo, illustration, artwork, wallpaper, ...) and gives no words to show. Vietnamese users say "ảnh" for posters too: "ảnh feedback ...", "ảnh quảng cáo ...", "ảnh khuyến mãi / giới thiệu / tuyển dụng / khai trương ...", "banner", "poster", "menu" are POSTERS even with no words given (you write the support copy, see 2); "ảnh một con bò", "cho tôi đứng cạnh con chó" are pictures. Never "image" when there is any user text to show. Most users are not prompt writers: a short vague prompt ("tạo poster", "làm banner cho quán tôi", "đẹp nhé") next to a filled form is "poster", and it never cancels the form -- every filled form field is still a text.
   For "image": texts = [], and prompt_en (and every variant) is a plain FLUX.2 image prompt following the same upsampling guidelines (subject, details, materials, lighting, mood, composition) WITHOUT poster layout and without adding any text; the rules about texts below do not apply.

1. prompt_en -- an English prompt for FLUX.2 that draws the WHOLE poster including its text. Follow the official FLUX.2 prompt-upsampling guidelines:
   - Strictly preserve the user's subject and intent. Convert the request into a detailed paragraph.
   - Add concrete visual specifics: subject, form, materials, textures, lighting (quality, direction, color), shadows, spatial relationships, environment, mood, and a clean professional poster layout (where the headline, price / offer, list and footer sit; flat graphic text, badges, pills or cards where suitable).
   - Text in images: put EVERY text of the poster in double quotes, EXACTLY as written in "texts" below (Vietnamese with full diacritics), and say its role, typographic style and its place (e.g. headline "..." in a bold condensed sans-serif at the top center). Never add any text that is not in "texts".
   - Put texts on calm areas, never on the main subject.
   - A star rating the user asks for ("đánh giá 5 sao", "review 5 sao") is a PICTURE: describe a row of gold stars; never make
     "5 sao" a text and never quote it.
   - Never invent commercial claims: no price, discount, promotional offer, deadline / date, phone number, address or warranty that the user did not give.
   - Never write aspect ratios, resolutions or sizes (e.g. 9:16, 4k, 1080p).
2. texts -- the texts the poster must show, each with "source". source "user" = the user's own words: copy every text EXACTLY character for character from the form values or from the quoted strings / explicit texts of the prompt (same case, accents, punctuation). Never invent, translate, shorten or rewrite text. Every filled content field appears in full (store name, phone, address, website included). The form fields "highlights" and "feedback_highlights" are lists: ALWAYS one text per comma-separated item (role list_item). A comma-separated list of highlights / features in the free prompt becomes one text per item ONLY if the user separated them; copy each item exactly, except that every list item (role list_item: a bullet / feature / step row) STARTS WITH A CAPITAL LETTER ("nhận phản hồi trực tiếp" -> "Nhận phản hồi trực tiếp"), in texts and inside the quotes of prompt_en alike.
   - role: one of """ + ", ".join(ROLES) + """ (old_price = an original price shown crossed out next to the new price).
   - SUPPORT COPY (source "ai"): most users fill the facts and nothing else, so the poster lacks its basic parts. You may
     WRITE up to 4 short Vietnamese texts yourself to complete it -- a headline when the user gave none (<= 7 words), one short
     tagline, a call-to-action button ("Đặt lịch ngay", "Xem thêm", "Liên hệ ngay"), a section label ("Điểm nổi bật", "Ưu đãi",
     "Khách hàng nói gì") -- each <= 5 words except the headline, natural for this business and category, with full diacritics.
     They carry NO facts and NO promises: never a number of any kind, price, discount, gift, "free", date, deadline, phone,
     address, link, name of a person / brand / shop / place, customer quote, rating, statistic, warranty, guarantee or
     superlative ("số 1", "tốt nhất"). Always Vietnamese ("Khai trương", never "Grand Opening"). A call-to-action never
     points to a contact the user did not give (no "Gọi điện đặt bàn" without a phone number: use "Ghé quán hôm nay").
     When the user gave no shop / brand name, the poster has none. Do not repeat what a user
     text already says. Only for intent "poster"; when the user's texts already make a complete poster, add none.
   - Prompt and form disagree (a different value for the same thing: discount 20% in the form, "GIẢM 30%" in the prompt; another
     headline): the PROMPT wins -- use the prompt's text, leave that form value out of texts and prompt_en, and put the form field
     name in "overridden". ONLY a real conflict overrides: the prompt must state another value for THAT SAME field. A prompt
     that adds something new ("ghi thêm ...", an extra offer, a scene description) overrides nothing: every filled form field
     it does not contradict stays in texts (a form price stays when the prompt only adds a combo discount).
3. variants -- ALTERNATIVE DESIGN DIRECTIONS for the same poster, exactly as many as asked below (empty list when none are asked).
   Each: name (2-4 words, e.g. "big type centered", "split photo / text", "card layout", "dark premium") and its own complete
   prompt_en following every rule of 1. Same subject, same product, same texts (every text in double quotes, exactly the same),
   same user style / color hints -- but a clearly DIFFERENT composition: layout (where the text block sits, centered / left /
   split / top band / framed card), scale of the subject, background treatment, typographic mood and palette within the hints.
   The directions must differ from prompt_en and from each other, and all must stay professional and suit the category.
Answer only the JSON object."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["intent", "prompt_en", "texts", "variants", "overridden"],
          "properties": {"intent": {"type": "string", "enum": ["poster", "image"]},
                         "overridden": {"type": "array", "items": {"type": "string"}},
                         "prompt_en": {"type": "string"},
                         "variants": {"type": "array", "items": {
                             "type": "object", "additionalProperties": False, "required": ["name", "prompt_en"],
                             "properties": {"name": {"type": "string"}, "prompt_en": {"type": "string"}}}},
                         "texts": {"type": "array", "items": {
                             "type": "object", "additionalProperties": False, "required": ["text", "role"],
                             "properties": {"text": {"type": "string"}, "role": {"type": "string", "enum": ROLES},
                                            "source": {"type": "string", "enum": ["user", "ai"]}}}}}}

# gợi ý thiết kế từ UI (không phải chữ)
STYLES = {"hien_dai": "modern, minimal, clean", "sang_trong": "luxury, premium, elegant", "nang_dong": "energetic, sporty, dynamic",
          "am_cung": "warm, cozy, friendly", "le_hoi": "festive, vibrant, colorful"}
CATEGORIES = {"promo": "promotion / advertising poster", "product_intro": "product introduction poster",
              "opening": "grand opening banner", "feedback": "customer feedback / testimonial poster",
              "recruitment": "recruitment / hiring poster", "guide": "step-by-step process / how-to guide poster"}


def _norm(s: str) -> str:
    return " ".join(s.split())


def make_brief(llm, prompt: str | None = None, form: dict | None = None, design: dict | None = None,
               product_image: bool = False, n_variants: int = 0, aspect: str | None = None) -> dict:
    """llm = llm.json_call(..., schema=SCHEMA). -> {mode, prompt_en, texts: [{text, role}], variants: [{name, prompt_en}], log}.
    mode: "poster" (có câu khách -> cả dây chuyền) | "image" (không chữ khách -> chỉ FLUX vẽ, trả nháp).
    n_variants: số hướng thiết kế KHÁC (người dùng chọn nhiều ảnh: ảnh 1 = prompt_en, ảnh 2.. = các hướng này; cùng chữ khách).
    Mỗi prompt (chính và từng hướng) qua cùng các bước giữ cam kết: bỏ tỉ lệ / độ phân giải, chữ khách thiếu ngoặc kép thì nối."""
    form = {k: v for k, v in (form or {}).items() if k not in TECH_KEYS and v not in (None, "", [], False)}
    if form.get("category") in CATEGORIES:
        form["category"] = CATEGORIES[form["category"]]
    user = []
    if form:
        user.append("Form (JSON):\n" + json.dumps(form, ensure_ascii=False, indent=1))
    if prompt:
        user.append("Prompt:\n" + prompt)
    hints = []
    st = (design or {}).get("style")
    if st and st != "auto":
        hints.append(f"visual style: {STYLES.get(st, st)}")
    if (design or {}).get("color"):
        hints.append(f"main brand color: {design['color']} (use it as the dominant accent color of the design)")
    if product_image:
        hints.append("the user uploaded a photo of their product, given to the image model as reference image 1: the poster must "
                     "feature exactly that product (say 'the product from image 1' in prompt_en)")
    if aspect in ORIENT:
        hints.append(f"canvas: {ORIENT[aspect]} -- compose the whole layout for this shape and describe it with words like "
                     "horizontal / vertical / square, never as a ratio")
    if hints:
        user.append("Design hints (not texts, never draw them as words):\n- " + "\n- ".join(hints))
    user.append(f"Alternative design directions asked (variants): {max(0, int(n_variants))}")
    raw = llm(SYSTEM, ["\n\n".join(user)])
    sources = [_norm(prompt or "")] + [_norm(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for k, v in form.items()
                                       if k != "category"]
    must = list(dict.fromkeys(QUOTED.findall(prompt or "") + CURLY.findall(prompt or "")))
    # giá trị khách (ô form, chuỗi trong ngoặc kép của prompt), dài trước: để nhận câu LLM ghép / bọc các giá trị đó
    vals = sorted({_norm(str(x)) for k, v in form.items() if k != "category" for x in (v if isinstance(v, list) else [v])}
                  | {_norm(q) for q in must}, key=len, reverse=True)
    vals = [v for v in vals if len(v) >= 3]
    out, log, seen = [], {"dropped": [], "added": []}, set()
    # PROMPT THẮNG (09/10, d23): trường form mà prompt ghi giá trị khác -> bỏ giá trị form khỏi câu khách, không thêm lại
    # PROMPT LUÔN THẮNG FORM (người dùng chốt 13/10): mọi ô form đều có thể bị prompt đè. Việc "có đè thật không" là của LLM --
    # lời dặn nói rõ prompt chỉ THÊM chữ thì không đè ô nào (13/10 accept1 f08: prompt "ghi thêm Mua 2 túi giảm 20.000đ", LLM
    # khai price + highlights bị đè -> giá 185.000đ mất). Mã chỉ chặn một ca chắc chắn sai: giá trị form có nguyên văn trong
    # prompt thì không thể là "bị prompt đè".
    over = {k for k in raw.get("overridden") or [] if k in form and k != "category"
            and not all(_norm(str(x)).casefold() in _norm(prompt).casefold() for x in (form[k] if isinstance(form[k], list) else [form[k]]))}         if prompt else set()
    gone_vals = {_norm(str(x)).casefold() for k in over for x in (form[k] if isinstance(form[k], list) else [form[k]])}
    if over:
        log["overridden"] = sorted(over)
    pe_raw = raw.get("prompt_en", "")
    pe_open = QUOTED.sub(" ", pe_raw).casefold()   # prompt_en ngoài các chuỗi trong ngoặc kép
    for t in raw.get("texts", []):
        text = _norm(t.get("text", ""))
        if not text or text in seen or text.casefold() in gone_vals:
            continue
        # không nguyên văn từ người dùng -> bỏ (không bịa chữ); so không phân biệt hoa / thường (08/10: "Khai trương cửa hàng ..."
        # viết hoa chữ đầu, prompt khách viết thường -> bị bỏ oan, tiêu đề mất khỏi câu khách -> VLM chép chữ nháp "TRƯỞNG")
        if not any(text.casefold() in s.casefold() for s in sources):
            role = t.get("role") if t.get("role") in ROLES else "body"
            rest = _strip_values(text, vals)
            # GHÉP các giá trị khách bằng dấu nối ("20/10 - 31/10" từ hai ô ngày, form1 f02: bị bỏ, hai ngày rời không lên poster):
            # bỏ hết giá trị khách mà không còn chữ / số nào -> vẫn là chữ khách
            if rest is not None and not any(ch.isalnum() for ch in rest):
                seen.add(text)
                out.append({"text": text, "role": role, "source": "user"})
                log.setdefault("joined", []).append(text)
                continue
            # không phải chữ khách: chỉ giữ khi LLM khai là chữ phụ trợ và phần LLM viết không chứa sự thật / cam kết nào (AI_BAN)
            why = _ai_reject(text if rest is None else rest, t, sum(1 for o in out if o.get("source") == "ai"), len(text))
            if why:
                log["dropped"].append(text)
                log.setdefault("ai_rejected", []).append(f"{text} ({why})")
                continue
            seen.add(text)
            # câu phụ trợ BỌC một giá trị khách ("Tuyển dụng Nhân viên bán hàng", f12): sự thật nằm trong đó -> giữ như chữ khách
            # (designer không được rút gọn / bỏ)
            out.append({"text": text, "role": role, "source": "ai" if rest is None else "user"})
            log.setdefault("ai_added" if rest is None else "ai_wraps_user", []).append(text)
            continue
        seen.add(text)
        out.append({"text": text, "role": t.get("role") if t.get("role") in ROLES else "body", "source": "user"})
    # SỐ SAO là hình, không phải chữ (12/10 run 38ef: prompt "hiển thị đánh giá 5 sao" -> LLM đưa "5 sao" vào texts -> poster có
    # hàng sao KÈM chữ "5 sao"): câu chỉ ghi số sao mà khách không đặt trong ngoặc kép / không điền ở form -> bỏ khỏi texts,
    # trong prompt đổi thành lời tả hàng sao
    for t in [t for t in out if STAR_TEXT.match(t["text"]) and t["text"] not in must
              and not any(_norm(str(v)).casefold() == t["text"].casefold() for v in form.values())]:
        out.remove(t)
        seen.discard(t["text"])
        n = re.match(r"\d+", t["text"]).group(0)
        raw = {**raw, "prompt_en": raw.get("prompt_en", "").replace(f'"{t["text"]}"', f"a row of {n} gold rating stars"),
               "variants": [{**v, "prompt_en": str(v.get("prompt_en") or "").replace(f'"{t["text"]}"', f"a row of {n} gold rating stars")}
                            for v in (raw.get("variants") or []) if isinstance(v, dict)]}
        log.setdefault("stars_not_text", []).append(t["text"])
    for q in must:   # chuỗi người dùng viết trong ngoặc kép mà LLM bỏ sót -> thêm lại nguyên văn (trừ khi đã tách thành các ý)
        qn = _norm(q)
        parts = [t for t in seen if t.casefold() in qn.casefold()]   # mục danh sách đã viết hoa chữ đầu vẫn là một phần của chuỗi
        if qn not in seen and not (parts and sum(len(p) for p in parts) >= 0.8 * len(qn)):
            # LLM dùng chuỗi đó làm TỪ TẢ trong prompt_en (không ngoặc kép): khách đặt ngoặc cho phong cách, không phải chữ in
            # (09/10 dev2 d06: phong cách "Y2K", "vintage" -> "Y2K-style t-shirt with a vintage aesthetic") -> không thêm lại
            if re.search(r"(?<!\w)" + re.escape(qn.casefold()) + r"(?!\w)", pe_open):
                log.setdefault("quoted_as_style", []).append(qn)
                continue
            out.append({"text": qn, "role": "body", "source": "user"})
            seen.add(qn)
            log["added"].append(qn)
    # TRƯỜNG FORM đã điền mà không có trong câu khách (LLM viết lại thành câu khác -- "Apply deadline: 15/12/2026" -- nên bị bỏ ở
    # bước nguyên văn, 09/10 dev2 d32 mất hạn nộp) -> thêm nguyên giá trị trường
    for k, v in form.items():
        if k == "category" or k in over:
            continue
        for x in (v if isinstance(v, list) else [v]):
            xn = _norm(str(x))
            if not xn or any(xn.casefold() in t.casefold() for t in seen):
                continue
            parts = [t for t in seen if t.casefold() in xn.casefold()]
            if parts and sum(len(p) for p in parts) >= 0.8 * len(xn):   # đã tách thành các ý / các dòng
                continue
            out.append({"text": xn, "role": FIELD_ROLE.get(k, "info"), "source": "user"})
            seen.add(xn)
            log.setdefault("form_added", []).append(xn)
    # TRƯỜNG DANH SÁCH của form: mỗi ý một câu list_item, đúng thứ tự người dùng viết; bỏ câu gộp cả trường và các mảnh LLM tách
    # kiểu khác (gộp hai ý, cắt giữa ý); trong prompt chuỗi gộp trong ngoặc kép -> từng ý trong ngoặc kép
    pe_raw2, var_raw = raw.get("prompt_en", ""), [dict(v) for v in (raw.get("variants") or []) if isinstance(v, dict)]
    for k in LIST_FIELDS:
        if k not in form or k in over or not isinstance(form[k], str):
            continue
        whole = _norm(form[k])
        items = [x for x in (_norm(y) for y in re.split(r",\s+|;\s*|\n+", form[k])) if x]
        if len(items) < 2:
            continue
        at = next((i for i, t in enumerate(out) if t["text"].casefold() in whole.casefold() and t.get("source") != "ai"), len(out))
        out = [t for t in out if t.get("source") == "ai" or t["text"].casefold() not in whole.casefold()]
        at = min(at, len(out))
        out[at:at] = [{"text": x, "role": "list_item", "source": "user"} for x in items]
        seen |= set(items)
        each = ", ".join(f'"{x}"' for x in items)
        pe_raw2 = pe_raw2.replace(f'"{whole}"', each).replace(f'"{form[k].strip()}"', each)
        for v in var_raw:
            v["prompt_en"] = str(v.get("prompt_en") or "").replace(f'"{whole}"', each).replace(f'"{form[k].strip()}"', each)
        log.setdefault("list_split", []).append(items)
    raw = {**raw, "prompt_en": pe_raw2, "variants": var_raw}
    # ẢNH THƯỜNG: LLM nói "image" và không có chữ khách nào -> bỏ chữ phụ trợ TRƯỚC khi sửa prompt (không tự thành poster,
    # không nối chữ vào prompt ảnh)
    if raw.get("intent") == "image" and not any(t.get("source") != "ai" for t in out):
        out = []
    # MỤC DANH SÁCH VIẾT HOA CHỮ ĐẦU (12/10), rồi chữ trong ngoặc kép của prompt lấy đúng cách viết của câu trong texts (LLM viết
    # "Nặng 180g", khách viết "nặng 180g": so đúng từng chữ thì coi là thiếu -> nối thêm lần nữa vào cuối prompt)
    for t in out:
        c = _cap_first(t["text"])
        if t.get("role") == "list_item" and c != t["text"]:
            t["text"] = c
            log.setdefault("list_capitalized", []).append(c)
    raw = {**raw, "prompt_en": _requote(raw.get("prompt_en", ""), out),
           "variants": [{**v, "prompt_en": _requote(str(v.get("prompt_en") or ""), out)} for v in (raw.get("variants") or [])
                        if isinstance(v, dict)]}
    pe, miss = _fix_prompt(raw.get("prompt_en", ""), out)
    gone = _quoted_extra(raw.get("prompt_en", ""), out)
    if gone:
        log["quoted_removed"] = gone
    if miss:
        log["quoted_added"] = miss
    variants = []
    for v in (raw.get("variants") or [])[:max(0, int(n_variants))]:
        if isinstance(v, dict) and (v.get("prompt_en") or "").strip():
            vp, vm = _fix_prompt(v["prompt_en"], out)
            variants.append({"name": " ".join(str(v.get("name") or "").split())[:60], "prompt_en": vp})
            if vm:
                log.setdefault("variant_quoted_added", []).append(vm)
    # ẢNH THƯỜNG (09/10): không có câu khách nào -> chỉ FLUX vẽ, bỏ xoá / ô / VLM / dựng (trước đây báo lỗi "không tìm ra chữ").
    # Có câu khách thì luôn là poster dù LLM nói "image" (chữ khách phải lên ảnh). LLM nói "poster" mà không có câu khách: prompt
    # có thể tả chỗ tiêu đề / nút -> FLUX vẽ chữ vô nghĩa, dặn thêm không chữ.
    intent = raw.get("intent") if raw.get("intent") in ("poster", "image") else "poster"
    mode = "poster" if out else "image"
    log["intent"] = intent
    if mode == "image" and intent == "poster":
        tail = " No text, letters, words or logos anywhere in the image."
        pe += tail
        variants = [{**v, "prompt_en": v["prompt_en"] + tail} for v in variants]
    return {"mode": mode, "prompt_en": pe, "texts": out, "variants": variants, "log": log}


def _requote(pe: str, texts: list[dict]) -> str:
    """Chuỗi trong ngoặc kép của prompt trùng một câu trong texts khi không phân biệt hoa / thường -> viết đúng như câu đó."""
    by = {t["text"].casefold(): t["text"] for t in texts}
    return QUOTED.sub(lambda m: '"' + by.get(_norm(m.group(1)).casefold(), m.group(1)) + '"', pe)


def _strip_values(text: str, vals: list[str]) -> str | None:
    """Bỏ các giá trị khách (không phân biệt hoa thường) khỏi câu -> phần còn lại; None nếu câu không chứa giá trị nào."""
    rest, hit = text, False
    for v in vals:
        r2 = re.sub(re.escape(v), " ", rest, flags=re.I)
        hit, rest = hit or r2 != rest, r2
    return rest if hit else None


def _ai_reject(text: str, t: dict, n_have: int, full_len: int | None = None) -> str | None:
    """Câu (hoặc phần LLM tự viết của câu) có được giữ làm chữ phụ trợ không. -> lý do bỏ, hoặc None (giữ)."""
    if t.get("source") != "ai":
        return "không nguyên văn của khách"
    if n_have >= AI_MAX:
        return f"quá {AI_MAX} câu phụ trợ"
    if len(text.strip()) > AI_MAX_CHARS or (full_len or 0) > 2 * AI_MAX_CHARS:
        return "dài"
    if AI_BAN.search(text):
        return "có số / tiền / liên kết"
    low = " " + re.sub(r"[^\w\s]", " ", text.casefold()) + " "
    hit = next((w for w in AI_BAN_WORDS if f" {w} " in low), None)
    return f"có chữ cam kết / ưu đãi '{hit}'" if hit else None


def _cap_first(s: str) -> str:
    """Viết hoa chữ cái đầu tiên (bỏ qua ký hiệu / số đứng đầu: "- ý", "3 bước"); phần còn lại giữ nguyên."""
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:] if ch.islower() else s
        if ch.isdigit():
            return s
    return s


def _approved(q: str, texts: list[dict]) -> bool:
    """Chuỗi trong ngoặc kép của prompt_en là chữ khách: trùng một câu, nằm trong một câu, hoặc ghép từ các câu khách."""
    q = _norm(q).casefold()
    tx = [t["text"].casefold() for t in texts]
    if any(q in t for t in tx):
        return True
    parts = [t for t in tx if t in q]
    return bool(parts) and sum(len(t) for t in parts) >= 0.8 * len(q)


def _sentences(pe: str) -> list[str]:
    """Tách câu, không tách trong ngoặc kép ("TP. HCM", "Mở cửa 7:00!")."""
    out, cur, inq = [], "", False
    for i, ch in enumerate(pe):
        cur += ch
        if ch == '"':
            inq = not inq
        elif ch in ".!?" and not inq and (i + 1 == len(pe) or pe[i + 1].isspace()):
            out.append(cur)
            cur = ""
    return out + ([cur] if cur.strip() else [])


def _quoted_extra(pe: str, texts: list[dict]) -> list[str]:
    return [q for q in dict.fromkeys(QUOTED.findall(pe)) if not _approved(q, texts)]


def _fix_prompt(pe: str, texts: list[dict]) -> tuple[str, list[str]]:
    """Không tỉ lệ / độ phân giải trong prompt -- chỉ NGOÀI ngoặc kép (chữ người dùng như "8:30" giữ nguyên); chữ khách chưa nằm
    trong ngoặc kép thì nối vào cuối (model chỉ vẽ được chữ nó thấy). Chữ trong ngoặc kép KHÔNG phải chữ khách (câu LLM bịa, đã
    bị bỏ khỏi texts) -> bỏ cả câu chứa nó: FLUX vẫn vẽ, VLM chép lại thành chữ không ai duyệt (08/10: địa chỉ bịa "123 Đường Số
    1, Quận 1, TP. HCM" bị bỏ khỏi texts nhưng còn trong prompt -> lên cả 4 poster)."""
    bad_q = set(_quoted_extra(pe, texts))
    if bad_q:
        keep = [s for s in _sentences(pe) if not set(QUOTED.findall(s)) & bad_q]
        pe = "".join(keep)
    parts = pe.split('"')
    bad = re.compile(r"\b(?:1:1|9:16|16:9|4:5|5:4|2:3|3:2|4:3|3:4|21:9)\b|\b\d{3,4}p\b|\b[248]k\b", re.I)
    pe = '"'.join(bad.sub("", p) if i % 2 == 0 else p for i, p in enumerate(parts))
    have = set(QUOTED.findall(pe))
    miss = [t["text"] for t in texts if t["text"] not in have]
    if miss:
        pe += " The poster shows exactly these texts: " + ", ".join(f'"{t}"' for t in miss) + "."
    return " ".join(pe.split()), miss
