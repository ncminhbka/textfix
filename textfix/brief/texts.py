"""Chốt danh sách chữ của poster từ đề xuất của LLM.

Mỗi câu LLM đề xuất được xếp vào một trong ba loại, theo bằng chứng chứ không theo lời LLM tự khai:
  user  mang chữ / sự thật của khách -> bắt buộc lên poster, các bước sau không được bỏ hay viết lại;
  ai    chữ phụ trợ LLM tự viết -> chỉ giữ khi không mang sự thật hay lời hứa nào; các bước sau được rút gọn / bỏ;
  bỏ    có số / liên kết khách không viết, hoặc thêm lời hứa vào câu khách.
Sau đó các bảo đảm không phụ thuộc LLM: chuỗi khách đặt trong ngoặc kép và ô form đã điền luôn có mặt, trường danh sách tách
mỗi ý một câu, số sao là hình chứ không phải chữ.
Câu nào đổi cách viết thì ghi một cặp thay thế cho prompt_en (`rewrites`), để prompt vẽ đúng các câu đã chốt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .evidence import Evidence, fold
from .instructions import ROLES

USER_COVERAGE = 0.5   # câu "user" phải có >= nửa số chữ trong lời khách; ít hơn là LLM tự viết
PRESENT = 0.8         # chuỗi khách coi là đã có khi >= 80% chữ của nó và mọi số của nó nằm trong các câu đã chốt
SNAP = 0.85           # câu "user" giống >= 85% một đoạn khách cùng số chữ, khác chữ cái = LLM chép sai -> lấy đoạn khách

# chữ phụ trợ (source "ai"): số lượng, độ dài, và những gì không được mang
AI_MAX = 6
AI_MAX_CHARS = 56
AI_BAN = re.compile(r"\d|[%@₫$€£]|https?:|www\.|\.(?:vn|com|net|org)\b", re.I)
PROMISE_WORDS = ("giảm", "tặng", "miễn phí", "khuyến mãi", "khuyến mại", "bảo hành", "cam kết", "hoàn tiền", "trả góp", "đổi trả",
                 "free", "sale", "voucher", "số một", "tốt nhất", "rẻ nhất", "hàng đầu", "chính hãng", "uy tín nhất")

# trường form là danh sách ý cách nhau dấu phẩy ("Pin 40 giờ, Bluetooth 5.3, Nặng 180g"): mỗi ý một câu list_item
LIST_FIELDS = ("highlights", "feedback_highlights")
# vai của ô form khi phải thêm lại ô LLM bỏ sót
FIELD_ROLE = {"title": "headline", "discount": "offer", "applied_product": "subheadline", "date_start": "date", "date_end": "date",
              "product_name": "headline", "product_desc": "subheadline", "highlights": "list_item", "price": "price",
              "opening_date": "date", "opening_promo": "offer", "booking_contact": "contact", "feedback_target": "label",
              "feedback_quote": "review_quote", "customer_name": "review_name", "feedback_highlights": "list_item",
              "special_offer": "offer", "job_position": "headline", "job_desc": "body", "apply_deadline": "date",
              "apply_method": "contact", "guide_steps": "list_item", "store_name": "info", "phone": "contact",
              "address": "contact", "website_link": "contact"}
ORDINAL = re.compile(r"^\s*(?:(?:bước|step)\s*)?\d{1,2}\s*[.):\-–]\s*", re.I)   # "Bước 1: ", "1. ", "2) " đầu câu
STAR_TEXT = re.compile(r"^\d+(?:[.,]\d+)?\s*(?:sao|stars?)$", re.I)   # câu chỉ ghi số sao ("5 sao")
USER_QUOTES = re.compile(r'"([^"]+)"|“([^”]+)”')
QUOTED = re.compile(r'"([^"]+)"')


class Notes(dict):
    """Nhật ký các quyết định (ghi vào brief.json để soi lại): khoá -> danh sách."""

    def add(self, key: str, item) -> None:
        self.setdefault(key, []).append(item)


@dataclass
class TextPlan:
    texts: list[dict] = field(default_factory=list)                 # [{text, role, source, ordered?}]
    rewrites: list[tuple[str, str]] = field(default_factory=list)   # (đoạn trong prompt_en, đoạn thay)
    notes: Notes = field(default_factory=Notes)

    def has(self, text: str) -> bool:
        return any(t["text"] == text for t in self.texts)


def norm(s: str) -> str:
    return " ".join(str(s).split())


def form_values(form: dict) -> dict[str, list[str]]:
    """Ô nội dung của form -> các giá trị (ô danh sách giữ nguyên chuỗi; tách ở `split_list_fields`)."""
    return {k: [norm(x) for x in (v if isinstance(v, list) else [v]) if norm(x)] for k, v in form.items() if k != "category"}


def plan_texts(proposed: list, *, prompt: str | None, form: dict, overridden: list, prompt_en: str, intent: str,
               ordered_lists: bool) -> TextPlan:
    """proposed: texts LLM trả. prompt_en: prompt chính đã thống nhất ngoặc (để nhận chuỗi khách LLM dùng làm từ tả)."""
    prompt = norm(prompt or "")
    fields = form_values(form)
    passages = [prompt] + [x for vs in fields.values() for x in vs]
    plan = TextPlan()
    over = accepted_overrides(overridden, fields, prompt, proposed)
    if over:
        plan.notes["overridden"] = sorted(over)
    gone = {fold(x) for k in over for x in fields[k]}
    vet(proposed, Evidence(passages), passages, gone, plan)
    user_quotes = list(dict.fromkeys(next(g for g in m.groups() if g) for m in USER_QUOTES.finditer(prompt)))
    stars_to_pictures(plan, user_quotes, fields)
    restore_user_quotes(plan, user_quotes, prompt_en)
    restore_form_fields(plan, fields, over)
    split_list_fields(plan, form, over)
    if intent == "image" and not any(t["source"] == "user" for t in plan.texts):
        plan.texts = []   # ảnh thường: không tự thành poster vì vài câu phụ trợ
    finalize(plan, ordered_lists)
    return plan


# loại vai: giá trị thay thế cho một ô form phải cùng loại với ô đó
ROLE_KIND = {"headline": "title", "subheadline": "title", "label": "title", "offer": "offer", "price": "offer",
             "old_price": "offer", "date": "date", "contact": "contact", "info": "contact", "body": "body",
             "review_quote": "body", "review_name": "name", "list_item": "list", "step": "list", "cta": "cta"}


def accepted_overrides(claimed: list, fields: dict[str, list[str]], prompt: str, proposed: list) -> set[str]:
    """Prompt thắng form (người dùng chốt) -- khi prompt THẬT SỰ đưa giá trị khác cho ô đó. Ô k LLM khai bị đè chỉ được nhận khi:
      - giá trị ô không có ngay trong prompt (có thì prompt chỉ nhắc lại), và
      - có một câu LLM đề xuất lấy từ PROMPT (không từ ô form nào) cùng loại vai với ô k: giá trị thay thế.
    Prompt "làm banner giúp mình" không thay được tên tiệm, địa chỉ hay ngày nào."""
    if not prompt:
        return set()
    from_form, in_prompt = Evidence([x for vs in fields.values() for x in vs]), Evidence([prompt])
    kinds = set()
    for t in proposed or []:
        text = norm(t.get("text") or "") if isinstance(t, dict) and t.get("source") != "ai" else ""
        if text and in_prompt.coverage(text)[0] >= USER_COVERAGE and not from_form.holds(text, PRESENT):
            kinds.add(ROLE_KIND.get(t.get("role"), "body"))
    return {k for k in claimed or [] if k in fields and not all(fold(x) in fold(prompt) for x in fields[k])
            and ROLE_KIND.get(field_role(k), "body") in kinds}


def vet(proposed: list, ev: Evidence, passages: list[str], gone: set[str], plan: TextPlan) -> None:
    """Xếp từng câu LLM đề xuất: user / ai / bỏ (xem đầu tệp)."""
    n_ai = 0
    for t in proposed or []:
        if not isinstance(t, dict):
            continue
        text, role = norm(t.get("text") or ""), t.get("role") if t.get("role") in ROLES else "body"
        if role in ("step", "list_item") and ORDINAL.match(text) and ORDINAL.sub("", text):
            # đánh số là việc của code (các bước theo thứ tự khách): bỏ số LLM tự viết, cả trong prompt
            plain = ORDINAL.sub("", text)
            plan.rewrites.append((f'"{text}"', f'"{plain}"'))
            plan.notes.add("ordinal_removed", text)
            text = plain
        claimed_user = t.get("source") != "ai"
        if not text or plan.has(text) or fold(text) in gone:
            continue
        bad = ev.unknown_facts(text)
        if bad:
            plan.notes.add("dropped", text)
            plan.notes.add("facts_not_from_user", f"{text} ({', '.join(bad)})")
            continue
        cov, extra = ev.coverage(text)
        if claimed_user and extra:
            fixed = snap_to_user(text, passages)
            if fixed:
                plan.notes.add("snapped", f"{text} -> {fixed}")
                plan.rewrites.append((f'"{text}"', f'"{fixed}"'))
                text = fixed
                cov, extra = ev.coverage(text)
        promise = promise_word(" ".join(extra))
        if claimed_user and cov >= USER_COVERAGE:
            if promise:
                plan.notes.add("dropped", text)
                plan.notes.add("ai_rejected", f"{text} (thêm '{promise}')")
                continue
            if extra:
                plan.notes.add("user_cleaned", text)
            source = "user"
        elif ev.facts(text) and not promise:
            plan.notes.add("ai_wraps_user", text)   # câu LLM viết quanh một sự thật của khách ("Ưu đãi 20%")
            source = "user"
        else:
            why = reject_support_copy(text, n_ai)
            if why:
                plan.notes.add("dropped", text)
                plan.notes.add("ai_rejected", f"{text} ({why})")
                continue
            n_ai += 1
            plan.notes.add("ai_added", text)
            source = "ai"
        plan.texts.append({"text": text, "role": role, "source": source})


def snap_to_user(text: str, passages: list[str]) -> str | None:
    """Đoạn khách cùng số chữ giống câu nhất, nếu giống >= SNAP và khác ở CHỮ CÁI (khác dấu / hoa thường là LLM chuẩn hoá, giữ).
    Đánh đổi: lỗi gõ chữ cái của chính khách cũng được giữ, vì không phân biệt được với LLM chép sai."""
    n, best = len(text.split()), (0.0, None)
    for s in passages:
        w = re.sub(r"[\"“”]", " ", s).split()
        for i in range(len(w) - n + 1):
            c = " ".join(w[i:i + n]).strip(" ,.;:!?\"'“”")
            r = SequenceMatcher(None, text.casefold(), c.casefold()).ratio()
            if r > best[0]:
                best = (r, c)
    r, c = best
    if not c or r < SNAP or fold(c) == fold(text):
        return None
    return cap_first(c) if text[:1].isupper() else c


def promise_word(s: str) -> str | None:
    low = " " + re.sub(r"[^\w\s]", " ", s.casefold()) + " "
    return next((w for w in PROMISE_WORDS if f" {w} " in low), None)


def reject_support_copy(text: str, n_have: int) -> str | None:
    """Lý do không nhận một câu phụ trợ, hoặc None."""
    if n_have >= AI_MAX:
        return f"quá {AI_MAX} câu phụ trợ"
    if len(text) > AI_MAX_CHARS:
        return "dài"
    if AI_BAN.search(text):
        return "có số / tiền / liên kết"
    w = promise_word(text)
    return f"có chữ cam kết / ưu đãi '{w}'" if w else None


def stars_to_pictures(plan: TextPlan, user_quotes: list[str], fields: dict[str, list[str]]) -> None:
    """"5 sao" là hàng sao vẽ, không phải chữ -- trừ khi khách viết đúng chuỗi đó trong ngoặc kép / ô form."""
    literal = {fold(x) for x in user_quotes} | {fold(x) for vs in fields.values() for x in vs}
    for t in [t for t in plan.texts if STAR_TEXT.match(t["text"]) and fold(t["text"]) not in literal]:
        plan.texts.remove(t)
        n = re.match(r"\d+", t["text"]).group(0)
        plan.rewrites.append((f'"{t["text"]}"', f"a row of {n} gold rating stars"))
        plan.notes.add("stars_not_text", t["text"])


def restore_user_quotes(plan: TextPlan, user_quotes: list[str], prompt_en: str) -> None:
    """Chuỗi khách đặt trong ngoặc kép luôn lên poster -- trừ khi LLM dùng nó làm TỪ TẢ trong prompt_en (khách đặt ngoặc cho
    phong cách: "Y2K", "vintage")."""
    outside = QUOTED.sub(" ", prompt_en).casefold()
    for q in map(norm, user_quotes):
        if Evidence([t["text"] for t in plan.texts]).holds(q, PRESENT):
            continue
        if re.search(r"(?<!\w)" + re.escape(q.casefold()) + r"(?!\w)", outside):
            plan.notes.add("quoted_as_style", q)
            continue
        plan.texts.append({"text": q, "role": "body", "source": "user"})
        plan.notes.add("added", q)


def restore_form_fields(plan: TextPlan, fields: dict[str, list[str]], over: set[str]) -> None:
    """Ô form đã điền (trừ ô bị prompt đè) luôn lên poster; LLM bỏ sót / viết thành câu khác thì thêm nguyên giá trị."""
    for k, values in fields.items():
        if k in over:
            continue
        for x in values:
            if not Evidence([t["text"] for t in plan.texts]).holds(x, PRESENT):
                plan.texts.append({"text": x, "role": field_role(k), "source": "user"})
                plan.notes.add("form_added", x)


def field_role(key: str) -> str:
    """Vai của một ô form. Các bước của form hướng dẫn đến từ API dưới dạng step_1, step_2, ... (server.app.form_of)."""
    return "step" if re.fullmatch(r"step_\d+", key) else FIELD_ROLE.get(key, "info")


def split_list_fields(plan: TextPlan, form: dict, over: set[str]) -> None:
    """Trường danh sách: mỗi ý một câu list_item, đúng thứ tự khách, thay mọi câu LLM tách / gộp kiểu khác từ trường đó."""
    for k in LIST_FIELDS:
        if k in over or not isinstance(form.get(k), str):
            continue
        whole = norm(form[k])
        items = [x for x in map(norm, re.split(r",\s+|;\s*|\n+", form[k])) if x]
        if len(items) < 2:
            continue
        inside = [i for i, t in enumerate(plan.texts) if t["source"] == "user" and fold(t["text"]) in fold(whole)]
        at = inside[0] if inside else len(plan.texts)
        plan.texts = [t for i, t in enumerate(plan.texts) if i not in inside]
        plan.texts[at:at] = [{"text": x, "role": "list_item", "source": "user"} for x in items]
        plan.rewrites.append((f'"{whole}"', ", ".join(f'"{x}"' for x in items)))
        plan.notes.add("list_split", items)


def finalize(plan: TextPlan, ordered_lists: bool) -> None:
    """Mục danh sách viết hoa chữ đầu; vai "step" = mục danh sách có thứ tự (các bước luôn theo thứ tự khách)."""
    for t in plan.texts:
        if t["role"] == "step":
            t["role"], t["ordered"] = "list_item", True
        elif ordered_lists and t["role"] == "list_item":
            t["ordered"] = True
        if t["role"] == "list_item":
            c = cap_first(t["text"])
            if c != t["text"]:
                plan.rewrites.append((f'"{t["text"]}"', f'"{c}"'))
                t["text"] = c
                plan.notes.add("list_capitalized", c)


def cap_first(s: str) -> str:
    """Viết hoa chữ cái đầu (bỏ qua ký hiệu đứng đầu: "- ý"); câu bắt đầu bằng số giữ nguyên ("3 bước")."""
    for i, ch in enumerate(s):
        if ch.isdigit():
            return s
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s
