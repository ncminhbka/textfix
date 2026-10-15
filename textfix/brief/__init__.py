"""LLM ĐẦU LUỒNG: yêu cầu người dùng (form và / hoặc prompt tự do) -> brief cho cả dây chuyền.

    B = make_brief(llm, prompt, form, design, product_image, n_variants, aspect)
    B = {mode: "poster" | "image", prompt_en, variants: [{name, prompt_en}], texts: [{text, role, source, ordered?}], log}

Một lời gọi LLM (lời dặn: instructions.py) trả về ý định, danh sách chữ, prompt FLUX và các hướng thiết kế. Code không tin lời
LLM tự khai mà kiểm lại theo bằng chứng:
  texts.py     chốt danh sách chữ: câu nào của khách, câu nào LLM viết thêm, câu nào bịa sự thật (evidence.py đối chiếu);
  prompt.py    sửa mọi prompt cho chữ trong ngoặc kép đúng bằng danh sách đã chốt và vừa giới hạn token của FLUX.
"""

from __future__ import annotations

from .instructions import ROLES, SCHEMA, SYSTEM, clean_form, request_message
from .prompt import compose, count_tokens, unify_quotes
from .texts import plan_texts

__all__ = ["make_brief", "SCHEMA", "SYSTEM", "ROLES", "count_tokens"]

NO_TEXT = " No text, letters, words or logos anywhere in the image."


def make_brief(llm, prompt: str | None = None, form: dict | None = None, design: dict | None = None,
               product_image: bool = False, n_variants: int = 0, aspect: str | None = None) -> dict:
    """llm(system, [message]) -> dict, thường là llm.json_call(..., schema=SCHEMA).
    n_variants: số hướng thiết kế khác (ảnh 1 = prompt_en, ảnh 2.. = các hướng; cùng danh sách chữ)."""
    raw_form = form or {}
    ordered_lists = raw_form.get("category") == "guide" or bool(raw_form.get("guide_steps"))
    form = clean_form(raw_form)
    raw = llm(SYSTEM, [request_message(prompt, form, design, product_image, n_variants, aspect)])
    if not isinstance(raw, dict):
        raise ValueError(f"LLM trả lời không phải JSON object: {type(raw).__name__}")
    intent = raw.get("intent") if raw.get("intent") in ("poster", "image") else "poster"
    main = unify_quotes(str(raw.get("prompt_en") or ""))
    variants = [{"name": " ".join(str(v.get("name") or "").split())[:60], "prompt_en": unify_quotes(str(v.get("prompt_en") or ""))}
                for v in (raw.get("variants") or []) if isinstance(v, dict) and str(v.get("prompt_en") or "").strip()]
    variants = variants[:max(0, int(n_variants))]

    plan = plan_texts(raw.get("texts"), prompt=prompt, form=form, overridden=raw.get("overridden") or [], prompt_en=main,
                      intent=intent, ordered_lists=ordered_lists)
    log = dict(plan.notes)
    main, notes = compose(main, plan.texts, plan.rewrites)
    log.update(notes)
    for v in variants:
        v["prompt_en"], vn = compose(v["prompt_en"], plan.texts, plan.rewrites)
        if vn:
            log.setdefault("variants", []).append({"name": v["name"], **vn})

    # không câu nào -> chỉ FLUX vẽ (bỏ gỡ chữ / VLM / dựng); LLM tưởng là poster thì dặn thêm "không chữ" để FLUX khỏi vẽ giun
    mode = "poster" if plan.texts else "image"
    log["intent"] = intent
    if mode == "image" and intent == "poster":
        main += NO_TEXT
        for v in variants:
            v["prompt_en"] += NO_TEXT
    return {"mode": mode, "prompt_en": main, "texts": plan.texts, "variants": variants, "log": log}
