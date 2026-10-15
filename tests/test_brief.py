"""Bước LLM đầu luồng, với LLM giả trả JSON dựng sẵn: mỗi test là một cam kết của textfix.brief."""

from __future__ import annotations

import pytest

from textfix.brief import make_brief
from textfix.brief.evidence import Evidence, fold
from textfix.brief.prompt import TOKEN_LIMIT, count_tokens


def fake_llm(texts, prompt_en, intent="poster", variants=(), overridden=()):
    resp = {"intent": intent, "texts": [{"text": t, "role": r, "source": s} for t, r, s in texts], "prompt_en": prompt_en,
            "variants": [{"name": n, "prompt_en": p} for n, p in variants], "overridden": list(overridden)}
    return lambda system, parts: resp


def brief(texts, prompt_en, prompt=None, form=None, **kw):
    n = len(kw.get("variants", ()))
    return make_brief(fake_llm(texts, prompt_en, **kw), prompt, form, n_variants=n)


def by_text(B):
    return {t["text"]: t for t in B["texts"]}


NAIL = "làm poster cho tiệm nail Hồng Nhung, khai trương ngày 5/11, giảm 30% tất cả dịch vụ sơn gel, màu hồng pastel"


# ---------------------------------------------------------------- đối chiếu lời khách

def test_fold_ignores_case_and_diacritics():
    assert fold("Khai Trương ĐẸP") == fold("khai truong dep")


def test_numbers_match_with_or_without_spaces():
    ev = Evidence(["sdt 0909 123 456, giá 185.000đ"])
    assert ev.unknown_facts("Hotline 0909123456") == []
    assert ev.unknown_facts("Giá 185.000đ") == []
    assert ev.unknown_facts("Giá 190.000đ") == ["190000"]


# ---------------------------------------------------------------- chốt danh sách chữ

def test_texts_found_without_quotes_are_user_texts():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user"), ("Khai trương ngày 5/11", "date", "user")],
              'A poster "Tiệm nail Hồng Nhung" with a ribbon "Khai trương ngày 5/11".', NAIL)
    assert [t["source"] for t in B["texts"]] == ["user", "user"]


def test_invented_fact_is_dropped_from_texts_and_prompt():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user"), ("Hotline 0909 123 456", "contact", "user")],
              'A poster "Tiệm nail Hồng Nhung". A card reads "Hotline 0909 123 456".', NAIL)
    assert "Hotline 0909 123 456" not in by_text(B)
    assert "0909" not in B["prompt_en"]


def test_llm_copy_error_snaps_back_to_user_wording_in_texts_and_prompt():
    B = brief([("Khánh trương ngày 5/11", "date", "user")], 'A ribbon "Khánh trương ngày 5/11".', NAIL)
    assert "Khai trương ngày 5/11" in by_text(B)
    assert '"Khai trương ngày 5/11"' in B["prompt_en"]


def test_llm_may_add_diacritics_to_unaccented_input():
    B = brief([("Bún bò Cô Ba", "headline", "user"), ("Giá 45.000đ", "price", "user")],
              'A sign "Bún bò Cô Ba", a tag "Giá 45.000đ".', "banner quan bun bo co ba, gia 45.000d")
    assert set(by_text(B)) == {"Bún bò Cô Ba", "Giá 45.000đ"}
    assert all(t["source"] == "user" for t in B["texts"])


def test_support_copy_kept_without_facts_or_promises():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user"), ("Đặt lịch ngay", "cta", "ai"),
               ("Giảm giá cực sốc", "offer", "ai"), ("Top 1 Sài Gòn", "label", "ai")],
              'A poster "Tiệm nail Hồng Nhung", a button "Đặt lịch ngay", a badge "Giảm giá cực sốc", "Top 1 Sài Gòn".', NAIL)
    assert by_text(B)["Đặt lịch ngay"]["source"] == "ai"
    assert "Giảm giá cực sốc" not in by_text(B) and "Top 1 Sài Gòn" not in by_text(B)
    assert "Giảm giá cực sốc" not in B["prompt_en"]


def test_support_copy_capped():
    ai = [(f"Dòng lợi ích chung {c}", "body", "ai") for c in "abcdefgh"]
    B = brief(ai, "A poster.", "poster tiệm bánh")
    assert sum(t["source"] == "ai" for t in B["texts"]) == 6


def test_promise_added_to_user_text_is_dropped():
    B = brief([("Giảm 30% tất cả dịch vụ sơn gel, tặng quà", "offer", "user")], "A poster.", NAIL)
    assert B["texts"] == [] or "tặng" not in B["texts"][0]["text"]


def test_user_quoted_string_and_form_fields_always_present():
    form = {"category": "promo", "title": "Sale hè", "phone": "0912 345 678"}
    B = brief([], "A summer poster.", 'thêm dòng "Chỉ 3 ngày"', form)
    assert {"Sale hè", "0912 345 678", "Chỉ 3 ngày"} <= set(by_text(B))
    assert '"Chỉ 3 ngày"' in B["prompt_en"]


def test_quoted_style_word_is_not_printed():
    B = brief([], "A Y2K-style t-shirt with a vintage aesthetic.", 'áo phong cách "Y2K"', {"category": "promo", "title": "Áo thun"})
    assert "Y2K" not in by_text(B)


def test_list_field_split_one_item_per_text_capitalized():
    form = {"category": "product_intro", "product_name": "Tai nghe AirX", "highlights": "Pin 40 giờ, Bluetooth 5.3, nặng 180g"}
    B = brief([("Tai nghe AirX", "headline", "user"), ("Pin 40 giờ, Bluetooth 5.3, nặng 180g", "body", "user")],
              'Headphones "Tai nghe AirX", features "Pin 40 giờ, Bluetooth 5.3, nặng 180g".', "tạo poster", form)
    items = [t["text"] for t in B["texts"] if t["role"] == "list_item"]
    assert items == ["Pin 40 giờ", "Bluetooth 5.3", "Nặng 180g"]
    assert '"Nặng 180g"' in B["prompt_en"] and "Pin 40 giờ, Bluetooth" not in B["prompt_en"]


def test_steps_are_ordered_list_items():
    B = brief([("Bước một rửa tay", "step", "user")], 'A guide "Bước một rửa tay".', "hướng dẫn: bước một rửa tay")
    assert B["texts"][0]["role"] == "list_item" and B["texts"][0]["ordered"]


def test_guide_form_steps_restored_as_ordered_steps():
    form = {"category": "guide", "title": "Cách pha cà phê phin", "step_1": "Tráng phin bằng nước sôi", "step_2": "cho 20g cà phê"}
    B = brief([("Cách pha cà phê phin", "headline", "user")], 'A guide poster "Cách pha cà phê phin".', None, form)
    steps = [t for t in B["texts"] if t.get("ordered")]
    assert [t["text"] for t in steps] == ["Tráng phin bằng nước sôi", "Cho 20g cà phê"]
    assert all(t["role"] == "list_item" for t in steps)


def test_star_rating_is_a_picture_not_text():
    B = brief([("5 sao", "label", "user")], 'A review card with "5 sao".', "ảnh feedback, hiển thị đánh giá 5 sao")
    assert B["texts"] == [] and "gold rating stars" in B["prompt_en"]


def test_prompt_wins_form_only_on_real_conflict():
    form = {"category": "promo", "discount": "Giảm 20%", "price": "185.000đ"}
    B = brief([("GIẢM 30%", "offer", "user")], 'A badge "GIẢM 30%".', "ghi GIẢM 30%, giá vẫn 185.000đ", form,
              overridden=["discount", "price"])
    assert "Giảm 20%" not in by_text(B)
    assert "185.000đ" in by_text(B)   # prompt nhắc lại đúng giá form -> không phải bị đè, dù LLM khai vậy


def test_plain_image_has_no_texts():
    B = brief([], "A cow in a field.", "ảnh một con bò", intent="image")
    assert B["mode"] == "image" and B["texts"] == []


# ---------------------------------------------------------------- hợp đồng của prompt

def test_every_text_quoted_exactly_once():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user"), ("Đặt lịch ngay", "cta", "ai")],
              'A poster "Tiệm nail Hồng Nhung". The sign repeats "Tiệm nail Hồng Nhung". Soft light.', NAIL)
    assert B["prompt_en"].count('"Tiệm nail Hồng Nhung"') == 1
    assert B["prompt_en"].count('"Đặt lịch ngay"') == 1


def test_missing_texts_inserted_after_first_sentence():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user")], "A pink salon poster. Soft light. Glossy nails.", NAIL)
    first, second = B["prompt_en"].split(". ")[:2]
    assert first == "A pink salon poster" and '"Tiệm nail Hồng Nhung"' in second


def test_single_quotes_and_curly_quotes_unified():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user")], "A poster 'Tiệm nail Hồng Nhung' with “Tiệm nail Hồng Nhung”.", NAIL)
    assert B["prompt_en"].count('"Tiệm nail Hồng Nhung"') == 1 and "'" not in B["prompt_en"]


def test_foreign_quote_removed_but_rest_of_sentence_kept():
    B = brief([("Bún bò Cô Ba", "headline", "user")], 'A rustic banner "Bún bò Cô Ba" and a tag "Giá 50.000đ", steam rising.',
              "banner quán bún bò cô ba")
    assert "50.000" not in B["prompt_en"] and '"Bún bò Cô Ba"' in B["prompt_en"] and "rustic" in B["prompt_en"]


def test_english_descriptor_in_quotes_is_unquoted():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user")], 'A poster "Tiệm nail Hồng Nhung" in a "before/after" layout.', NAIL)
    assert "before/after layout" in B["prompt_en"] and '"before/after"' not in B["prompt_en"]


def test_ratio_removed_outside_quotes_only():
    B = brief([("Mở cửa 8:30", "info", "user")], 'A 9:16 poster, "Mở cửa 8:30", 4k.', "poster mở cửa 8:30")
    assert "9:16" not in B["prompt_en"] and "4k" not in B["prompt_en"] and '"Mở cửa 8:30"' in B["prompt_en"]
    assert ", ." not in B["prompt_en"]


def test_long_prompt_fits_token_limit_keeping_texts():
    long = 'A poster "Tiệm nail Hồng Nhung". ' + " ".join(
        f"Sentence {i} describes soft pink light, glossy textures, pastel flowers and elegant decor." for i in range(60))
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user")], long, NAIL)
    assert count_tokens(B["prompt_en"]) <= TOKEN_LIMIT and '"Tiệm nail Hồng Nhung"' in B["prompt_en"]


def test_variants_get_the_same_contract():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user")], 'A poster "Tiệm nail Hồng Nhung".', NAIL,
              variants=[("dark", "A dark luxurious salon scene.")])
    assert '"Tiệm nail Hồng Nhung"' in B["variants"][0]["prompt_en"]


def test_non_object_answer_raises():
    with pytest.raises(ValueError):
        make_brief(lambda s, p: ["not", "an", "object"], "poster")
