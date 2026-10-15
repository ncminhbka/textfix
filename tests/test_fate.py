"""Số phận dòng (compose.fate) và hỏi lại dòng bỏ sót (compose.vlm.reask) với VLM giả: mỗi test là một luật."""

from __future__ import annotations

import numpy as np

from textfix.compose.parse import parse

from textfix.compose.assign import resolve
from textfix.compose import fate, vlm


def line(i: int, y: int, **kw) -> dict:
    return {"id": f"L{i}", "box": [100, y, 500, y + 40], "poly": None, "ocr": f"chu {i}", "size": 40.0, "color": "#000000",
            "angle": 0.0, **kw}


def page(*L) -> dict:
    return {"L": list(L), "size": [800, 800], "by": {l["id"]: l for l in L}}


TEXTS = [{"text": "Khai trương", "role": "headline"}, {"text": "Giảm 30%", "role": "offer"}]


def entry(marks, kind="text", client=None, text=None) -> dict:
    return {"lines": marks, "client": client, "kind": kind, "text": text, "why": "test"}


def fates(M, entries):
    A = parse({"assign": entries, "styles": {}})
    U, log, _ = resolve(A, M, TEXTS)
    return fate.table(U, M)


def test_vlm_skip_is_trusted():
    M = page(line(1, 100), line(2, 200))
    t = fates(M, [entry(["L1"], client="T0"), entry(["L2"], kind="skip")])
    assert t["L1"]["fate"] == "text" and t["L2"]["fate"] == "skip"


def test_omitted_line_is_kept():
    M = page(line(1, 100), line(2, 200))
    t = fates(M, [entry(["L1"], client="T0")])
    assert t["L2"]["fate"] == "keep" and "bỏ sót" in t["L2"]["why"]


def test_mentioned_line_without_words_is_wiped():
    # FLUX vẽ tiêu đề hai lần, VLM ghép cùng một câu vào cả hai: dòng thừa không còn chữ -> gỡ (không để chữ giun ở lại)
    M = page(line(1, 100, ocr="Khai truong"), line(2, 300, ocr="Khoi trung"))
    t = fates(M, [entry(["L1"], client="T0"), entry(["L2"], client="T0"), entry([], client="T1")])
    assert {t["L1"]["fate"], t["L2"]["fate"]} == {"text", "skip"}


def test_printed_line_is_kept_unless_client_text():
    M = page(line(1, 100, printed=True), line(2, 200, printed=True))
    t = fates(M, [entry(["L1"], kind="skip"), entry(["L2"], client="T0")])
    assert t["L1"]["fate"] == "keep" and t["L2"]["fate"] == "text"


def test_omitted_lists_unmentioned_lines():
    M = page(line(1, 100), line(2, 200), line(3, 300))
    A = parse({"assign": [entry(["L1"], client="T0"), entry(["L3"], kind="skip")]})
    assert fate.omitted(A, M) == ["L2"]


def test_reask_asks_fresh_and_keeps_only_asked_lines(monkeypatch):
    calls = []

    def call(system, parts, meta=None, fresh=False):
        calls.append((parts[0], fresh))
        return {"assign": [entry(["L2"], kind="keep"), entry(["L1"], client="T1")]}
    monkeypatch.setattr("textfix.compose.call.marked_image", lambda img, M: __import__("PIL.Image").Image.fromarray(img))
    M = page(line(1, 100), line(2, 200))
    img = np.zeros((64, 64, 3), np.uint8)
    A = vlm.reask(img, M, "prompt", TEXTS, call, ["L2"], base=img)
    assert [u["marks"] for u in A["assign"]] == [["L2"]] and A["assign"][0]["kind"] == "keep"
    assert calls[0][1] is True and "LEFT OUT LINES L2" in calls[0][0]


def test_line_without_words_is_asked_again_before_wiping():
    # pill thừa: VLM ghép câu T0 vào cả L1 lẫn L2; L2 không còn chữ -> hỏi lại (REFILL), VLM đưa chữ lấp -> viết, không xoá trống
    M = page(line(1, 100, ocr="Khai truong"), line(2, 300, ocr="Khoi trung"))
    A = parse({"assign": [entry(["L1"], client="T0"), entry(["L2"], client="T0")], "styles": {}})
    U, _, _ = resolve(A, M, TEXTS)
    empty = fate.no_words(U)
    assert empty and empty[0] in ("L1", "L2")
    fate.merge(A, parse({"assign": [entry(empty, text="Ưu đãi hấp dẫn")]}), empty)
    U, _, _ = resolve(A, M, TEXTS)
    t = fate.table(U, M)
    assert t["L1"]["fate"] == "text" and t["L2"]["fate"] == "text"


def test_empty_split_line_beside_its_row_stays():
    # final2 g18: "Giảm" | "50%" cùng hàng, VLM dồn cả câu vào ô "50%" và để trống ô "Giảm"; "mow" là mẩu OCR trùng trong dòng dưới
    M = page({**line(1, 820), "box": [209, 832, 387, 922], "ocr": "Giam"}, {**line(2, 820), "box": [385, 822, 563, 922], "ocr": "5%"},
             {**line(3, 920), "box": [215, 921, 554, 977], "ocr": "moi dich vu"}, {**line(4, 935), "box": [223, 935, 324, 970], "ocr": "mow"})
    texts = [{"text": "Giảm 50% mọi dịch vụ", "role": "offer"}]
    A = parse({"assign": [{"lines": ["L1", "L2", "L3", "L4"], "client": "T0", "kind": "text",
                                "split": ["", "Giảm 50%", "mọi dịch vụ", ""]}]})
    U, _, _ = resolve(A, M, texts)
    t = fate.table(U, M)
    assert t["L1"]["fate"] == "text" and t["L4"]["fate"] == "skip"
