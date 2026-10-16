"""BƯỚC 4c: SỐ PHẬN MỖI DÒNG -- text (viết câu khách / chữ lấp), keep (giữ điểm ảnh nháp), skip (gỡ trống). Một bảng, mỗi dòng một
lý do (plan.json "fates").

Luật, theo thứ tự (người dùng chốt 15/10; 16/10: không hỏi lại VLM):
  1. VLM ghép câu khách / chữ lấp vào dòng -> text (đơn vị do bước ghép, compose.assign.resolve, quyết);
  2. chữ in trên sản phẩm khách tải (lines.mark_printed) -> keep, trừ khi VLM ghép câu khách (compose.assign.resolve);
     mẩu nằm trong khung một dòng khác theo số phận dòng đó (compose.assign.inside);
  3. VLM nói keep / skip -> theo VLM: TIN VLM khi nó xoá;
  4. VLM CÓ nhắc dòng mà không còn chữ cho nó (câu đã có chỗ khác, chữ lấp bị bỏ) -> skip: chữ giun của dòng thừa không được ở
     lại. Dòng đó là hàng của danh sách / bảng -> gỡ CẢ HÀNG, dàn đều các hàng còn lại (compose.rows);
  5. VLM KHÔNG nhắc dòng (khuôn JSON bắt buộc đủ dòng nên chỉ còn khi VLM không theo khuôn) -> keep: không biết dòng là gì thì
     không xoá.
"""

from __future__ import annotations


def mentioned(A: dict) -> set:
    """Các dòng câu trả lời của VLM có nhắc tới (mọi số phận)."""
    return {m for u in A.get("assign") or [] for m in u.get("marks") or []}


def leftovers(U: list[dict], M: dict, seen: set) -> tuple[list[dict], list[str]]:
    """Luật 4 + 5 cho các dòng chưa thuộc đơn vị nào sau bước ghép. -> (đơn vị thêm, nhật ký)."""
    have = {m for u in U for m in u["marks"]}
    out, log = [], []
    for l in M["L"]:
        if l["id"] in have:
            continue
        base = {"id": f"g_{l['id']}", "marks": [l["id"]], "clients": [], "split": None, "text_vlm": "", "role_vlm": None}
        if l.get("printed"):
            why = "chữ in trên sản phẩm khách tải"
            out.append({**base, "kind": "keep", "why": why})
        elif l["id"] in seen:
            why = "VLM có nhắc nhưng không còn chữ cho dòng (câu đã có chỗ khác / chữ lấp bị bỏ)"
            out.append({**base, "kind": "skip", "why": why, "no_words": True})
        else:
            why = "VLM bỏ sót"
            out.append({**base, "kind": "keep", "why": why})
        log.append(f"{l['id']} ({l.get('ocr')!r}): {why} -> {out[-1]['kind']}")
    return out, log


def table(U: list[dict], M: dict) -> dict:
    """{id dòng: {fate, unit, why}} cho mọi dòng."""
    out = {}
    for u in U:
        fate = "text" if u["kind"] == "text" else u["kind"]
        for m in u["marks"]:
            out[m] = {"fate": fate, "unit": u["id"], "why": str(u.get("why") or "")[:160]}
    return {l["id"]: out.get(l["id"], {"fate": "keep", "unit": None, "why": "không đơn vị nào"}) for l in M["L"]}
