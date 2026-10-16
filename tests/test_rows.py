"""Danh sách / bảng nháp vẽ và hàng thừa (compose.rows): nhận hàng bằng hình học, gỡ cả hàng, dàn đều bằng điểm ảnh."""

from __future__ import annotations

import numpy as np

from textfix.compose import rows
from textfix.compose.assign import resolve
from textfix.compose.parse import parse


def line(i: int, box: list, **kw) -> dict:
    return {"id": f"L{i}", "box": list(box), "poly": None, "ocr": f"chu {i}", "size": float(box[3] - box[1]), "color": "#000000",
            "angle": 0.0, **kw}


def page(*L, size=(800, 1000)) -> dict:
    return {"L": list(L), "size": list(size), "by": {l["id"]: l for l in L}}


def menu(n: int, y0: int = 100, pitch: int = 60) -> list[dict]:
    """n hàng tên món (lề trái x=100) | giá (lề trái x=500), cao 40."""
    out = []
    for r in range(n):
        y = y0 + r * pitch
        out += [line(2 * r + 1, [100, y, 100 + 120 + 30 * (r % 3), y + 40]), line(2 * r + 2, [500, y, 640, y + 40])]
    return out


def entry(marks, client=None, kind="text", text=None, role=None) -> dict:
    return {"lines": marks, "client": client, "kind": kind, "text": text, "role": role, "why": "test"}


def test_menu_columns_pair_into_table_rows():
    G = rows.lists(page(*menu(4)))
    assert len(G) == 1 and G[0]["cols"] == 2
    assert G[0]["rows"] == [["L1", "L2"], ["L3", "L4"], ["L5", "L6"], ["L7", "L8"]]


def test_two_stacked_lines_are_not_a_list():
    # tiêu đề + phụ đề cùng cỡ, cùng tâm: không phải danh sách (hàng thừa của danh sách mới gỡ cả hàng)
    assert rows.lists(page(line(1, [200, 100, 600, 140]), line(2, [220, 160, 580, 200]))) == []


def test_far_block_breaks_the_column():
    # ba hàng đều nhịp 60, dòng thứ tư cách 300: khối khác
    L = [line(1, [100, 100, 300, 140]), line(2, [100, 160, 300, 200]), line(3, [100, 220, 300, 260]), line(4, [100, 560, 300, 600])]
    G = rows.lists(page(*L))
    assert [g["rows"] for g in G] == [[["L1"], ["L2"], ["L3"]]]


TEXTS = [{"text": "Cơm sườn", "role": "label"}, {"text": "35.000đ", "role": "price"},
         {"text": "Canh khổ qua", "role": "label"}, {"text": "15.000đ", "role": "price"},
         {"text": "Trà đá", "role": "label"}, {"text": "3.000đ", "role": "price"}]


def test_surplus_row_is_removed_whole_without_filler():
    # FLUX vẽ 4 hàng cho 3 món: hàng 3 VLM ghi chữ lấp tên món + giá bịa -> gỡ CẢ HÀNG (tên lẫn giá), không viết chữ lấp
    M = page(*menu(4))
    A = parse({"assign": [entry(["L1"], "T0"), entry(["L2"], "T1"), entry(["L3"], "T2"), entry(["L4"], "T3"),
                          entry(["L5"], text="Canh rau muống", role="label"), entry(["L6"], text="3.000đ"),
                          entry(["L7"], "T4"), entry(["L8"], "T5")]})
    U, log, miss = resolve(A, M, TEXTS)
    kind = {m: u["kind"] for u in U for m in u["marks"]}
    assert kind["L5"] == kind["L6"] == "skip" and all(kind[f"L{i}"] == "text" for i in (1, 2, 3, 4, 7, 8))
    assert M["lists"][0]["surplus"] == [2] and not miss


def test_duplicate_price_goes_to_the_row_of_its_dish():
    # cơm tấm f1cf: VLM ghi "3.000đ" (T5) cho giá hàng Trà đá (L6) và cả giá hàng thừa (L8) -- hai hàng khác nhau không gộp
    # thành một khối; hàng có tên món liền kề (T4 Trà đá) nhận giá, hàng kia thừa -> gỡ cả hàng
    M = page(*menu(4))
    A = parse({"assign": [entry(["L1"], "T0"), entry(["L2"], "T1"), entry(["L3"], "T2"), entry(["L4"], "T3"),
                          entry(["L5"], "T4"), entry(["L6"], "T5"), entry(["L7"], text="Chè đậu xanh"), entry(["L8"], "T5")]})
    U, log, _ = resolve(A, M, TEXTS)
    by = {m: u for u in U for m in u["marks"]}
    assert by["L6"]["clients"] == [5] and by["L6"]["marks"] == ["L6"]
    assert by["L7"]["kind"] == by["L8"]["kind"] == "skip"


def pills(n: int, H: int = 600, W: int = 600, y0: int = 60, pitch: int = 110, h: int = 80) -> tuple[np.ndarray, list[dict]]:
    """Ảnh nền trắng, n pill xanh (viền đậm) mỗi pill một chấm đầu dòng đỏ và một dòng chữ (đã gỡ)."""
    img = np.full((H, W, 3), 245, np.uint8)
    L = []
    for r in range(n):
        y = y0 + r * pitch
        img[y:y + h, 80:520] = (40, 90, 160)
        img[y + 4:y + h - 4, 84:516] = (200, 220, 245)
        img[y + 34:y + 46, 110:122] = (220, 30, 30)
        L.append(line(r + 1, [140, y + 20, 420 + 20 * (r % 2), y + 60]))
    return img, L


def test_respace_removes_whole_pill_and_spaces_the_rest_evenly():
    # câu hỏi 16/10: FLUX vẽ 4 pill cho 3 món -> còn 3 pill NGUYÊN VẸN, cách đều trên khoảng nháp, không pill trống, không lệch
    img, L = pills(4)
    M = page(*L, size=(600, 600))
    texts = [{"text": t, "role": "list_item"} for t in ("Một", "Hai", "Ba")]
    A = parse({"assign": [entry(["L1"], "T0"), entry(["L2"], "T1"), entry(["L3"], text="Bốn bịa"), entry(["L4"], "T2")]})
    U, _, _ = resolve(A, M, texts)
    for u in U:
        u["els"] = [{"box": list(M["by"][m]["box"])} for m in u["marks"]]
    out, log = rows.respace(img, U, M)
    assert any("dàn đều 3 hàng" in l for l in log), log
    col = out[:, 82]   # cột qua viền trái các pill
    edge = np.flatnonzero((col == (40, 90, 160)).all(axis=1))
    runs = np.split(edge, np.flatnonzero(np.diff(edge) > 1) + 1)
    assert len(runs) == 3 and all(len(r) == 80 for r in runs)            # đúng 3 pill, mỗi pill cao 80 như nháp
    tops = [int(r[0]) for r in runs]
    assert tops[0] == 60 and tops[-1] + 80 == 60 + 3 * 110 + 80          # trên đúng khoảng nháp (pill đầu, mép dưới pill cuối)
    assert abs((tops[1] - tops[0]) - (tops[2] - tops[1])) <= 1           # cách đều
    dots = np.flatnonzero((out[:, 115] == (220, 30, 30)).all(axis=1))
    assert len(np.split(dots, np.flatnonzero(np.diff(dots) > 1) + 1)) == 3   # chấm đầu dòng đi theo hàng
    by = {m: u for u in U for m in u["marks"]}
    assert by["L4"]["els"][0]["box"][1] == tops[2] + 20                  # khung chữ dời theo pill của nó


def test_respace_refuses_when_rows_touch():
    # pill sát nhau (không khe nền giữa hai pill): chỗ cắt rơi vào viền -> không dời, chỉ gỡ hàng thừa
    img, L = pills(4, pitch=80)
    M = page(*L, size=(600, 600))
    texts = [{"text": t, "role": "list_item"} for t in ("Một", "Hai", "Ba")]
    A = parse({"assign": [entry(["L1"], "T0"), entry(["L2"], "T1"), entry(["L3"]), entry(["L4"], "T2")]})
    U, _, _ = resolve(A, M, texts)
    out, log = rows.respace(img, U, M)
    assert (out == img).all() and any("không dàn đều" in l for l in log), log
