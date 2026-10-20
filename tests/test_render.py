"""Bộ kiểm sau khi dựng (render.check): chữ dính nhau trên cùng hàng, cách tính điểm của vòng sửa."""

from __future__ import annotations


def el(i, ink, size=30.0, unit=None, **kw):
    return {"id": i, "unit": unit or i, "ink": ink, "size": size, "over": 0, "shrink": 1.0, **kw}


def test_touching_on_one_row():
    from textfix.render.check import check_units
    a = el("u13c0", [100, 500, 260, 530])
    b = el("u13c1", [262, 502, 600, 530])          # "0322 111 333" sát "123 Hoàng Hoa Thám": khe 2 px < 0,3 x 30
    c = el("u13c2", [640, 500, 800, 530])          # cách 40 px: đủ khe
    errs = check_units([], [a, b, c])
    assert [(e["type"], e["ops"]) for e in errs] == [("touching", ["u13c0", "u13c1"])]


def test_touching_inside_one_unit():
    from textfix.render.check import check_units
    giam = el("u3#0", [100, 300, 220, 340], unit="u3")
    pct = el("u3#1", [222, 280, 360, 350], size=60.0, unit="u3")   # "Giảm" nhỏ | "20%" to, cùng một câu
    assert [e["type"] for e in check_units([], [giam, pct])] == ["touching"]


def test_stacked_lines_are_not_touching():
    from textfix.render.check import touch_gap
    assert touch_gap(el("a", [100, 100, 400, 130]), el("b", [100, 134, 400, 164])) is None   # hai hàng chồng nhau
    assert touch_gap(el("a", [100, 100, 400, 130], angle=12.0), el("b", [401, 100, 500, 130])) is None


def test_info_errors_do_not_decide_a_fix():
    from textfix.render.check import score
    assert score([{"type": "overflow"}, {"type": "touching"}]) == (1, 1)
    assert score([{"type": "touching"}, {"type": "touching"}]) < score([{"type": "too_small"}])
