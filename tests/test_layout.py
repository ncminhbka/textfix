"""Bố cục chữ (fill): cỡ theo font thật, nới khung vào nền trống, màu từ ba nguồn."""

from __future__ import annotations

import pytest


def test_font_heights_differ_by_family():
    from textfix.render.fonts import CAP_EM, X_EM, heights
    assert heights("anton")[0] > 0.8 and heights("allura")[0] < 0.62   # chữ hẹp cao / thư pháp thấp: không dùng một hằng chung
    assert heights("bungee")[1] == pytest.approx(heights("bungee")[0], abs=0.01)   # font toàn chữ hoa
    assert heights("khong-co") == (CAP_EM, X_EM)


def test_room_grows_into_free_background_only():
    import numpy as np
    from textfix.compose.size import room
    lab = np.zeros((100, 400, 3))
    lab[:, 290:] = (50, 40, 0)                 # mép vỏ / hình ở x >= 290
    blocked = np.zeros((100, 400), bool)
    blocked[:, :60] = True                     # dòng khác ở x < 60
    box = [100, 40, 260, 60]                   # rộng 160 -> nới tối đa 40
    assert room(box, "left", lab, blocked)[2] == 286          # chạm vỏ ở 290 (chưa hết 40): chừa khe 0,2 cao dòng (4 px)
    assert room(box, "right", lab, blocked)[0] == 60           # nới trái đủ 40 px (không chạm gì)
    c = room(box, "center", lab, blocked)
    assert c[0] == 80 and c[2] == 280                          # đều hai bên, mỗi bên 20


@pytest.mark.parametrize("glyph, ink, vlm, want", [
    ("#e8d9a0", "#e0d4a2", "#000000", "#e8d9a0"),   # chữ vàng trên đen, VLM ghi màu vai "đen": hai số đo đồng ý
    ("#fdfdfd", "#8a8a8a", "#ffffff", "#fdfdfd"),   # chữ trắng mảnh: lõi nét lẫn nền, đo khung + VLM đồng ý
    ("#c9a45a", "#fffdf5", "#ffffff", "#fffdf5"),   # chữ trắng trên nền vàng: khung đo ra nền, nét + VLM đồng ý
    (None, None, "#123456", "#123456"),
])
def test_color_needs_two_sources(glyph, ink, vlm, want):
    from textfix.compose.style import pick_color
    assert pick_color(glyph, ink, vlm)[0] == want


def test_row_of_mixed_sizes_reads_left_to_right():
    from textfix.compose.geometry import _reading
    giam, big = [100, 120, 180, 160], [190, 60, 420, 165]   # "Giảm" nhỏ | "50%" to, cùng hàng: mép trên của chữ to cao hơn
    under = [100, 200, 420, 240]
    assert _reading([big, under, giam]) == [2, 0, 1]
