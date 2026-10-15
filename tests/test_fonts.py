"""Số đo font (textfix.fonts) mà bước bố cục dựa vào."""

from __future__ import annotations

import pytest


def test_font_heights_differ_by_family():
    from textfix.fonts import CAP_EM, X_EM, heights
    assert heights("anton")[0] > 0.8 and heights("allura")[0] < 0.62   # chữ hẹp cao / thư pháp thấp: không dùng một hằng chung
    assert heights("bungee")[1] == pytest.approx(heights("bungee")[0], abs=0.01)   # font toàn chữ hoa
    assert heights("khong-co") == (CAP_EM, X_EM)
