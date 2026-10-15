"""Bước gỡ chữ trên ảnh tổng hợp (không cần Hi-SAM / LaMa): mỗi test là một cam kết của textfix.erase."""

from __future__ import annotations

import numpy as np
import pytest

from textfix import erase

H, W = 200, 400
BG = (240, 240, 240)


def poster():
    """Nền xám nhạt; dòng A (y 40-70): 5 "chữ" đen có bóng đổ xám đậm lệch 3 px, icon tròn đỏ đứng trước cách một khe;
    dòng B (y 130-160): 4 "chữ" xanh (dòng in trên sản phẩm, sẽ giữ)."""
    img = np.full((H, W, 3), BG, np.uint8)
    ink = np.zeros((H, W), bool)
    for i in range(5):
        x = 100 + i * 30
        img[43:73, x + 3:x + 21] = (150, 150, 150)   # bóng đổ
        img[40:70, x:x + 18] = (20, 20, 20)
        ink[40:70, x:x + 18] = True
    yy, xx = np.ogrid[:H, :W]
    icon = (yy - 55) ** 2 + (xx - 70) ** 2 <= 10 ** 2
    img[icon] = (220, 30, 30)
    for i in range(4):
        x = 100 + i * 30
        img[130:160, x:x + 18] = (30, 60, 200)
        ink[130:160, x:x + 18] = True
    lines = [{"id": "L0", "box": [55, 38, 250, 72], "size": 30, "color": "#141414"},
             {"id": "L1", "box": [95, 128, 222, 162], "size": 30, "color": "#1e3cc8"}]
    return img, ink, {"L": lines}, icon


def test_mask_covers_strokes_and_drop_shadow():
    img, ink, M, _ = poster()
    mask = erase.erase_mask(img, ink, M, set())
    letters_a = ink.copy()
    letters_a[100:] = False
    shadow = (np.abs(img.astype(int) - 150).sum(axis=2) == 0)
    assert mask[letters_a].all()
    assert mask[shadow].mean() > 0.95


def test_icon_separated_by_a_gap_is_left_alone():
    img, ink, M, icon = poster()
    mask = erase.erase_mask(img, ink, M, set())
    assert mask[icon].mean() < 0.05


def test_kept_line_never_masked():
    img, ink, M, _ = poster()
    mask = erase.erase_mask(img, ink, M, {"L1"})
    assert not mask[130:160].any()
    assert mask[40:70, 100:118].all()


def test_settle_restores_kept_line_from_draft(monkeypatch):
    img, ink, M, _ = poster()
    monkeypatch.setattr(erase, "lama_ready", lambda: True)
    monkeypatch.setattr(erase, "inpaint", lambda im, mk, device=None: np.full_like(im, BG))
    clean, _ = erase.erase(img, ink, M)
    assert np.abs(clean[40:70, 100:118].astype(int) - BG).max() <= 2       # chữ dòng A đã gỡ
    base, _ = erase.settle(img, clean, ink, M, {"ops": [{"kind": "keep", "marks": ["L1"]}]})
    assert np.array_equal(base[135:155, 103:115], img[135:155, 103:115])  # dòng giữ trở về điểm ảnh nháp


def test_outside_mask_is_untouched(monkeypatch):
    img, ink, M, _ = poster()
    monkeypatch.setattr(erase, "lama_ready", lambda: True)
    monkeypatch.setattr(erase, "inpaint", lambda im, mk, device=None: np.zeros_like(im))
    clean, mask = erase.erase(img, ink, M)
    far = ~erase._dilate(mask, np.ones((9, 9), np.uint8))   # mép mềm: Gauss σ 1, OpenCV dùng nhân 9 px với ảnh float
    assert np.array_equal(clean[far], img[far])


def test_missing_lama_weights_is_an_error(monkeypatch):
    img, ink, M, _ = poster()
    monkeypatch.setattr(erase, "lama_ready", lambda: False)
    with pytest.raises(RuntimeError):
        erase.erase(img, ink, M)


def test_no_lines_means_no_mask():
    img, ink, _, _ = poster()
    assert not erase.erase_mask(img, ink, {"L": []}, set()).any()


def test_kept_of_reads_marks_and_legacy_slots():
    P = {"ops": [{"kind": "keep", "marks": ["L1"]}, {"kind": "keep", "slots": ["L4"]}, {"kind": "text", "marks": ["L2"]}]}
    assert erase.kept_of(P) == {"L1", "L4"} and erase.kept_of(None) == set()
