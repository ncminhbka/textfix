"""Bước vẽ FLUX, phần không cần GPU: bảng cỡ ảnh, tìm trọng số. (Lấy mẫu cần GPU: scripts/check_server.py.)"""

from __future__ import annotations

import pytest

from textfix import flux
from textfix.brief.instructions import ORIENT


@pytest.mark.parametrize("aspect,size", flux.SIZES.items())
def test_sizes_are_one_megapixel_multiples_of_16_with_right_ratio(aspect, size):
    w, h = size
    a, b = map(int, aspect.split(":"))
    assert w % 16 == 0 and h % 16 == 0
    assert 0.95e6 <= w * h <= 1.1e6
    assert abs(w / h - a / b) / (a / b) < 0.03


def test_every_aspect_the_api_draws_is_described_to_the_llm_and_back():
    assert set(flux.SIZES) == set(ORIENT)


def test_generate_rejects_sizes_not_multiple_of_16():
    F = flux.Flux.__new__(flux.Flux)
    with pytest.raises(ValueError):
        F.generate("x", 1000, 1000, 0)


@pytest.fixture
def clean_env(monkeypatch):
    for _, _, env, _, _ in flux.MODELS.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("AE_MODEL_PATH", raising=False)
    monkeypatch.delenv("TEXT_ENCODER_PATH", raising=False)
    monkeypatch.setattr(flux, "SEARCH", [])


def test_find_weights_in_diffusers_layout_sets_env(tmp_path, clean_env):
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "flux-2-klein-4b.safetensors").write_bytes(b"")
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"")
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "text_encoder" / "config.json").write_text("{}")
    found = flux.find_weights([tmp_path])
    assert set(found) == {"distill", "vae", "text_encoder"}
    import os
    assert os.environ["KLEIN_4B_MODEL_PATH"] == found["distill"]


def test_find_weights_env_wins(tmp_path, clean_env, monkeypatch):
    mine = tmp_path / "mine.safetensors"
    mine.write_bytes(b"")
    (tmp_path / "flux-2-klein-4b.safetensors").write_bytes(b"")
    monkeypatch.setenv("KLEIN_4B_MODEL_PATH", str(mine))
    assert flux.find_weights([tmp_path])["distill"] == str(mine)
