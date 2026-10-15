"""Lớp API (server/app.py) không cần model: đầu vào sai trả 422 có mã cố định, model chưa nạp trả 503 (docs/API.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import app as server  # noqa: E402

client = TestClient(server.app)   # không "with": không chạy lifespan, không nạp model


def post(**body):
    return client.post("/api/jobs", json=body)


@pytest.mark.parametrize("body, code", [
    ({"category": "flyer", "discount": "20%"}, "invalid_category"),
    ({"category": "promo", "discount": "20%", "aspect_ratio": "5:7"}, "invalid_aspect_ratio"),
    ({"category": "promo", "discount": "20%", "num_images": 9}, "invalid_num_images"),
    ({"category": "promo", "discount": "20%", "image_base64": "không phải ảnh"}, "invalid_image"),
    ({"category": "promo"}, "missing_required_fields"),
    ({"category": "recruitment", "job_position": "Pha chế"}, "missing_required_fields"),
])
def test_bad_input_is_422_with_code(body, code):
    r = post(**body)
    assert r.status_code == 422 and r.json()["detail"]["code"] == code


def test_required_fields_listed():
    r = post(category="recruitment", job_position="Pha chế")
    assert r.json()["detail"]["fields"] == ["apply_deadline", "apply_method"]


@pytest.mark.parametrize("body", [
    {"category": "guide", "title": "Đặt hàng 3 bước"},          # hướng dẫn: không trường nào bắt buộc (như app)
    {"category": "promo", "prompt": "poster giảm giá cuối tuần"},   # có prompt: prompt quyết, không đòi trường
])
def test_valid_input_reaches_model_check(body):
    r = post(**body)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "model_not_ready"


def test_fields_of_other_categories_are_reported():
    req = server.GenerateRequest(category="promo", discount="20%", job_position="Pha chế", foo="x")
    assert server.ignored_fields(req) == ["job_position", "foo"]


def test_remote_flux_draw_threads_follow_concurrency_not_gpus(monkeypatch):
    # FLUX ở máy khác: số luồng vẽ = số nháp gửi cùng lúc, chia vòng lên card (chỉ để chạy Hi-SAM / LaMa); không card: CPU
    monkeypatch.setenv("TEXTFIX_FLUX_URL", "https://api.test/v1/flux-2-klein-9b")
    monkeypatch.setenv("TEXTFIX_FLUX_CONCURRENCY", "3")
    monkeypatch.setattr(server, "gpu_list", lambda: [0, 1])
    assert server.draw_devices() == ["cuda:0", "cuda:1", "cuda:0"]
    monkeypatch.setattr(server, "gpu_list", lambda: [])
    assert server.draw_devices() == ["cpu"] * 3


def test_flux_moderation_has_its_own_code():
    from textfix.draw.remote import FluxError, FluxModerated
    assert server.draw_error(FluxModerated("Content Moderated"), "FLUX vẽ nháp")["code"] == "draw_moderated"
    assert server.draw_error(FluxError("HTTP 500"), "FLUX vẽ nháp")["code"] == "draw_failed"
