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
