"""HỢP ĐỒNG API của máy chủ (docs/API.md): form từng loại, kiểm đầu vào (422 có mã cố định), mã lỗi trả cho người gọi.

Không phụ thuộc model hay dây chuyền: server/app.py dùng, test (tests/test_api.py) và script (scripts/brief_probe.py) gọi thẳng.
"""

from __future__ import annotations

import base64
import io
from typing import List, Optional

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

# trường NỘI DUNG từng loại form (khớp UI); trường chung cửa hàng gửi cho mọi loại
CATEGORY_FIELDS = {
    "promo": ["title", "discount", "applied_product", "date_start", "date_end"],
    "product_intro": ["title", "product_name", "product_desc", "highlights", "price"],
    "opening": ["title", "opening_date", "opening_promo", "booking_contact"],
    "feedback": ["title", "feedback_target", "feedback_quote", "feedback_highlights", "special_offer"],
    "recruitment": ["title", "job_position", "job_desc", "apply_deadline", "apply_method"],
    "guide": ["title"],
}
REQUIRED = {"promo": ["discount"], "product_intro": ["product_name", "product_desc"], "opening": ["opening_date"],
            "feedback": ["feedback_target", "feedback_quote"], "recruitment": ["job_position", "apply_deadline", "apply_method"],
            "guide": []}   # như form của app: hướng dẫn không trường nào bắt buộc (không bước nào: LLM viết theo tiêu đề / prompt)
COMMON = ["store_name", "phone", "address", "website_link"]
MAX_IMAGES = 4
RATIOS = ["1:1", "4:5", "9:16", "16:9"]   # 4 tỉ lệ của app (bộ vẽ có thêm 3:4, 2:3, 4:3 -- flux.SIZES -- app không dùng)


class ApiError(Exception):
    """Lỗi trả cho người gọi API: mã cố định (docs/API.md), bước hỏng, thông điệp."""

    def __init__(self, code: str, stage: str, message: str = ""):
        super().__init__(message or code)
        self.code, self.stage, self.message = code, stage, message or code

    def as_dict(self) -> dict:
        return {"code": self.code, "stage": self.stage, "message": self.message}


def failure(code: str, stage: str, e: BaseException | None = None) -> dict:
    return {"code": code, "stage": stage, "message": f"{type(e).__name__}: {e}" if e else code}


def draw_error(e: BaseException, stage: str) -> dict:
    """Trạm VẼ hỏng: máy FLUX lọc nội dung -> draw_moderated (không phải lỗi của TextFix), còn lại draw_failed."""
    from textfix.draw.remote import FluxModerated
    return failure("draw_moderated" if isinstance(e, FluxModerated) else "draw_failed", stage, e)


# bước thiết kế (engine.design báo qua progress) -> mã lỗi khi hỏng ở bước đó
DESIGN_STAGES = (("đo dòng", "lines_failed"), ("gỡ chữ", "erase_failed"), ("VLM", "vlm_failed"), ("dựng", "render_failed"))


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")   # khoá lạ không làm hỏng yêu cầu: báo lại trong "ignored_fields"
    category: str = "promo"
    # thông tin chung theo loại
    title: Optional[str] = ""
    discount: Optional[str] = ""
    applied_product: Optional[str] = ""
    date_start: Optional[str] = ""
    date_end: Optional[str] = ""
    product_name: Optional[str] = ""
    product_desc: Optional[str] = ""
    highlights: Optional[str] = ""
    price: Optional[str] = ""
    opening_date: Optional[str] = ""
    opening_promo: Optional[str] = ""
    booking_contact: Optional[str] = ""
    feedback_target: Optional[str] = ""
    feedback_quote: Optional[str] = ""
    feedback_highlights: Optional[str] = ""
    special_offer: Optional[str] = ""
    job_position: Optional[str] = ""
    job_desc: Optional[str] = ""
    apply_deadline: Optional[str] = ""
    apply_method: Optional[str] = ""
    guide_steps: Optional[List[str]] = None
    prompt: Optional[str] = ""
    image_base64: Optional[str] = None   # ảnh sản phẩm tải lên (data URI hoặc base64)
    # cửa hàng
    store_name: Optional[str] = ""
    phone: Optional[str] = ""
    address: Optional[str] = ""
    website_link: Optional[str] = ""
    # hiển thị / thiết kế
    style_pref: Optional[str] = "auto"
    primary_color: Optional[str] = None
    aspect_ratio: str = "1:1"
    num_images: int = 1
    seed: Optional[int] = None


def form_of(req: GenerateRequest) -> dict:
    """Chỉ trường thuộc loại form đang chọn + trường cửa hàng, bỏ ô trống. Các bước hướng dẫn -> step_1, step_2..."""
    cat = req.category if req.category in CATEGORY_FIELDS else "promo"
    out: dict = {"category": cat}
    for k in CATEGORY_FIELDS[cat] + COMMON:
        v = (getattr(req, k) or "").strip()
        if v:
            out[k] = v
    if cat == "guide":
        for i, s in enumerate([s.strip() for s in req.guide_steps or [] if s and s.strip()], 1):
            out[f"step_{i}"] = s
    return out


def missing_fields(req: GenerateRequest, form: dict) -> list[str]:
    """Trường bắt buộc -- chỉ khi KHÔNG có prompt (có prompt thì prompt quyết)."""
    if (req.prompt or "").strip():
        return []
    return [k for k in REQUIRED.get(form["category"], []) if k not in form]


def ignored_fields(req: GenerateRequest) -> list[str]:
    """Trường có giá trị nhưng không thuộc loại form đang chọn, và khoá lạ: bỏ qua, báo lại cho người gọi."""
    own = set(CATEGORY_FIELDS.get(req.category, [])) | set(COMMON) | ({"guide_steps"} if req.category == "guide" else set())
    content = {k for fs in CATEGORY_FIELDS.values() for k in fs} | {"guide_steps"}
    return [k for k in sorted(content - own) if getattr(req, k, None)] + sorted((req.model_extra or {}).keys())


def validate(req: GenerateRequest) -> None:
    """Đầu vào sai -> 422 với mã cố định (người gọi sửa được; không tự đoán thay)."""
    if req.category not in CATEGORY_FIELDS:
        raise HTTPException(422, {"code": "invalid_category", "value": req.category, "allowed": sorted(CATEGORY_FIELDS)})
    if req.aspect_ratio not in RATIOS:
        raise HTTPException(422, {"code": "invalid_aspect_ratio", "value": req.aspect_ratio, "allowed": RATIOS})
    if not 1 <= int(req.num_images or 0) <= MAX_IMAGES:
        raise HTTPException(422, {"code": "invalid_num_images", "value": req.num_images, "allowed": [1, MAX_IMAGES]})
    if req.image_base64:
        try:
            decode_image(req.image_base64)
        except Exception as e:
            raise HTTPException(422, {"code": "invalid_image", "message": f"{type(e).__name__}: {e}"}) from e


def decode_image(b64: str | None) -> np.ndarray | None:
    if not b64:
        return None
    from PIL import Image
    raw = base64.b64decode(b64.split(",", 1)[1] if "," in b64 else b64)
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((1024, 1024))
    return np.asarray(im)
