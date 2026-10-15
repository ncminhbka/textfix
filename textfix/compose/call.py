"""Một lời gọi VLM: lời dặn + PROMPT của FLUX + câu khách + bảng dòng + ba ảnh (nháp, nháp đánh dấu L#, nền đã gỡ chữ).
"""

from __future__ import annotations

import io
import json

import numpy as np

from ..lines.view import marked_image


def _png(im) -> bytes:
    from PIL import Image
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _system(base: str) -> str:
    """Lời dặn VLM + danh mục font (khoá, họ font, lớp, bề rộng chữ). Không còn danh mục icon: TextFix chỉ sửa chữ."""
    from ..render.fonts import CATALOG, WIDTH
    fonts = "; ".join(f"{k}: {v[0]} ({v[1]}" + (f", {WIDTH[k][0]:.2f} / caps {WIDTH[k][1]:.2f}em" if k in WIDTH else "") + ")"
                      for k, v in CATALOG.items())
    return base.replace("{fonts}", fonts)


def _obj(x) -> dict:
    """Đầu ra VLM về dict: đôi khi trả JSON lồng trong chuỗi, hoặc mảng."""
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except ValueError:
            return {}
    if isinstance(x, list):
        x = {"ops": x}
    return x if isinstance(x, dict) else {}


def _texts(texts: list[dict]) -> str:
    # câu "ai" = chữ phụ trợ bước brief tự viết (tiêu đề / khẩu hiệu / nút / nhãn khi khách chỉ điền sự thật): không phải lời khách
    return "\n".join(f"T{i} ({t.get('role', '')}{', OURS' if t.get('source') == 'ai' else ''}): {t['text']}"
                     for i, t in enumerate(texts))


PRODUCT_NOTE = ("The client uploaded a photo of their product and the image model copied it into the draft. Text, logos and "
                "graphics printed on that product (label, package, screen) are part of the photo and MUST stay exactly as they "
                "are: give every slot on the product kind \"keep\" -- never rewrite, correct, translate, redraw or skip it, "
                "and never write a client text there, even when the product's name is one of the client texts. Lines the list "
                "marks PRINTED ON THE CLIENT'S PRODUCT were recognised by code as product text: they are always \"keep\".")


def plan(draft: np.ndarray, M: dict, prompt: str, texts: list[dict], call, system: str, desc: str, base: np.ndarray,
         product: bool = False, meta: dict | None = None, fresh: bool = False) -> dict:
    """Một lời gọi VLM: lời dặn (system), PROMPT của FLUX, câu khách, bảng mô tả dòng (desc), ba ảnh DRAFT / SLOTS / BASE.
    meta: nhận số token / thời gian (llm.json_call). -> câu trả lời JSON (compose.parse đọc)."""
    mk = marked_image(draft, M)
    W, H = mk.size
    imgs = ["DRAFT:", _png(draft), "SLOTS (marked):", _png(mk),
            "BASE (the draft with the old letters wiped; your text is drawn on it):", _png(base)]
    user = (f"PROMPT given to the image model:\n{prompt}\n\nClient texts (exact words):\n{_texts(texts)}\n\nPoster size: {W}x{H}px.\n\n"
            + (PRODUCT_NOTE + "\n\n" if product else "") + desc)
    args = (system, [user, *imgs]) + ((meta,) if meta is not None else ())
    return _obj(call(*args, fresh=True) if fresh else call(*args))


def _tid(c) -> int | None:
    c = str(c or "")
    return int(c[1:]) if c[:1] in "Tt" and c[1:].isdigit() else int(c) if c.isdigit() else None
