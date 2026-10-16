"""CỜ NGHI HỎNG của một poster, tính từ plan.json (không cần ảnh, không cần AI) -- để theo dõi xu hướng và chọn poster cần xem,
KHÔNG phải điểm chất lượng. Đo trên 39 poster accept1 chấm mắt (13/10, 15 hỏng): bộ kiểm hình học riêng nó bắt 6 / 15, báo nhầm
6 / 24; gộp các cờ dưới đây bắt 11 / 15, báo nhầm 6 / 24. Bốn ca sót là loại chỉ mắt thấy (danh sách vỡ cột, cỡ chữ đảo, huy
hiệu méo, nhãn mất chữ). Mỗi lần chấm mắt một bộ mới thì đo lại hai con số này (docs/TUNING.md)."""

from __future__ import annotations

import re

SEVERE = ("empty_plan", "content_overlap", "overflow", "low_contrast")   # = render.SEVERE (chặng B)
SHRINK = 0.6   # lệnh chữ co dưới 0.6 cỡ nháp


def flags(plan: dict) -> list[str]:
    """-> các cờ (chuỗi ngắn, tiếng Việt); rỗng = không thấy dấu hiệu gì."""
    log = plan.get("log") or []
    out = []
    left = sorted({e["type"] for e in plan.get("errors") or [] if e.get("type") in SEVERE})
    if left:
        out.append("còn lỗi nặng: " + ", ".join(left))
    first = sorted({e["type"] for e in plan.get("errors_first") or [] if e.get("type") in SEVERE and e["type"] != "low_contrast"} - set(left))
    if first:
        out.append("lượt đầu lỗi nặng (mã đã sửa): " + ", ".join(first))
    shr = [float(x) for l in log for x in re.findall(r"co còn ([0-9.]+)", l)]
    if any(x < SHRINK for x in shr):
        out.append(f"chữ co còn {min(shr):.2f}")
    miss = plan.get("missing") or []
    if miss:
        out.append("mất câu khách: " + " · ".join(miss)[:120])
    # chặng B: chữ là nguyên văn câu khách; mã chỉ còn chặn split không nối lại đúng câu
    guard = [l for l in log if "không nối lại đúng câu khách" in l]
    if guard:
        out.append(f"mã chặn {len(guard)} chỗ cắt câu sai của VLM")
    if any(e.get("type") == "empty_plan" for e in plan.get("errors") or []):
        out.append("VLM trả kế hoạch rỗng")
    # ---- đường chỉ dòng chữ (compose): những lần MÃ phải làm thay VLM, và bước gỡ chữ rơi về đường dự phòng
    blank = sum("không còn chữ cho dòng" in l for l in log)
    if blank:
        out.append(f"{blank} dòng nháp xoá trống (không còn câu nào để viết)")
    kept = sum("VLM bỏ sót ->" in l for l in log)
    if kept:
        out.append(f"{kept} dòng VLM bỏ sót (không theo khuôn JSON) -> giữ nguyên")
    own = sum("-> dòng lấp" in l for l in log)
    if own:
        out.append(f"{own} dòng ghép trùng câu đã có -> thành dòng lấp")
    er = plan.get("erase") or {}
    if er.get("orphans"):
        out.append(f"{len(er['orphans'])} cụm nét OCR không đọc (VLM quyết giữ / xoá)")
    core = sum(int(x) for l in (er.get("log") or []) + log for x in re.findall(r"lõi dòng (\d+)", l)[:1])
    if core:
        out.append(f"{core} dòng Hi-SAM không thấy nét (gỡ theo lõi khung)")
    return out
