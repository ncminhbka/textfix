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
    # chặng B: chữ là nguyên văn câu khách; mã chỉ còn chặn chữ LẤP có số bịa và split không nối lại đúng câu
    guard = [l for l in log if "có số không có trong câu khách" in l or "không nối lại đúng câu khách" in l]
    if guard:
        out.append(f"mã chặn {len(guard)} chữ lấp / chỗ cắt câu sai của VLM")
    if any("VẪN RỖNG" in l for l in log):
        out.append("VLM trả kế hoạch rỗng")
    # ---- đường chỉ dòng chữ (lonly): những lần MÃ phải làm thay VLM, và bước gỡ chữ rơi về đường dự phòng
    blank = sum("xoá trống" in l for l in log)
    if blank:
        out.append(f"{blank} dòng nháp xoá trống (không còn câu nào để viết)")
    fill = sum("chưa ai ghép" in l and "-> câu khách" in l for l in log)
    if fill >= 2:
        out.append(f"mã tự ghép {fill} dòng VLM bỏ sót")
    own = sum("-> dòng lấp" in l for l in log)
    if own:
        out.append(f"{own} dòng ghép trùng câu đã có -> thành dòng lấp")
    if any("gọi lại" in l and "dòng" in l for l in log):
        out.append("VLM viết thiếu dòng, đã gọi lại")
    er = plan.get("erase") or {}
    if er.get("orphans"):
        out.append(f"{len(er['orphans'])} cụm nét OCR không đọc (VLM quyết giữ / xoá)")
    core = sum(int(x) for l in (er.get("log") or []) + log for x in re.findall(r"lõi dòng (\d+)", l)[:1])
    if core:
        out.append(f"{core} dòng Hi-SAM không thấy nét (gỡ theo lõi khung)")
    return out
