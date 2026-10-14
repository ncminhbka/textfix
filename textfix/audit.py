"""CỜ NGHI HỎNG của một poster, tính từ plan.json (không cần ảnh, không cần AI) -- để theo dõi xu hướng và chọn poster cần xem,
KHÔNG phải điểm chất lượng. Đo trên 39 poster accept1 chấm mắt (13/10, 15 hỏng): bộ kiểm hình học riêng nó bắt 6 / 15, báo nhầm
6 / 24; gộp các cờ dưới đây bắt 11 / 15, báo nhầm 6 / 24. Bốn ca sót là loại chỉ mắt thấy (danh sách vỡ cột, cỡ chữ đảo, huy
hiệu méo, nhãn mất chữ). Mỗi lần chấm mắt một bộ mới thì đo lại hai con số này (docs/TUNING.md)."""

from __future__ import annotations

import re

SEVERE = ("empty_plan", "unassigned", "content_overlap", "boxes_overlap", "overflow", "low_contrast", "mark_twice")
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
    miss = [l.split(":", 1)[1].strip() for l in log if l.startswith("câu khách không có trên poster")]
    if miss:
        out.append("mất câu khách: " + miss[0][:120])
    guard = [l for l in log if any(w in l for w in ("chữ chệch", "bỏ chữ không có", "số không có trong câu khách", "lặp chữ", "chữ không có trong câu khách"))]
    if guard:
        out.append(f"bộ chặn chữ can thiệp {len(guard)} lệnh")
    if any("VẪN RỖNG" in l for l in log):
        out.append("VLM trả kế hoạch rỗng")
    # ---- đường chỉ dòng chữ (lonly): những lần MÃ phải làm thay VLM, và bước gỡ chữ rơi về đường dự phòng
    blank = sum("xoá trống" in l for l in log)
    if blank:
        out.append(f"{blank} dòng nháp xoá trống (không còn câu nào để viết)")
    fill = sum("dòng bị bỏ" in l and "viết câu khách" in l for l in log)
    if fill >= 2:
        out.append(f"mã tự lấp {fill} dòng VLM bỏ")
    if any("gọi lại" in l and "dòng" in l for l in log):
        out.append("VLM viết thiếu dòng, đã gọi lại")
    er = plan.get("erase") or {}
    if er.get("orphans"):
        out.append(f"{len(er['orphans'])} cụm nét OCR không đọc (VLM quyết giữ / xoá)")
    core = sum(int(x) for l in (er.get("log") or []) + log for x in re.findall(r"lõi dòng (\d+)", l)[:1])
    if core:
        out.append(f"{core} dòng Hi-SAM không thấy nét (gỡ theo lõi khung)")
    if sum("lệnh gộp" in l and "không xếp chồng" in l for l in log):
        out.append("VLM gộp dòng của nhiều khối (mã đã tách)")
    return out
