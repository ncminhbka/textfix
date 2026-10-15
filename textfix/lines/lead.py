"""Đầu dòng: chữ bắt đầu ở đâu, và thứ đứng trước chữ.

  trang trí  mảnh đầu tiên nhỏ (<= ORNAMENT cao chữ cả hai chiều), tâm ở khoảng giữa chiều cao dòng (MID: dấu ngoặc mở đầu
             nằm cao, dấu phẩy nằm thấp), cách mảnh sau >= ORNAMENT_GAP cao chữ (chấm "•", gạch, tick nhỏ): không thuộc chữ --
             ở lại là điểm ảnh nháp như icon;
  mép chữ    mép trái mảnh chữ đầu tiên. Khung bộ dò ôm cả thứ đứng trước (icon không phải nét Hi-SAM, chấm đầu dòng) thì khung
             dòng bắt đầu từ đây -- khi khoảng thừa rõ (>= TRIM cao chữ, hoặc có trang trí) và không quá xa (> TRIM_MAX cao chữ
             là Hi-SAM thấy thiếu chữ đầu dòng, không biết chữ bắt đầu ở đâu).
Chỉ đo cho dòng ngang (dòng nghiêng / dọc / cong: khung theo hình học riêng).
"""

from __future__ import annotations

from .group import Group

ORNAMENT = 0.6       # cạnh trang trí <= 0,6 cao chữ (chữ thường không nét lên ~0,74)
ORNAMENT_GAP = 0.3   # ... cách chữ sau >= 0,3 cao chữ (khe giữa hai chữ cái ~0,1)
MID = (0.2, 0.8)     # ... tâm trong khoảng 20-80% chiều cao dòng (tính từ đỉnh các mảnh chính)
TRIM = 0.6           # khung thừa bên trái >= 0,6 cao chữ
TRIM_MAX = 4.0       # ... và <= 4 cao chữ


def lead(g: Group, ink, box_x0: float) -> tuple[list[int], float | None]:
    """-> (id các mảnh trang trí, mép trái mới của khung hoặc None khi giữ khung)."""
    main = g.main()
    if len(main) < 3:
        return [], None
    order = sorted(g.ids, key=lambda i: ink.stats[i, 0])
    a = order[0]
    x, y, w, h = (int(v) for v in ink.stats[a, :4])
    top = min(ink.stats[i, 1] for i in main)
    bot = max(ink.stats[i, 1] + ink.stats[i, 3] for i in main)
    mid = (y + h / 2 - top) / max(1, bot - top)
    nxt = min(ink.stats[i, 0] for i in order[1:])
    orn = [a] if (max(w, h) <= ORNAMENT * g.h and MID[0] <= mid <= MID[1] and nxt - (x + w) >= ORNAMENT_GAP * g.h) else []
    first = float(min(ink.stats[i, 0] for i in g.ids if i not in orn))
    gap = first - box_x0
    if (orn or gap >= TRIM * g.h) and gap <= TRIM_MAX * g.h:
        return orn, first - max(1.0, 0.1 * g.h)
    return orn, None
