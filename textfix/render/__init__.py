"""BƯỚC 5 -- DỰNG các đơn vị chữ (textfix/compose/layout.py) lên nền (nháp đã gỡ chữ) trong Chromium; báo lỗi khách quan cho bước sửa bằng code.

  poster, res, log = render_units(page, base, U, M)   # mỗi phần tử = một div tại khung nét nháp, chữ nguyên văn câu khách
  errs = check_units(U, res)                           # nội dung chồng nhau, tràn, phải co quá nhỏ, chìm nền
  U2, log = autofix_units(U, errs, res)                # màu tương phản, nén sâu hơn, cắt khung chồng

Dòng model vẽ nghiêng: div dựng thẳng rồi xoay đúng góc; dòng cong: SVG textPath. TẬP cùng vai (compose.layout): đo bề rộng tự
nhiên mọi mục ở cỡ chung, rồi CẢ TẬP một mức nén ngang + một cỡ theo mục chật nhất (không mục nào co riêng).

Phần: page.py (dựng), check.py (bộ kiểm, tương phản), fix.py (vòng sửa), geometry.py (khung / xoay / đường cong),
style.py (font theo khoá), js.py (mã dựng trong trang), view.py (ảnh debug), fonts.py, browser.py.
"""

from __future__ import annotations

from .check import SEVERE, check_units  # noqa: F401
from .fix import FIX_ROUNDS, autofix_units  # noqa: F401
from .geometry import ROT_MIN, op_box, op_geom, path_of  # noqa: F401
from .page import CONDENSE, render_units  # noqa: F401
from .view import ops_image  # noqa: F401

__all__ = ["CONDENSE", "FIX_ROUNDS", "ROT_MIN", "SEVERE", "autofix_units", "check_units", "op_box", "op_geom", "ops_image", "path_of", "render_units"]
