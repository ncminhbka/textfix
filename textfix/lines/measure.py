"""Số đo kiểu chữ của một dòng, lấy thẳng trên NÉT Hi-SAM của dòng (không gom cụm màu trong khung như trước).

  cao chữ hoa  = phân vị 90 cao các mảnh chính (chữ hoa / chữ số / nét lên); cao chữ thường = trung vị;
  đường chân   = trung vị đáy các mảnh chính (y trên ảnh; chân thò g, y là thiểu số nên không kéo lệch) -- nghĩa với dòng ngang;
  cỡ (em)      = cao chữ hoa / CAP_EM (cỡ cuối cùng tính lại theo font + câu khách ở bước bố cục);
  màu          = trung vị màu điểm LÕI nét (bỏ viền khử răng cưa);
  độ dày nét   = 2 x phân vị 90 khoảng cách tới mép trên xương chữ; tương phản nét = p90 / p25 (chữ có chân dày-mảnh rõ).
"""

from __future__ import annotations

import numpy as np

from .group import MAIN, Group

CAP_EM = 0.72   # cao chữ hoa / em (~ cao chữ số, nét lên b d h k l t) ở hầu hết font


def measure(draft: np.ndarray, ink, ids: list[int]) -> dict:
    import cv2
    g = Group(ink, ids)
    hs = sorted(g.ext[i][3] - g.ext[i][2] + 1 for i in g.ids)
    main = [x for x in hs if x >= MAIN * g.h] or hs or [1.0]
    h_cap, h_x = float(np.percentile(main, 90)), float(np.median(main))
    base = float(np.median([ink.stats[i, 1] + ink.stats[i, 3] for i in (g.main() or g.ids)]))
    mask = ink.mask_of(ids)
    core = mask & (cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    px = draft[core] if int(core.sum()) >= 30 else draft[mask]
    color = "#%02x%02x%02x" % tuple(int(v) for v in np.median(px, axis=0)) if len(px) else None
    ys, xs = np.nonzero(mask)
    stroke, contrast = None, None
    if len(ys):
        sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
        dist = cv2.distanceTransform(np.pad(sub, 1), cv2.DIST_L2, 3)
        sk = (dist > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))   # xương chữ ~ đỉnh cục bộ của khoảng cách
        v = dist[sk] if sk.sum() > 20 else dist[dist > 0]
        v = v[v > 0]
        if len(v):
            stroke = round(float(2 * np.percentile(v, 90)), 1)
            contrast = round(float(np.percentile(v, 90) / max(0.5, np.percentile(v, 25))), 2)
    return {"h_cap": round(h_cap, 1), "h_x": round(h_x, 1), "base": round(base, 1), "color": color,
            "stroke": stroke, "stroke_contrast": contrast}
