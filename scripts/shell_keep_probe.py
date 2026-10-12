#!/usr/bin/env python3
"""MÁY CÁ NHÂN: thử GIỮ VỎ NHÁP (S "keep": dán điểm ảnh nháp theo mặt nạ vỏ + lấp nét chữ cũ) thay cho vẽ lại vỏ bằng CSS, trên
mọi plan.json đã lưu có ô vỏ. Giữ nguyên lệnh VLM, chỉ đổi lệnh "shape" của ô S giữ được (keepable) thành "keep", dựng lại.

  python -X utf8 scripts/shell_keep_probe.py <thư mục ra> [thư mục lượt chạy ...]   # mặc định: output/dev2, output/dev3, output/runs
Ra <ra>/sheet_<n>.jpg: mỗi ô vỏ một hàng ba ảnh  nháp | poster đã lưu (vỏ CSS) | poster giữ vỏ ; in từng ô: giữ được hay không.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import numpy as np
    from PIL import Image, ImageDraw
    from textfix import render, slots
    from textfix.browser import Browser
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    roots = sys.argv[2:] or [str(ROOT / "output" / d) for d in ("dev2", "dev3", "runs")]
    plans = sorted(p for r in roots for p in glob.glob(f"{r}/run_*/v*/plan.json"))
    rows, n_keep, n_no = [], 0, 0
    with Browser() as B:
        for f in plans:
            P = json.load(open(f, encoding="utf-8"))
            if not (P.get("slots") or {}).get("S"):
                continue
            r = os.path.dirname(f)
            D = np.asarray(Image.open(r + "/draft.png").convert("RGB"))
            Pl = np.asarray(Image.open(r + "/plate.png").convert("RGB").resize((D.shape[1], D.shape[0])))
            for l in P["slots"]["L"]:
                if l.get("ocr_box"):
                    l["box"] = list(l["ocr_box"])
            M = {**P["slots"], "size": [D.shape[1], D.shape[0]],
                 "by": {x["id"]: x for x in P["slots"]["L"] + P["slots"]["S"] + P["slots"]["I"]}}
            slots.attach_shells(M, D, Pl)
            ops = [{k: w for k, w in o.items() if k != "fix"} for o in P["ops"]]
            for o in ops:
                o.setdefault("marks", o.get("slots") or [])
                sid = [m for m in o["marks"] if m.startswith("S")]
                if o.get("kind") == "shape" and sid and all(M["by"][m].get("keepable") for m in sid if m in M["by"]):
                    o["kind"], o["box_style"], o["html"] = "keep", "", ""
            P2 = {**P, "ops": ops}
            poster, _, _ = render.render(B.page, slots.base_image(D, Pl, M, P2), P2, M)
            pp = r + "/poster.jpg" if os.path.exists(r + "/poster.jpg") else r + "/poster.png"
            old = np.asarray(Image.open(pp).convert("RGB").resize((D.shape[1], D.shape[0])))
            key = os.path.basename(os.path.dirname(r))[-4:] + "_" + os.path.basename(r)
            kinds = {m: o["kind"] for o in ops for m in o["marks"]}
            for s in M["S"]:
                ok = bool(s.get("keepable")) and kinds.get(s["id"]) == "keep"
                n_keep += ok
                n_no += not ok
                print(f"{key} {s['id']}: {'GIỮ' if ok else 'không giữ (' + str(kinds.get(s['id'])) + ', keepable=' + str(s.get('keepable')) + ')'}")
                rows.append((f"{key} {s['id']} {'keep' if ok else kinds.get(s['id'])}", s["box"], D, old, np.asarray(poster)))
    T, cols = 300, 3
    for n in range(0, len(rows), 3 * 7):
        sel = rows[n:n + 21]
        im = Image.new("RGB", (cols * (3 * T + 16), ((len(sel) + cols - 1) // cols) * (T + 20)), "white")
        dr = ImageDraw.Draw(im)
        for k, (tag, b, D, old, new) in enumerate(sel):
            x0, y0, x1, y1 = b
            s = max(x1 - x0, y1 - y0, 80) * 1.2
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            bb = (int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))
            ox, oy = (k % cols) * (3 * T + 16), (k // cols) * (T + 20)
            for j, src in enumerate((D, old, new)):
                im.paste(Image.fromarray(src).crop(bb).resize((T, T), Image.LANCZOS), (ox + j * T, oy + 18))
            dr.text((ox + 2, oy + 3), f"#{n + k} {tag}", fill=(0, 0, 0))
        im.save(out / f"sheet_{n // 21}.jpg", quality=90)
    print(f"{n_keep} ô giữ vỏ nháp, {n_no} ô không; ảnh ghép ở {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
