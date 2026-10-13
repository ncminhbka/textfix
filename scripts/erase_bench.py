#!/usr/bin/env python3
"""MÁY CÁ NHÂN: ĐO bước gỡ chữ của chế độ chỉ dòng chữ (lonly.draft_base) bằng SỐ, trên các cặp nháp / bản xoá đã lưu.

  python -X utf8 scripts/erase_bench.py <thư mục ra> mix1 accept1 [--step 2] [--rules] [--sheet]

Mỗi poster (bỏ lệnh VLM, gỡ MỌI dòng):
  sót   = số ký tự chữ / số mà OCR còn đọc được trên nền sau khi gỡ, tại các dòng của nháp (0 là sạch)
  đổi   = % điểm ảnh của nền khác nháp (> 12 mức) -- cùng mức "sót" thì đổi ÍT hơn là tốt hơn (ít đụng vỏ / hình)
  ngoài = % điểm ảnh đổi nằm NGOÀI mọi vùng dòng (nới 0,6 cao dòng) -- hỏng thứ không phải chữ
--rules: ép đường luật cũ (không LaMa) để so. --step N: lấy 1 / N số poster cho nhanh. --sheet: lưu ảnh nháp | nền."""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("sets", nargs="+")
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--rules", action="store_true")
ap.add_argument("--sheet", action="store_true")
a = ap.parse_args()
if a.rules:
    os.environ["TEXTFIX_LAMA"] = "off"
from textfix import lonly, ocr   # noqa: E402

os.makedirs(a.out, exist_ok=True)
rows, t0 = [], time.time()
for setn in a.sets:
    M0 = json.load(open(f"output/{setn}/{setn}_manifest.json", encoding="utf-8"))
    for cid in sorted(M0)[::a.step]:
        r = f"output/{setn}/{M0[cid].get('run_id')}/v0/"
        f = lambda k: next((r + k + x for x in (".png", ".jpg") if os.path.exists(r + k + x)), None)   # noqa: E731
        if not (os.path.exists(r + "plan.json") and f("draft") and f("plate")):
            continue
        P = json.load(open(r + "plan.json", encoding="utf-8"))
        L = (P.get("slots") or {}).get("L") or []
        if not L:
            continue
        D = np.asarray(Image.open(f("draft")).convert("RGB"))
        Pl = np.asarray(Image.open(f("plate")).convert("RGB").resize((D.shape[1], D.shape[0])))
        H, W = D.shape[:2]
        for l in L:
            if l.get("ocr_box"):
                l["box"] = list(l["ocr_box"])
        M = lonly.view({"L": L, "S": [], "I": [], "size": [W, H], "by": {}})
        log = []
        base = lonly.draft_base(D, Pl, M, None, log=log)
        zone = np.zeros((H, W), bool)
        for l in L:
            zone |= lonly._region(l, H, W, 0.6)
        left = 0
        for o in ocr.read_lines(base):
            b = o.get("box") or [min(p[0] for p in o["poly"]), min(p[1] for p in o["poly"]), max(p[0] for p in o["poly"]), max(p[1] for p in o["poly"])]
            cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
            if 0 <= cy < H and 0 <= cx < W and zone[cy, cx]:
                left += sum(ch.isalnum() for ch in o.get("text") or "")
        ch = np.abs(base.astype(int) - D.astype(int)).max(axis=2) > 12
        rows.append({"set": setn, "id": cid, "left": left, "changed": round(100 * float(ch.mean()), 2),
                     "outside": round(100 * float((ch & ~zone).mean()), 3), "log": log[-1] if log else ""})
        print(f"{setn} {cid}: sót {left:3d}  đổi {rows[-1]['changed']:5.2f}%  ngoài {rows[-1]['outside']:.3f}%", flush=True)
        if a.sheet:
            im = Image.fromarray(np.hstack([D, base]))
            im.thumbnail((1400, 1400))
            im.save(f"{a.out}/{setn}_{cid}.jpg", quality=84)
n = max(1, len(rows))
tot = {"poster": len(rows), "poster còn sót chữ": sum(r["left"] > 0 for r in rows), "ký tự sót": sum(r["left"] for r in rows),
       "đổi TB %": round(sum(r["changed"] for r in rows) / n, 2), "ngoài TB %": round(sum(r["outside"] for r in rows) / n, 3),
       "giây / poster": round((time.time() - t0) / n, 1), "đường": "luật" if a.rules else ("LaMa" if lonly.lama_ready() else "luật (không có LaMa)")}
json.dump({"total": tot, "rows": rows}, open(f"{a.out}/erase_bench.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(tot)
