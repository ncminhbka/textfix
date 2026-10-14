#!/usr/bin/env python3
"""MÁY CÁ NHÂN: ĐO bước gỡ chữ (lonly.erase: nét chữ Hi-SAM + LaMa) bằng SỐ, trên các nháp đã lưu.

  python -X utf8 scripts/erase_bench.py <thư mục ra> accept2 [mix1c:mix1] [--step 2] [--recheck] [--sheet]

Tập viết <thư mục trong output>[:<tên manifest>] (mix1c:mix1 = output/mix1c/mix1_manifest.json). Mỗi poster (bỏ lệnh VLM, gỡ MỌI
dòng trừ dòng plan đã lưu ghi "keep"):
  sót   = số ký tự chữ / số mà OCR còn đọc được trên nền sau khi gỡ, tại các dòng đã gỡ (0 là sạch)
  nét   = (--recheck) chạy lại Hi-SAM trên NỀN: % số điểm nét cũ ở các dòng đã gỡ mà Hi-SAM vẫn thấy là chữ -- bóng chữ, mẩu sót
  đổi   = % điểm ảnh của nền khác nháp (> 12 mức) -- cùng mức "sót" thì đổi ÍT hơn là tốt hơn (ít đụng vỏ / hình)
  ngoài = % điểm ảnh đổi nằm NGOÀI mọi vùng dòng (nới 0,6 cao dòng) -- hỏng thứ không phải chữ
Mặt nạ nét chữ đọc ở <lượt chạy>/v0/ink.png (máy chủ lưu); chưa có thì chạy Hi-SAM tại máy (CPU ~13 s) rồi lưu lại.
--sheet: lưu ảnh nháp | nét chữ + mặt nạ | nền."""
import argparse
import glob
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
ap.add_argument("--only", default="")
ap.add_argument("--recheck", action="store_true")
ap.add_argument("--sheet", action="store_true")
a = ap.parse_args()
from textfix import hisam, lonly, ocr, slots   # noqa: E402

os.makedirs(a.out, exist_ok=True)
only = {x for x in a.only.split(",") if x}
rows, t0 = [], time.time()
for spec in a.sets:
    setn, mfn = (spec.split(":") + [spec])[:2] if ":" in spec else (spec, spec)
    M0 = json.load(open(f"output/{setn}/{mfn}_manifest.json", encoding="utf-8"))
    for cid in sorted(M0)[::a.step]:
        if only and cid not in only:
            continue
        r = f"output/{setn}/{M0[cid].get('run_id')}/v0/"
        dr = next(iter(glob.glob(r + "draft.*")), None)
        if not (os.path.exists(r + "plan.json") and dr):
            continue
        P = json.load(open(r + "plan.json", encoding="utf-8"))
        L = (P.get("slots") or {}).get("L") or []
        if not L:
            continue
        D = np.asarray(Image.open(dr).convert("RGB"))
        H, W = D.shape[:2]
        if os.path.exists(r + "ink.png"):
            ink = hisam.from_png(r + "ink.png", (W, H))
        else:
            ink = hisam.strokes(D)
            hisam.to_png(ink, r + "ink.png")
        for l in L:
            if l.get("ocr_box"):
                l["box"] = list(l["ocr_box"])
            if not l.get("path"):
                l["angle"] = slots._line_angle(l)
        M = lonly.view({"L": L, "S": [], "I": [], "size": [W, H], "by": {}})
        keep = {"ops": [o | {"marks": o.get("slots") or o.get("marks") or []} for o in P["ops"] if o.get("kind") == "keep"]}
        kept = lonly.kept_of(keep)
        log = []
        clean, _all = lonly.erase(D, ink, M)
        base, mask = lonly.settle(D, clean, ink, M, keep, log=log)
        zone = np.zeros((H, W), bool)
        gone = np.zeros((H, W), bool)
        for l in L:
            zone |= lonly._region(l, H, W, 0.6)
            if l["id"] not in kept:
                gone |= lonly._region(l, H, W, 0.3)
        left = 0
        for o in ocr.read_lines(base):
            b = o.get("box") or [min(p[0] for p in o["poly"]), min(p[1] for p in o["poly"]), max(p[0] for p in o["poly"]), max(p[1] for p in o["poly"])]
            cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
            if 0 <= cy < H and 0 <= cx < W and gone[cy, cx]:
                left += sum(ch.isalnum() for ch in o.get("text") or "")
        ch = np.abs(base.astype(int) - D.astype(int)).max(axis=2) > 12
        row = {"set": setn, "id": cid, "left": left, "changed": round(100 * float(ch.mean()), 2),
               "outside": round(100 * float((ch & ~zone).mean()), 3), "mask": round(100 * float(mask.mean()), 2), "log": log[-1] if log else ""}
        if a.recheck:
            again = hisam.strokes(base)
            old = ink & gone & mask
            row["ghost"] = round(100 * float((again & gone).sum()) / max(1.0, float(old.sum())), 1)
        rows.append(row)
        print(f"{setn} {cid}: sót {left:3d}" + (f"  nét còn {row['ghost']:5.1f}%" if a.recheck else "") +
              f"  mặt nạ {row['mask']:5.2f}%  đổi {row['changed']:5.2f}%  ngoài {row['outside']:.3f}%", flush=True)
        if a.sheet:
            im = Image.fromarray(np.hstack([D, lonly.ink_image(D, ink, mask, M, kept), base]))
            im.thumbnail((2100, 1400))
            im.save(f"{a.out}/{setn}_{cid}.jpg", quality=84)
n = max(1, len(rows))
tot = {"poster": len(rows), "poster còn sót chữ": sum(r["left"] > 0 for r in rows), "ký tự sót": sum(r["left"] for r in rows),
       "mặt nạ TB %": round(sum(r["mask"] for r in rows) / n, 2), "đổi TB %": round(sum(r["changed"] for r in rows) / n, 2),
       "ngoài TB %": round(sum(r["outside"] for r in rows) / n, 3), "giây / poster": round((time.time() - t0) / n, 1)}
if a.recheck:
    tot["nét còn TB %"] = round(sum(r["ghost"] for r in rows) / n, 1)
    tot["poster nét còn > 5%"] = sum(r["ghost"] > 5 for r in rows)
json.dump({"total": tot, "rows": rows}, open(f"{a.out}/erase_bench.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(tot)
