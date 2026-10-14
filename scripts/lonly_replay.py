#!/usr/bin/env python3
"""MÁY CÁ NHÂN: dựng lại poster trên các plan đã lưu, KHÔNG cần VLM -- giữ lệnh chữ VLM đã viết, chạy lại phần còn lại của trạm
thiết kế (gỡ chữ Hi-SAM + LaMa, rà lệnh, dựng, kiểm) -> ảnh  nháp | poster cũ | nét chữ + mặt nạ | nền mới | poster mới  + cờ.

  python -X utf8 scripts/lonly_replay.py <thư mục ra> accept2 [mix1c:mix1]     # cả tập (<thư mục trong output>[:<tên manifest>])
  python -X utf8 scripts/lonly_replay.py <thư mục ra> accept2 --only a01,f02
Dòng lấy từ plan ĐÃ LƯU (số ô khớp với lệnh VLM cũ), không đo lại bằng slots.build: phép đo này trả lời phần NỀN và phần RÀ
LỆNH -- chữ cũ có sạch không, vỏ / icon của nháp có nguyên không, chữ mới có đè lên thứ gì không. Bước đo dòng mới (cụm nét
chưa ai nhận, chấm đầu dòng) và VLM phải chạy trên máy chủ.
Mặt nạ nét chữ: <lượt chạy>/v0/ink.png; chưa có thì chạy Hi-SAM tại máy (CPU ~13 s) rồi lưu lại."""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from textfix import audit, hisam, lonly, render, slots   # noqa: E402
from textfix.browser import Browser   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("sets", nargs="+")
ap.add_argument("--only", default="")
a = ap.parse_args()
only = {x for x in a.only.split(",") if x}
os.makedirs(a.out, exist_ok=True)
rows = []
with Browser() as B:
    for spec in a.sets:
        setn, mfn = spec.split(":") if ":" in spec else (spec, spec)
        M0 = json.load(open(f"output/{setn}/{mfn}_manifest.json", encoding="utf-8"))
        cases = {c["id"]: c for c in json.load(open(f"bench/{mfn}.json", encoding="utf-8"))} if os.path.exists(f"bench/{mfn}.json") else {}
        for cid, m in sorted(M0.items()):
            if not m.get("run_id") or (only and cid not in only):
                continue
            r = f"output/{setn}/{m['run_id']}/v0/"
            dr = next(iter(glob.glob(r + "draft.*")), None)
            if not (os.path.exists(r + "plan.json") and dr):
                continue
            P = json.load(open(r + "plan.json", encoding="utf-8"))
            if not (P.get("slots") or {}).get("L"):
                continue
            T = json.load(open(f"output/{setn}/{m['run_id']}/brief.json", encoding="utf-8"))["texts"]
            D = np.asarray(Image.open(dr).convert("RGB"))
            H, W = D.shape[:2]
            if os.path.exists(r + "ink.png"):
                ink = hisam.from_png(r + "ink.png", (W, H))
            else:
                ink = hisam.strokes(D)
                hisam.to_png(ink, r + "ink.png")
            for l in P["slots"]["L"]:
                if l.get("ocr_box"):
                    l["box"] = list(l["ocr_box"])
                l.pop("mark", None)   # plan cũ: dấu đầu dòng do mã gắn -- nay icon là của nháp
                if not l.get("path"):   # góc dựng theo luật hiện hành (plan lưu góc OCR thô): nhiễu về 0, dòng viết dọc về ±90
                    v = slots._vertical(D, {**l, "text": l.get("ocr") or ""})
                    if v.get("vertical"):
                        l.update(angle=v["angle"], poly=v["poly"], ocr=v["text"], size=0.8 * min(l["box"][2] - l["box"][0], l["box"][3] - l["box"][1]))
                    else:
                        l["angle"] = slots._line_angle(l)
            M = lonly.view({"L": P["slots"]["L"], "S": [], "I": [], "size": [W, H], "by": {}})
            for l in M["L"]:
                l.pop("dot", None)
            dots = slots._lead_dots(M["L"], ink) + slots._trim_lead(M["L"], ink)
            prod = bool((cases.get(cid) or {}).get("image"))
            ops = [{k: v for k, v in o.items() if k != "fix"} | {"marks": o.get("slots") or o.get("marks") or []} for o in P["ops"]]
            P2, flog = lonly.fix_ops({**P, "ops": ops}, M)
            P2, slog = lonly.split_far(P2, M)
            P2, vlog = slots.validate(P2, M, T, product=prod)
            P2, glog = lonly.fill_gaps(P2, M, T, product=prod)
            P2, mlog = lonly.marks_rule(P2, M)
            log = []
            clean, _all = lonly.erase(D, ink, M)
            base, mask = lonly.settle(D, clean, ink, M, P2, log)
            poster, res, rlog = render.render(B.page, base, P2, M)
            errs = render.check(P2, M, res)
            fl = audit.flags({**P2, "errors": errs, "errors_first": errs, "log": dots + flog + slog + vlog + glog + mlog + log + rlog,
                              "erase": {"log": log, "orphans": []}})
            old = Image.open(next(iter(glob.glob(r + "poster.*")))).convert("RGB")
            tiles = [Image.fromarray(D), old, Image.fromarray(lonly.ink_image(D, ink, mask, M, lonly.kept_of(P2))), Image.fromarray(base),
                     Image.fromarray(np.asarray(poster))]
            h = 640
            tiles = [t.resize((round(t.width * h / t.height), h)) for t in tiles]
            sheet = Image.new("RGB", (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), h), (30, 30, 30))
            x = 0
            for t in tiles:
                sheet.paste(t, (x, 0))
                x += t.width + 8
            sheet.save(f"{a.out}/{setn}_{cid}.jpg", quality=82)
            Image.fromarray(np.asarray(poster)).save(f"{a.out}/{setn}_{cid}_poster.jpg", quality=90)
            rows.append({"set": setn, "id": cid, "errors": [e["type"] for e in errs], "flags": fl, "log": log, "marks": mlog, "split": slog,
                         "old_errors": [e["type"] for e in P.get("errors") or []]})
            print(f"{setn} {cid}: lỗi cũ {rows[-1]['old_errors']} -> mới {rows[-1]['errors']} | {log[0] if log else ''}"
                  + (f" | dấu: {mlog}" if mlog else "") + (f" | tách: {slog}" if slog else ""), flush=True)
json.dump(rows, open(f"{a.out}/rows.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(f"{len(rows)} poster; còn lỗi bộ kiểm: cũ {sum(bool(r['old_errors']) for r in rows)}, mới {sum(bool(r['errors']) for r in rows)}")
