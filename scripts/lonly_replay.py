#!/usr/bin/env python3
"""MÁY CÁ NHÂN: đo chế độ CHỈ DÒNG CHỮ (textfix/lonly.py) trên các plan đã lưu, KHÔNG cần VLM -- giữ lệnh chữ VLM đã viết, bỏ
mọi lệnh vỏ / chi tiết, dựng trên nền "nháp trừ chữ" -> ảnh  nháp | poster cũ | nền mới | poster mới  + cờ nghi hỏng.

  python -X utf8 scripts/lonly_replay.py <thư mục ra> accept1 [dev2 dev3 form1]      # cả tập
  python -X utf8 scripts/lonly_replay.py <thư mục ra> accept1 --only a01,f02
Lưu ý: lệnh chữ ở đây là của VLM CŨ (nó có thể đã bỏ dòng, vẽ nền riêng cho chữ): phép đo này chỉ trả lời phần NỀN -- vỏ / icon
/ hoạ tiết của nháp có trở lại đúng không, có lộ bóng chữ cũ hay chữ rác không. Phần VLM phải chạy trên máy chủ."""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from textfix import audit, lonly, render, slots   # noqa: E402
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
    for setn in a.sets:
        M0 = json.load(open(f"output/{setn}/{setn}_manifest.json", encoding="utf-8"))
        cases = {c["id"]: c for c in json.load(open(f"bench/{setn}.json", encoding="utf-8"))}
        for cid, m in sorted(M0.items()):
            if not m.get("run_id") or (only and cid not in only):
                continue
            r = f"output/{setn}/{m['run_id']}/v0/"
            img = lambda k: next((r + k + e for e in (".png", ".jpg") if os.path.exists(r + k + e)), None)   # noqa: E731
            if not (os.path.exists(r + "plan.json") and img("draft") and img("plate")):
                continue
            P = json.load(open(r + "plan.json", encoding="utf-8"))
            if not (P.get("slots") or {}).get("L"):
                continue
            T = json.load(open(f"output/{setn}/{m['run_id']}/brief.json", encoding="utf-8"))["texts"]
            D = np.asarray(Image.open(img("draft")).convert("RGB"))
            Pl = np.asarray(Image.open(img("plate")).convert("RGB").resize((D.shape[1], D.shape[0])))
            for l in P["slots"]["L"]:
                if l.get("ocr_box"):
                    l["box"] = list(l["ocr_box"])
                if not l.get("path"):   # góc dựng theo luật hiện hành (plan lưu góc OCR thô): nhiễu về 0, dòng viết dọc về ±90
                    v = slots._vertical(D, {**l, "text": l.get("ocr") or ""})
                    if v.get("vertical"):
                        l.update(angle=v["angle"], poly=v["poly"], ocr=v["text"], size=0.8 * min(l["box"][2] - l["box"][0], l["box"][3] - l["box"][1]))
                    else:
                        l["angle"] = slots._line_angle(l)
            M = lonly.view({"L": P["slots"]["L"], "S": [], "I": [], "size": [D.shape[1], D.shape[0]], "by": {}})
            ops = [{k: v for k, v in o.items() if k != "fix"} | {"marks": o.get("slots") or o.get("marks") or []} for o in P["ops"]]
            P2, flog = lonly.fix_ops({**P, "ops": ops}, M)
            P2, vlog = slots.validate(P2, M, T, product=bool(cases[cid].get("image")))
            log = []
            base = lonly.draft_base(D, Pl, M, P2, log=log)
            poster, res, rlog = render.render(B.page, base, P2, M)
            errs = render.check(P2, M, res)
            fl = audit.flags({**P2, "errors": errs, "errors_first": errs, "log": flog + vlog + log + rlog})
            old = Image.open(img("poster")).convert("RGB")
            tiles = [Image.fromarray(D), old, Image.fromarray(base), Image.fromarray(np.asarray(poster))]
            h = 640
            tiles = [t.resize((round(t.width * h / t.height), h)) for t in tiles]
            sheet = Image.new("RGB", (sum(t.width for t in tiles) + 8 * 3, h), (30, 30, 30))
            x = 0
            for t in tiles:
                sheet.paste(t, (x, 0))
                x += t.width + 8
            sheet.save(f"{a.out}/{setn}_{cid}.jpg", quality=82)
            rows.append({"set": setn, "id": cid, "errors": [e["type"] for e in errs], "flags": fl, "log": log,
                         "old_errors": [e["type"] for e in P.get("errors") or []]})
            print(f"{setn} {cid}: lỗi cũ {rows[-1]['old_errors']} -> mới {rows[-1]['errors']} | {log[0] if log else ''}", flush=True)
json.dump(rows, open(f"{a.out}/rows.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(f"{len(rows)} poster; còn lỗi bộ kiểm: cũ {sum(bool(r['old_errors']) for r in rows)}, mới {sum(bool(r['errors']) for r in rows)}")
