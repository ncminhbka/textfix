#!/usr/bin/env python3
"""MÁY CÁ NHÂN: dựng lại poster theo chặng B trên các plan ĐÃ LƯU, KHÔNG gọi VLM -- phần ghép câu <-> dòng và kiểu theo vai rút
từ lệnh html cũ (fill.from_ops), phần còn lại chạy như trạm thiết kế: fill.resolve -> fill.layout -> nền -> render_units ->
check_units -> autofix_units. -> ảnh  nháp | poster cũ | đơn vị chữ (màu = tập) | poster mới  + rows.json (số đo).

  python -X utf8 scripts/lonly_replay.py <thư mục ra> accept2h:accept2 mix1c:mix1      # <thư mục trong output>[:<tên manifest>]
  python -X utf8 scripts/lonly_replay.py <thư mục ra> accept2h:accept2 --only e05,h02
Dòng lấy từ plan đã lưu (id khớp với lệnh cũ); độ dày nét (độ đậm mặc định) đo lại như slots.lines. Nền LaMa gỡ mọi dòng: tính
MỘT lần mỗi poster (sau bước dời khung khỏi icon / chấm đầu dòng), lưu ở <thư mục ra>/cache/ -- LaMa trên CPU ~10 s / poster.
Mặt nạ nét chữ: <lượt chạy>/v0/ink.png; chưa có thì chạy Hi-SAM tại máy rồi lưu lại."""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from textfix import erase, fill, hisam, lonly, render, slots   # noqa: E402
from textfix.browser import Browser   # noqa: E402
from textfix.glyph import measure   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("sets", nargs="+")
ap.add_argument("--only", default="")
a = ap.parse_args()
only = {x for x in a.only.split(",") if x}
os.makedirs(f"{a.out}/cache", exist_ok=True)
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
            if not (P.get("slots") or {}).get("L") or not (P.get("ops") or P.get("assign")):
                continue
            T = json.load(open(f"output/{setn}/{m['run_id']}/brief.json", encoding="utf-8"))["texts"]
            D = np.asarray(Image.open(dr).convert("RGB"))
            H, W = D.shape[:2]
            if os.path.exists(r + "ink.png"):
                ink = hisam.from_png(r + "ink.png", (W, H))
            else:
                ink = hisam.strokes(D)
                hisam.to_png(ink, r + "ink.png")
            L = P["slots"]["L"]
            for l in L:
                if l.get("ocr_box"):
                    l["box"] = list(l["ocr_box"])
                l.pop("mark", None)
                if not l.get("path"):   # góc dựng theo luật hiện hành (plan lưu góc OCR thô): nhiễu về 0, dòng viết dọc về ±90
                    v = slots._vertical(D, {**l, "text": l.get("ocr") or ""})
                    if v.get("vertical"):
                        l.update(angle=v["angle"], poly=v["poly"], ocr=v["text"], size=0.8 * min(l["box"][2] - l["box"][0], l["box"][3] - l["box"][1]),
                                 vertical=True)
                    else:
                        l["angle"] = slots._line_angle(l)
                if not l.get("path") and abs(l.get("angle") or 0) < 2 and not l.get("stroke"):
                    g = measure(D, l["box"], l.get("ocr") or "", [o["box"] for o in L if o is not l])
                    if g:
                        l["stroke"] = float(g["stroke"])
            M = lonly.view({"L": L, "size": [W, H]})
            for l in M["L"]:
                l.pop("dot", None)
            glog = slots._lead_dots(M["L"], ink) + slots._trim_lead(M["L"], ink)
            prod = bool((cases.get(cid) or {}).get("image"))
            cpath = f"{a.out}/cache/{setn.replace('/', '_')}_{cid}_clean.png"
            if os.path.exists(cpath):
                clean = np.asarray(Image.open(cpath).convert("RGB"))
            else:
                clean = erase.erase(D, ink, M)[0]
                Image.fromarray(clean).save(cpath)
            # plan chặng B: câu trả lời VLM đã ở khuôn ghép; plan cũ: rút từ lệnh html
            A = {"assign": P["assign"], "styles": P.get("style") or {}, "accent": P.get("accent") or []} if P.get("assign") \
                else fill.from_ops(P, T)
            U, rlog, miss = fill.resolve(A, M, T, product=prod)
            U, llog = fill.layout(U, M, ink, A["styles"], A["accent"], T)
            elog = []
            base, mask = erase.settle(D, clean, ink, M, {"ops": U}, elog)
            poster, res, dlog = render.render_units(B.page, base, U, M)
            errs = render.check_units(U, res)
            sev = lambda E: sum(e["type"] in render.SEVERE for e in E)   # noqa: E731
            flog = []
            for k in range(render.FIX_ROUNDS if sev(errs) else 0):
                U2, f2 = render.autofix_units(U, errs, res)
                if not f2:
                    break
                p2, res2, _ = render.render_units(B.page, base, U2, M)
                e2 = render.check_units(U2, res2)
                flog += f2
                if (sev(e2), len(e2)) >= (sev(errs), len(errs)):
                    break
                poster, U, errs, res = p2, U2, e2, res2
            old = Image.open(next(iter(glob.glob(r + "poster.*")))).convert("RGB")
            pub = {**P, "ops": fill.public(U)}
            tiles = [Image.fromarray(D), old, render.ops_image(D, M, pub), Image.fromarray(np.asarray(poster))]
            h = 720
            tiles = [t.resize((round(t.width * h / t.height), h)) for t in tiles]
            sheet = Image.new("RGB", (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), h), (30, 30, 30))
            x = 0
            for t in tiles:
                sheet.paste(t, (x, 0))
                x += t.width + 8
            sheet.save(f"{a.out}/{setn.replace('/', '_')}_{cid}.jpg", quality=82)
            Image.fromarray(np.asarray(poster)).save(f"{a.out}/{setn.replace('/', '_')}_{cid}_poster.jpg", quality=90)
            # SỐ ĐO: chữ đặt khác câu khách (phải 0 ngoài dòng lấp), tập dựng lệch cỡ (phải 0), phần tử co < 0,6, câu khách thiếu
            import re   # số bước / ngoặc kép do mã đặt lại không tính là lệch
            placed = lambda u: re.sub(r"^\d{1,2}\.\s+", "", "".join(x for e in u.get("els") or [] for x in e["lines"]).strip().strip("“”"))  # noqa: E731
            off = [u["id"] for u in U if u["kind"] == "text" and u["clients"]
                   and slots._flat(placed(u)) != slots._flat("".join(T[k]["text"] for k in u["clients"]))]
            fs = {}
            for rr in res:
                u = next((u for u in U if u["id"] == rr["unit"]), None)
                if u and u.get("set"):
                    fs.setdefault(u["set"], []).append(round(rr["fs"] * (rr.get("sx") or 1), 1))
            uneven = {k: v for k, v in fs.items() if max(v) / max(0.1, min(v)) > 1.03}
            small = [(rr["id"], round(rr["shrink"], 2)) for rr in res if rr.get("shrink", 1) < 0.6]
            rows.append({"set": setn, "id": cid, "errors": [e["type"] for e in errs], "sets": sorted(fs), "uneven": uneven,
                         "small": small, "off": off, "missing": [T[k]["text"] for k in miss], "fill": [u["id"] for u in U if u["kind"] == "text" and not u["clients"]],
                         "log": glog + rlog + llog + dlog + flog})
            print(f"{setn} {cid}: tập {len(fs)}, lệch {uneven or '-'}, co<0.6 {small or '-'}, chữ lệch câu {off or '-'}, thiếu {rows[-1]['missing'] or '-'}, "
                  f"lỗi {rows[-1]['errors'] or '-'}", flush=True)
json.dump(rows, open(f"{a.out}/rows.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(f"{len(rows)} poster; tập lệch cỡ: {sum(bool(r['uneven']) for r in rows)}; chữ lệch câu khách: {sum(len(r['off']) for r in rows)}; "
      f"phần tử co < 0,6: {sum(len(r['small']) for r in rows)}; còn lỗi bộ kiểm: {sum(bool(r['errors']) for r in rows)}")
