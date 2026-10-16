#!/usr/bin/env python3
"""MÁY CÁ NHÂN: dựng lại poster trên các plan ĐÃ LƯU, KHÔNG gọi VLM -- dùng lại câu trả lời VLM đã lưu (assign, kiểu theo vai),
phần còn lại chạy như trạm thiết kế: resolve -> layout -> nền -> render_units ->
check_units -> autofix_units. -> ảnh  nháp | poster cũ | đơn vị chữ (màu = tập) | poster mới  + rows.json (số đo).

  python -X utf8 scripts/replay.py <thư mục ra> accept2h:accept2 mix1c:mix1      # <thư mục trong output>[:<tên manifest>]
  python -X utf8 scripts/replay.py <thư mục ra> accept2h:accept2 --only e05,h02
Dòng lấy từ plan đã lưu (id khớp với lệnh cũ); độ dày nét (độ đậm mặc định) đo lại như textfix.lines.detect.lines. Nền LaMa gỡ mọi dòng: tính
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


def letter_em(font, text, upper):
    """Thước đo cỡ (độc lập phiên bản mã): cao chữ cao nhất của câu theo HÌNH chữ "H" / "x" của font."""
    import unicodedata
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.ttLib import TTFont
    from textfix.render.fonts import CATALOG, FONTS_DIR
    if font not in _EM:
        try:
            f = TTFont(str(FONTS_DIR / next(iter(CATALOG[font][3].values()))))
            gs, cm, upm = f.getGlyphSet(), f.getBestCmap(), f["head"].unitsPerEm

            def top(ch):
                p = BoundsPen(gs)
                gs[cm[ord(ch)]].draw(p)
                return p.bounds[3] / upm
            _EM[font] = (top("H"), top("x"))
        except Exception:
            _EM[font] = (0.72, 0.53)
    base = [unicodedata.normalize("NFD", c)[0] for c in text if c.isalnum()]
    tall = upper or any(c.isupper() or c.isdigit() or c in "bdfhkltgjpqy" for c in base)
    return _EM[font][0 if tall else 1]


_EM = {}
from textfix import erase, render
from textfix.draw import hisam
from textfix.compose.layout import layout, public
from textfix.compose.assign import resolve
from textfix.lines.printed import _flat
from textfix.lines.detect import reorient   # noqa: E402
from textfix.compose import vlm as cvlm   # noqa: E402
from textfix.render.browser import Browser   # noqa: E402
from textfix.lines.glyph import measure   # noqa: E402

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
            if not (P.get("slots") or {}).get("L") or not P.get("assign"):   # plan trước 14/10 (lệnh html): không dựng lại
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
            reorient(D, ink, L)
            for l in L:
                if l.get("vertical"):   # cỡ lưu trong plan đo theo dòng ngang
                    l["size"] = 0.8 * min(l["box"][2] - l["box"][0], l["box"][3] - l["box"][1])
                if not l.get("path") and abs(l.get("angle") or 0) < 2 and not l.get("stroke"):
                    g = measure(D, l["box"], l.get("ocr") or "", [o["box"] for o in L if o is not l])
                    if g:
                        l["stroke"] = float(g["stroke"])
            M = cvlm.view({"L": L, "size": [W, H]})
            glog = []
            prod = bool((cases.get(cid) or {}).get("image"))
            cpath = f"{a.out}/cache/{setn.replace('/', '_')}_{cid}_clean.png"
            if os.path.exists(cpath):
                clean = np.asarray(Image.open(cpath).convert("RGB"))
            else:
                clean = erase.erase(D, ink, M)[0]
                Image.fromarray(clean).save(cpath)
            A = {"assign": P["assign"], "styles": P.get("style") or {}, "accent": P.get("accent") or []}
            U, rlog, miss = resolve(A, M, T)
            U, llog = layout(U, M, ink, A["styles"], A["accent"], T, base=clean)
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
            pub = {**P, "ops": public(U)}
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
                   and _flat(placed(u)) != _flat("".join(T[k]["text"] for k in u["clients"]))]
            fs = {}
            for rr in res:
                u = next((u for u in U if u["id"] == rr["unit"]), None)
                if u and u.get("set"):
                    fs.setdefault(u["set"], []).append(round(rr["fs"] * (rr.get("sx") or 1), 1))
            uneven = {k: v for k, v in fs.items() if max(v) / max(0.1, min(v)) > 1.03}
            small = [(rr["id"], round(rr["shrink"], 2)) for rr in res if rr.get("shrink", 1) < 0.6]
            # CỠ GIỐNG NHÁP: cao chữ dựng (cỡ cuối x cao chữ của font) / cao nét nháp đo được (lines: h_cap) -- 1 là đúng cỡ nháp
            fit = []
            for rr in res:
                u = next((u for u in U if u["id"] == rr["unit"]), None)
                hs = [(l.get("ink") or {}).get("h_cap") for l in (u or {}).get("Ls") or [] if not (l.get("path") or l.get("vertical"))]
                if u and hs and all(hs) and rr.get("size"):
                    em = letter_em(u["font"], " ".join(u["text"]), "text-transform:uppercase" in u.get("css", ""))
                    fit.append(round(rr["size"] * em / float(np.median(hs)), 3))
            rows.append({"set": setn, "id": cid, "errors": [e["type"] for e in errs], "errs": errs, "sets": sorted(fs), "uneven": uneven,
                         "small": small, "fit": fit, "off": off, "missing": [T[k]["text"] for k in miss], "fill": [u["id"] for u in U if u["kind"] == "text" and not u["clients"]],
                         "log": glog + rlog + llog + dlog + flog})
            print(f"{setn} {cid}: tập {len(fs)}, lệch {uneven or '-'}, co<0.6 {small or '-'}, chữ lệch câu {off or '-'}, thiếu {rows[-1]['missing'] or '-'}, "
                  f"lỗi {rows[-1]['errors'] or '-'}", flush=True)
json.dump(rows, open(f"{a.out}/rows.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(f"{len(rows)} poster; tập lệch cỡ: {sum(bool(r['uneven']) for r in rows)}; chữ lệch câu khách: {sum(len(r['off']) for r in rows)}; "
      f"phần tử co < 0,6: {sum(len(r['small']) for r in rows)}; còn lỗi bộ kiểm: {sum(bool(r['errors']) for r in rows)}")
fits = np.array([x for r in rows for x in r["fit"]])
if len(fits):
    print(f"cỡ so với nháp ({len(fits)} phần tử): trung vị {np.median(fits):.2f}, lệch |log| trung vị {np.median(np.abs(np.log(fits))):.3f}, "
          f"trong +-15%: {np.mean(np.abs(fits - 1) <= 0.15):.0%}, nhỏ hơn 0,8: {np.mean(fits < 0.8):.0%}")
