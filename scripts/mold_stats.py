"""MÁY CÁ NHÂN: dấu hiệu "OCR làm khuôn" trên dev2 + dev3 (dựng lại plan server, ~15 phút) -> <ra>/mold.json, mỗi poster:
A_missing câu thiếu, B_uneven nhóm dòng lệch cỡ, C_ragged lệch lề, D_ocr_miss chữ lớn OCR không đọc, E_rows số hàng nháp khác số
mục, F_orphan icon trơ, small số khối chữ co < 0.7. Xem docs/PLAN_BLOCKS.md.

  python -X utf8 scripts/mold_stats.py <thư mục ra>
"""
import json, os, sys, collections
import numpy as np
from PIL import Image
sys.path.insert(0, ".")
from textfix import render, slots
from textfix.browser import Browser
S = sys.argv[1]
rows = []
with Browser() as B:
    for setn in ("dev2", "dev3"):
        M = json.load(open(f"output/{setn}/{setn}_manifest.json", encoding="utf-8"))
        cases = {c["id"]: c for c in json.load(open(f"bench/{setn}.json", encoding="utf-8"))}
        for cid, m in sorted(M.items()):
            if not m.get("run_id"): continue
            T = json.load(open(f"output/{setn}/{m['run_id']}/brief.json", encoding="utf-8"))["texts"]
            for v in range(m["num_images"]):
                r = f"output/{setn}/{m['run_id']}/v{v}/"
                if not os.path.exists(r + "plan.json"): continue
                P = json.load(open(r + "plan.json", encoding="utf-8"))
                if P.get("mode") == "image": continue
                D = np.asarray(Image.open(r + "draft.png").convert("RGB"))
                pl = r + "plate.png" if os.path.exists(r + "plate.png") else r + "plate_raw.jpg"
                Pl = np.asarray(Image.open(pl).convert("RGB").resize((D.shape[1], D.shape[0])))
                by = {x["id"]: x for x in P["slots"]["L"] + P["slots"]["S"] + P["slots"]["I"]}
                Mm = {**P["slots"], "size": [D.shape[1], D.shape[0]], "by": by}
                _, res, _ = render.render(B.page, slots.base_image(D, Pl, Mm, P), P, Mm)
                R = {x["id"]: x for x in res}
                ops = {o["id"]: o for o in P["ops"]}
                L = P["slots"]["L"]
                hmed = float(np.median([l["box"][3] - l["box"][1] for l in L])) if L else 0
                # nhóm dòng: lệnh chữ một dòng, xếp chồng sát nhau, cỡ nháp gần nhau, thẳng mép trái hoặc tâm
                tx = [o for o in P["ops"] if o["kind"] == "text" and o["id"] in R and R[o["id"]].get("fs") and not R[o["id"]].get("angle")]
                tx.sort(key=lambda o: R[o["id"]]["ink"][1])
                def dsize(o):
                    s = [by[k]["size"] for k in o["marks"] if k.startswith("L") and k in by]
                    return float(np.median(s)) if s else None
                groups, cur = [], []
                for o in tx:
                    if len([k for k in o["marks"] if k.startswith("L")]) != 1 or dsize(o) is None:
                        if len(cur) >= 3: groups.append(cur)
                        cur = []; continue
                    if cur:
                        p = cur[-1]; a, b = by[[k for k in p["marks"] if k.startswith("L")][0]]["box"], by[[k for k in o["marks"] if k.startswith("L")][0]]["box"]
                        h = max(a[3] - a[1], b[3] - b[1])
                        ok = 0 <= b[1] - a[3] <= 2.2 * h and max(dsize(p), dsize(o)) <= 1.35 * min(dsize(p), dsize(o)) and \
                            (abs(a[0] - b[0]) <= 0.8 * h or abs((a[0] + a[2]) - (b[0] + b[2])) / 2 <= 0.8 * h)
                        if not ok:
                            if len(cur) >= 3: groups.append(cur)
                            cur = []
                    cur.append(o)
                if len(cur) >= 3: groups.append(cur)
                unev = rag = 0
                for g in groups:
                    fs = [R[o["id"]]["fs"] for o in g]
                    if max(fs) >= 1.2 * min(fs): unev += 1
                    x0 = [R[o["id"]]["ink"][0] for o in g]; cx = [(R[o["id"]]["ink"][0] + R[o["id"]]["ink"][2]) / 2 for o in g]
                    if min(np.ptp(x0), np.ptp(cx)) > 0.5 * float(np.median(fs)): rag += 1
                # chữ lớn OCR không đọc: >= 2 chi tiết I cao cỡ chữ (>= 0.8 cao dòng trung vị) mà VLM skip / viết chữ vào
                bigI = [c for c in P["slots"]["I"] if hmed and c["box"][3] - c["box"][1] >= 0.8 * hmed and c["box"][2] - c["box"][0] <= 6 * (c["box"][3] - c["box"][1])]
                own = {k: o for o in P["ops"] for k in o["marks"]}
                missI = [c for c in bigI if own.get(c["id"], {}).get("kind") in ("skip", "text")]
                # số hàng danh sách nháp (nhóm L lớn nhất) so với số mục list_item của khách
                n_item = sum(t["role"] == "list_item" for t in T)
                Ls = sorted(L, key=lambda l: l["box"][1]); best, cur = [], []
                for l in Ls:
                    if cur and 0 <= l["box"][1] - cur[-1]["box"][3] <= 2.2 * (l["box"][3] - l["box"][1]) and abs(l["box"][0] - cur[-1]["box"][0]) <= 0.8 * (l["box"][3] - l["box"][1]) \
                            and max(l["size"], cur[-1]["size"]) <= 1.35 * min(l["size"], cur[-1]["size"]):
                        cur.append(l)
                    else:
                        if len(cur) > len(best): best = cur
                        cur = [l]
                if len(cur) > len(best): best = cur
                rowmis = bool(n_item >= 3 and len(best) >= 3 and len(best) != n_item)
                # icon trơ
                dead = {k for o in P["ops"] if o["kind"] == "skip" for k in o["marks"] if k.startswith("L")}
                orphan = sum(1 for o in P["ops"] if o["kind"] == "icon" for k in o["marks"]
                             if (by.get(k, {}).get("near") or "").split()[:1] in (["left"], ["on"]) and (by[k]["near"].split()[-1] in dead))
                small = sum(1 for x in res if x["id"] in ops and ops[x["id"]]["kind"] == "text" and x["shrink"] < 0.7)
                rows.append({"k": f"{setn}:{cid}_v{v}", "grp": cid[0] if setn == "dev3" else "d", "A_missing": len(P.get("missing") or []),
                             "B_uneven": unev, "C_ragged": rag, "D_ocr_miss": len(missI) if len(missI) >= 2 else 0, "E_rows": int(rowmis),
                             "F_orphan": orphan, "small": small, "n_groups": len(groups)})
        print(setn, len(rows), flush=True)
json.dump(rows, open(S + "/mold.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
