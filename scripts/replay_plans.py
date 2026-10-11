"""MÁY CÁ NHÂN: dựng lại plan.json của một tập dev (output/<tập>/, giữ lệnh VLM) bằng validate + render + autofix HIỆN TẠI -> ảnh
<ra>/rr_<tập>/<ca>_v<i>.jpg, per.json (lỗi server / lỗi mới / log), tổng lỗi trước -> sau. Dùng để kiểm hồi quy sau mỗi sửa.

  python -X utf8 scripts/replay_plans.py <thư mục ra> dev3              # cả tập
  python -X utf8 scripts/replay_plans.py <thư mục ra> dev3 g11_v1,g12_v1  # vài ảnh (in log từng ảnh)
  RRTAG=_base ...   # hậu tố thư mục ra (so mã cũ / mới)
"""
import json, os, sys, collections
import numpy as np
from PIL import Image
sys.path.insert(0, ".")
from textfix import render, slots
from textfix.browser import Browser
S, setn = sys.argv[1], sys.argv[2]
keys = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else None
out = f"{S}/rr_{setn}" + os.environ.get("RRTAG", ""); os.makedirs(out, exist_ok=True)
M = json.load(open(f"output/{setn}/{setn}_manifest.json", encoding="utf-8"))
cases = {c["id"]: c for c in json.load(open(f"bench/{setn}.json", encoding="utf-8"))}
tot0, tot1 = collections.Counter(), collections.Counter()
per = {}
with Browser() as B:
    for cid, m in sorted(M.items()):
        if not m.get("run_id"): continue
        T = json.load(open(f"output/{setn}/{m['run_id']}/brief.json", encoding="utf-8"))["texts"]
        for v in range(m["num_images"]):
            k = f"{cid}_v{v}"
            if keys and k not in keys: continue
            r = f"output/{setn}/{m['run_id']}/v{v}/"
            if not os.path.exists(r + "plan.json"): continue
            P = json.load(open(r + "plan.json", encoding="utf-8"))
            if P.get("mode") == "image": continue
            D = np.asarray(Image.open(r + "draft.png").convert("RGB"))
            pl = r + "plate.png" if os.path.exists(r + "plate.png") else r + "plate_raw.jpg"
            Pl = np.asarray(Image.open(pl).convert("RGB").resize((D.shape[1], D.shape[0])))
            Mm = {**P["slots"], "size": [D.shape[1], D.shape[0]], "by": {x["id"]: x for x in P["slots"]["L"] + P["slots"]["S"] + P["slots"]["I"]}}
            for e in P.get("errors") or []: tot0[e["type"]] += 1
            P = {**P, "ops": [{kk: w for kk, w in o.items() if kk != "fix"} for o in P["ops"]]}
            P, vlog = slots.validate(P, Mm, T, product=bool(cases[cid].get("image")))
            poster, res, log = render.render(B.page, slots.base_image(D, Pl, Mm, P), P, Mm)
            errs = render.check(P, Mm, res)
            sev = lambda E: sum(e["type"] in slots.SEVERE for e in E)
            for i in range(render.FIX_ROUNDS if sev(errs) else 0):
                P2, flog = render.autofix(P, Mm, errs, res)
                if not flog: break
                p2, r2, l2 = render.render(B.page, slots.base_image(D, Pl, Mm, P2), P2, Mm)
                e2 = render.check(P2, Mm, r2)
                log += flog
                if (sev(e2), len(e2)) >= (sev(errs), len(errs)): break
                poster, P, errs, res = p2, P2, e2, r2
            for e in errs: tot1[e["type"]] += 1
            per[k] = {"server": [(e["type"], e.get("ops")) for e in json.load(open(r + "plan.json", encoding="utf-8")).get("errors") or []],
                      "new": [(e["type"], e.get("ops")) for e in errs], "log": [l for l in vlog + log if "->" in l or "dải màu" in l]}
            Image.fromarray(np.asarray(poster)).save(f"{out}/{k}.jpg", quality=88)
            if keys: print(k, [e["type"] for e in errs], [l for l in vlog + log if any(w in l for w in ("viền", "dải màu", "trùng câu", "đi kèm", "màu"))][:6])
print("lỗi cuối server:", dict(tot0)); print("lỗi cuối dựng lại:", dict(tot1))
json.dump(per, open(f"{out}/per.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
