#!/usr/bin/env python3
"""MÁY CÁ NHÂN: phần ĐO chữ hình dạng khó (textfix/geo.py) trên mọi nháp đã lưu -- không cần VLM, không OCR lại (dùng các ô đã lưu
trong plan.json; OCR trên CPU mất ~1 phút / nháp). Mỗi nháp: tính lại lớp phủ, dò cung, ghi lại ô nào bị phần đo mới đổi. Hai con số cần: trên nháp chữ khó (dev3 g01-g16) bắt được bao nhiêu cung / dòng
sót; trên poster THƯỜNG có bao nhiêu ô bị đổi (mục tiêu 0: mỗi ô đổi sai là một lỗi mới cho phần đang ổn).

  python -X utf8 scripts/geo_measure.py <tệp ra .json> [thư mục lượt chạy ...]   # mặc định: output/dev2, dev3, form1, runs
Ra JSON {key: {"hard": bool, "log": [...], "paths": [{id, geo, box, ocr, size}], "n_L", "n_I"}} + in tóm tắt.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["TEXTFIX_GEO"] = "on"


def main() -> int:
    import numpy as np
    from PIL import Image
    from textfix import slots
    out = Path(sys.argv[1])
    roots = sys.argv[2:] or [str(ROOT / "output" / d) for d in ("dev2", "dev3", "form1", "runs")]
    hard = set()
    mf = ROOT / "output" / "dev3" / "dev3_manifest.json"
    if mf.exists():
        hard = {v["run_id"] for k, v in json.loads(mf.read_text(encoding="utf-8")).items() if k.startswith("g")}
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    drafts = sorted(p for r in roots for p in glob.glob(f"{r}/run_*/v*/draft.png") if os.path.exists(os.path.dirname(p) + "/plan.json"))
    for k, f in enumerate(drafts, 1):
        r = os.path.dirname(f)
        run = os.path.basename(os.path.dirname(r))
        key = f"{run[-4:]}_{os.path.basename(r)}"
        if key in res or not os.path.exists(r + "/plate.png"):
            continue
        D = np.asarray(Image.open(f).convert("RGB"))
        Pl = np.asarray(Image.open(r + "/plate.png").convert("RGB").resize((D.shape[1], D.shape[0])))
        try:
            P = json.load(open(r + "/plan.json", encoding="utf-8"))
            for l in P["slots"]["L"]:
                if l.get("ocr_box"):
                    l["box"] = list(l["ocr_box"])
            M = {**P["slots"], "size": [D.shape[1], D.shape[0]],
                 "by": {x["id"]: x for x in P["slots"]["L"] + P["slots"]["S"] + P["slots"]["I"]}}
            M["geo_log"] = slots.attach_geo(M, D, Pl)
        except Exception as e:
            res[key] = {"hard": run in hard, "error": f"{type(e).__name__}: {e}"}
            print(key, "LỖI", res[key]["error"], flush=True)
            continue
        paths = [{"id": l["id"], "geo": l.get("geo"), "box": [round(v) for v in l["box"]], "ocr": l.get("ocr"), "size": round(l["size"]),
                  "curve": l.get("curve"), "path": l.get("path")} for l in M["L"] if l.get("path") or l.get("geo")]
        res[key] = {"hard": run in hard, "dir": r, "log": [g for g in M.get("geo_log") or []], "paths": paths, "n_L": len(M["L"]),
                    "n_I": len(M["I"])}
        if res[key]["log"]:
            print(f"[{k}/{len(drafts)}] {key} {'KHÓ' if run in hard else 'thường'}: {res[key]['log']}", flush=True)
        if k % 10 == 0:
            out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    H = [v for v in res.values() if v.get("hard")]
    N = [v for v in res.values() if not v.get("hard")]
    geo = lambda v: [g for g in v.get("log", []) if g.startswith("chữ ")]   # noqa: E731
    print(f"nháp chữ khó: {len(H)}, có ô đổi: {sum(1 for v in H if geo(v))}; poster thường: {len(N)}, có ô đổi: {sum(1 for v in N if geo(v))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
