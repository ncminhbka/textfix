#!/usr/bin/env python3
"""MÁY CHỦ: chạy RIÊNG bước LLM đầu luồng (textfix/brief.py) trên các tập dev / accept -- không vẽ, không cần GPU, chỉ cần
TEXTFIX_LLM_URL / TEXTFIX_MODEL như máy chủ. Mỗi ca: request -> brief (texts, prompt_en, log, số token FLUX). Gói một zip nhẹ để
so với brief của lượt chạy trước tại máy cá nhân. Không chạy tập thi (final*).

  %run scripts/brief_probe.py                          # dev2, dev3, mix1, accept1, accept2, form1
  %run scripts/brief_probe.py bench/dev2.json --only d03,d05
  %run scripts/brief_probe.py --fresh                  # bỏ qua cache LLM (gọi lại thật, đo độ ổn định)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SETS = ["bench/dev2.json", "bench/dev3.json", "bench/mix1.json", "bench/accept1.json", "bench/accept2.json", "bench/form1.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", default=SETS)
    ap.add_argument("--only", default="", help="các ca, cách nhau dấu phẩy")
    ap.add_argument("--fresh", action="store_true", help="không đọc cache LLM")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    from server.app import GenerateRequest, form_of
    from textfix import config, ship
    from textfix.brief import SCHEMA, count_tokens, make_brief
    from textfix.llm import json_call
    config.load_env()
    m = config.models()
    call = json_call(m["llm"]["url"], m["llm"]["model"], m["llm"]["key_env"], tag="brief", schema=SCHEMA)
    llm = (lambda system, parts: call(system, parts, fresh=True)) if a.fresh else call
    only = {x for x in a.only.split(",") if x}

    def one(c: dict) -> dict:
        r = c["request"]
        req = GenerateRequest(**{k: v for k, v in r.items() if k in GenerateRequest.model_fields})
        form = form_of(req)
        t0 = time.time()
        try:
            B = make_brief(llm, (req.prompt or "").strip() or None, form if len(form) > 1 else None,
                           {"style": req.style_pref or "auto", "color": req.primary_color or None},
                           bool(c.get("image") or r.get("image_base64")), n_variants=max(1, min(4, int(req.num_images or 1))) - 1,
                           aspect=req.aspect_ratio)
        except Exception as e:
            return {"id": c["id"], "error": f"{type(e).__name__}: {e}"}
        return {"id": c["id"], "s": round(time.time() - t0, 1), "tokens": [count_tokens(B["prompt_en"])] + [count_tokens(v["prompt_en"])
                                                                                                    for v in B["variants"]], **B}

    out_dir = ROOT / "output" / "brief_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    files = []
    for s in a.sets:
        cases = [c for c in json.loads((ROOT / s).read_text(encoding="utf-8")) if not only or c["id"] in only]
        if not cases:
            continue
        t0 = time.time()
        with ThreadPoolExecutor(a.threads) as ex:
            res = list(ex.map(one, cases))
        name = Path(s).stem
        f = out_dir / f"{name}_{stamp}.json"
        f.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        files.append((f, f.name))
        err = [x for x in res if "error" in x]
        ok = [x for x in res if "error" not in x]
        cnt = lambda k: sum(bool(x["log"].get(k)) for x in ok)   # noqa: E731
        print(f"{name}: {len(res)} ca, {len(err)} lỗi, {time.time() - t0:.0f}s | chữ bịa số bị bỏ {cnt('facts_not_from_user')}, "
              f"sửa chép sai {cnt('snapped')}, ô form thêm lại {cnt('form_added')}, ngoặc khách thêm lại {cnt('added')}, "
              f"chèn chữ vào prompt {cnt('quoted_added')}, cắt cho vừa 512 {cnt('trimmed_sentences')}, "
              f"token max {max((t for x in ok for t in x['tokens']), default=0)}")
        for x in err:
            print("  LỖI", x["id"], x["error"])
    if files:
        ship.offer(ship.pack(files, ROOT / "output" / "runs_zip", f"brief_probe_{stamp}"))


if __name__ == "__main__":
    main()
