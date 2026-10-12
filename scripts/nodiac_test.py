#!/usr/bin/env python3
"""MÁY CHỦ: FLUX.2 klein 4B DISTILL viết tiếng Việt KHÔNG DẤU có đúng chữ hơn không? Vẽ lại các nháp trong bench/base_test.json
(mục gen: đúng prompt + seed + cỡ của nháp distill đã có) nhưng BỎ DẤU toàn bộ prompt ("Giảm 50%" -> "Giam 50%") -> so từng cặp
nháp có dấu (đã có) / không dấu.

  !pkill -f server/app.py                  # tắt máy chủ TextFix trước (nó giữ FLUX trên cả hai card)
  %run scripts/nodiac_test.py              # 24 nháp, mọi GPU -> một zip < 24 MB (JPG 92), tự tải về
  %run scripts/nodiac_test.py --only dev3_p04_v0,dev2_d20_v0
  %run scripts/nodiac_test.py --ship-only
Ra output/nodiac_test/<key>.png + info.json (prompt đã bỏ dấu, thời gian).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)


def strip_marks(s: str) -> str:
    """Bỏ dấu tiếng Việt: đ -> d, bỏ mọi dấu thanh / mũ / móc; ký tự khác giữ nguyên."""
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from PIL import Image
    from textfix import config, multigpu, ship
    from textfix.flux import Flux
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="?", type=Path, default=ROOT / "bench" / "base_test.json")
    ap.add_argument("--only", default="", help="key, cách nhau dấu phẩy")
    ap.add_argument("--runs", type=Path, default=ROOT / "output" / "runs", help="nơi có product.png của lượt gốc (ca có ảnh sản phẩm)")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "output" / "nodiac_test")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ship-only", action="store_true")
    multigpu.add_args(ap)
    a = ap.parse_args(argv)
    config.load_env()
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    cases = [c for c in json.loads(a.cases.read_text(encoding="utf-8"))["gen"] if not only or c["key"] in only]
    a.out.mkdir(parents=True, exist_ok=True)
    info_f = a.out / "info.json"
    todo = [] if a.ship_only else [c for c in cases if a.force or not (a.out / f"{c['key']}.png").exists()]
    if todo and multigpu.fanout(a, __file__, argv):
        todo = []
    todo = multigpu.shard(todo, a)
    info = {} if a.shard else multigpu.merge(info_f, deep=("runs",))
    info_out = multigpu.part(info_f, a)
    if todo:
        t0 = time.time()
        F = Flux("distill").load()
        print(f"nạp FLUX distill {time.time() - t0:.0f}s; {len(todo)} nháp")
    for k, c in enumerate(todo, 1):
        prompt = strip_marks(c["prompt_en"])
        prod = a.runs / c["run_id"] / "product.png"
        refs = [np.asarray(Image.open(prod).convert("RGB"))] if prod.is_file() else None
        try:
            t1 = time.time()
            img = F.generate(prompt, c["W"], c["H"], c["seed"], refs=refs)
        except Exception as e:
            print(f"{c['key']}: LỖI {type(e).__name__}: {str(e)[:300]}")
            continue
        Image.fromarray(img).save(a.out / f"{c['key']}.png")
        info.setdefault("gen", {"runs": {}})["runs"][c["key"]] = {"seed": c["seed"], "s": round(time.time() - t1, 1), "prompt": prompt}
        info_out.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{k}/{len(todo)}] {c['key']}: {time.time() - t1:.1f}s")
    if a.shard:
        return 0
    info_out.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    multigpu.merge(info_f, deep=("runs",))
    pngs = sorted(a.out.glob("*.png"))
    if not pngs:
        print("chưa có ảnh nào để gói")
        return 1
    files = [(info_f, "info.json")]
    for p in pngs:
        j = a.out / "ship" / f"{p.stem}.jpg"
        if not j.exists() or j.stat().st_mtime < p.stat().st_mtime:
            j.parent.mkdir(parents=True, exist_ok=True)
            Image.open(p).convert("RGB").save(j, quality=92)
        files.append((j, j.name))
    print(f"xong: {len(pngs)} nháp không dấu ở {a.out}")
    if not a.no_ship:
        ship.offer(ship.pack(files, a.out.parent / "nodiac_test_zip", "nodiac"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
