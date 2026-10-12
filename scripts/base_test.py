#!/usr/bin/env python3
"""MÁY CHỦ: thử FLUX.2 klein 4B BASE (50 bước, CFG 4.0) ở cả hai việc, so với bản distill đang dùng. Ca thử: bench/base_test.json.

  gen    vẽ lại ĐÚNG prompt + seed + cỡ của các nháp distill đã có (prompt dài thật trong prompt_test.txt, ca đếm số lượng, chữ khó
         có dấu) -> so từng cặp nháp distill / base.
  erase  xoá lại các nháp DISTILL mà bản distill xoá hỏng (sót chữ, đổi hình / bố cục, vỏ lúc xoá lúc giữ) + vài ca đối chứng,
         cùng lời dặn (flux.ERASE_PROMPT), cùng seed -> so từng cặp bản xoá distill / base trên CÙNG một nháp.
  chain  (--chain) xoá luôn các nháp base vừa vẽ: đường đi thật nếu đổi cả hai bước sang base.

  !pkill -f server/app.py                        # tắt máy chủ TextFix trước: nó giữ một bản distill trên mỗi card, không đủ VRAM
  %run scripts/base_test.py                      # mọi ca, mọi GPU -> zip < 24 MB, tự tải về
  %run scripts/base_test.py --what erase --only dev2_d09_v0,dev3_g04_v0
  %run scripts/base_test.py --steps 28           # thử ít bước hơn (ra thư mục riêng output/base_test_s28)
  %run scripts/base_test.py --what erase --why sót --neg "text, letters, numbers, words, typography, captions"
                                                 # PROMPT ÂM cho bước xoá (chỉ base có): 13 ca sót chữ -> output/base_test_neg
  %run scripts/base_test.py --what erase --why sót --neg "..." --cfg 6 --tag neg6     # đổi CFG, thư mục riêng theo --tag
  %run scripts/base_test.py --ship-only          # chỉ gói lại kết quả đã có
Chạy lại sau khi đứt: việc đã có ảnh thì bỏ qua (--force để chạy lại).

Nháp distill lấy ở output/runs/<run_id>/v<i>/draft.png (máy chủ còn giữ lượt dev2 / dev3); thiếu thì tìm --drafts/<key>.png, vẫn
thiếu thì bỏ ca đó và báo. Ra output/base_test/{gen,erase,chain}/<key>.png + info.json (thời gian từng việc, độ lệch khi căn).
Zip gửi ảnh JPG chất lượng 95 (--png: gửi PNG gốc). Chấm ở máy cá nhân.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

NEED_GB = 18   # một bản FLUX (DiT + bộ mã hoá chữ + VAE) ~ 17 GB


def _vram_ok() -> bool:
    """Mỗi card phải trống >= NEED_GB (máy chủ TextFix đang chạy thì không đủ)."""
    try:
        import torch
        free = [torch.cuda.mem_get_info(k)[0] / 2**30 for k in range(torch.cuda.device_count())]
    except Exception:
        return True
    if any(f < NEED_GB for f in free):
        print(f"VRAM trống mỗi card: {[round(f, 1) for f in free]} GB, cần >= {NEED_GB} GB -- tắt máy chủ TextFix "
              "(và kernel notebook khác đang giữ model) rồi chạy lại")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from PIL import Image
    from textfix import config, multigpu, ship
    from textfix.flux import ERASE_PROMPT, Flux, register
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="?", type=Path, default=ROOT / "bench" / "base_test.json")
    ap.add_argument("--what", default="gen,erase", help="gen,erase (phẩy)")
    ap.add_argument("--chain", action="store_true", help="xoá luôn các nháp base vừa vẽ")
    ap.add_argument("--only", default="", help="key, cách nhau dấu phẩy (dev3_p04_v0,dev2_d20_v0)")
    ap.add_argument("--why", default="", help="chỉ các ca có nhóm lỗi chứa chuỗi này (sót, đổi hình, vỏ, đối chứng)")
    ap.add_argument("--steps", type=int, default=0, help="số bước khử nhiễu (0 = mặc định của base: 50)")
    ap.add_argument("--neg", default="", help="prompt âm cho bước XOÁ (erase / chain); bước vẽ nháp không dùng")
    ap.add_argument("--cfg", type=float, default=0, help="CFG (0 = mặc định của base: 4.0)")
    ap.add_argument("--tag", default="", help="hậu tố thư mục ra (mặc định: neg khi có --neg, cfg<x> khi có --cfg)")
    ap.add_argument("--runs", type=Path, default=ROOT / "output" / "runs", help="nơi máy chủ giữ các lượt chạy (nháp distill)")
    ap.add_argument("--drafts", type=Path, default=ROOT / "output" / "base_test_drafts", help="nháp distill dự phòng: <key>.png")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ship-only", action="store_true")
    ap.add_argument("--png", action="store_true", help="gửi PNG gốc thay vì JPG 95")
    ap.add_argument("--zip-mb", type=float, default=None)
    multigpu.add_args(ap)
    a = ap.parse_args(argv)
    config.load_env()
    C = json.loads(a.cases.read_text(encoding="utf-8"))
    tag = a.tag or "_".join(([f"cfg{a.cfg:g}"] if a.cfg else []) + (["neg"] if a.neg else []))
    out = a.out or ROOT / "output" / ("base_test" + (f"_s{a.steps}" if a.steps else "") + (f"_{tag}" if tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    what = {x.strip() for x in a.what.split(",")}
    pick = lambda rows: [c for c in rows if (not only or c["key"] in only) and a.why in c.get("why", "")]   # noqa: E731

    def draft_of(c: dict) -> Path | None:
        return next((p for p in (a.runs / c["run_id"] / f"v{c['v']}" / "draft.png", a.drafts / f"{c['key']}.png") if p.is_file()),
                    None)

    # việc: (loại, ca). chain đứng sau gen của chính nó (cùng tiến trình: chia việc theo CA, không theo việc lẻ)
    gen = pick(C["gen"]) if "gen" in what else []
    era = pick(C["erase"]) if "erase" in what else []
    lost = [c["key"] for c in era if draft_of(c) is None]
    if lost and not a.shard:
        print(f"THIẾU nháp distill của {len(lost)} ca xoá (không có trong {a.runs} lẫn {a.drafts}), bỏ qua: {lost}")
    era = [c for c in era if c["key"] not in lost]
    units = [[("gen", c)] + ([("chain", c)] if a.chain else []) for c in gen] + [[("erase", c)] for c in era]
    done = lambda j: (out / j[0] / f"{j[1]['key']}.png").exists()   # noqa: E731
    todo = [] if a.ship_only else [u for u in units if a.force or not all(done(j) for j in u)]
    info_f = out / "info.json"
    if todo and not a.shard and not _vram_ok():
        return 2
    if todo and multigpu.fanout(a, __file__, argv):   # mỗi card một tiến trình con, mỗi con một bản base
        todo = []
    todo = multigpu.shard(todo, a)
    info = {} if a.shard else multigpu.merge(info_f, deep=("runs",))
    info_out = multigpu.part(info_f, a)
    save = lambda: info_out.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")   # noqa: E731
    if todo:
        t0 = time.time()
        F = Flux("base", steps=a.steps or None).load()
        F.cfg = a.cfg or F.cfg
        info["flux"] = {**F.info(), "erase_prompt": ERASE_PROMPT, "erase_neg": a.neg}
        n = sum(len(u) for u in todo)
        print(f"nạp FLUX base {time.time() - t0:.0f}s ({F.steps} bước, CFG {F.cfg}); {n} việc")
    k = 0
    for u in todo:
        for kind, c in u:
            k += 1
            dst = out / kind / f"{c['key']}.png"
            if dst.exists() and not a.force:
                continue
            dst.parent.mkdir(exist_ok=True)
            rec = {"seed": c["seed"], "why": c.get("why")}
            try:
                t1 = time.time()
                if kind == "gen":
                    prod = a.runs / c["run_id"] / "product.png"   # ca có ảnh sản phẩm: cùng ảnh tham chiếu như lượt distill
                    refs = [np.asarray(Image.open(prod).convert("RGB"))] if prod.is_file() else None
                    img = F.generate(c["prompt_en"], c["W"], c["H"], c["seed"], refs=refs)
                    rec["product"] = refs is not None
                else:   # erase: nháp distill; chain: nháp base vừa vẽ
                    src = out / "gen" / f"{c['key']}.png" if kind == "chain" else draft_of(c)
                    draft = np.asarray(Image.open(src).convert("RGB"))
                    img, rec["register"] = register(draft, F.edit(draft, ERASE_PROMPT, c["seed"], neg=a.neg or None))
                rec["s"] = round(time.time() - t1, 1)
            except Exception as e:
                print(f"{kind} {c['key']}: LỖI {type(e).__name__}: {str(e)[:300]}")
                continue
            Image.fromarray(img).save(dst)
            info.setdefault(kind, {"runs": {}})["runs"][c["key"]] = rec
            save()
            print(f"[{k}/{n}] {kind} {c['key']}: {rec['s']}s")
    if a.shard:
        return 0
    save()
    info = multigpu.merge(info_f, deep=("runs",))
    kinds = ("gen", "erase", "chain")
    for kd in kinds:
        ss = [r["s"] for r in (info.get(kd) or {}).get("runs", {}).values() if "s" in r]
        if ss:
            print(f"{kd}: {len(ss)} ảnh, trung bình {sum(ss) / len(ss):.1f}s / ảnh")
    pngs = [p for kd in kinds for p in sorted((out / kd).glob("*.png"))]
    if not pngs:
        print("chưa có ảnh nào để gói")
        return 1
    files = [(info_f, "info.json"), (a.cases, a.cases.name)]
    for p in pngs:
        if a.png:
            files.append((p, f"{p.parent.name}/{p.name}"))
            continue
        j = out / "ship" / p.parent.name / f"{p.stem}.jpg"
        if not j.exists() or j.stat().st_mtime < p.stat().st_mtime:
            j.parent.mkdir(parents=True, exist_ok=True)
            Image.open(p).convert("RGB").save(j, quality=95, subsampling=0)
        files.append((j, f"{p.parent.name}/{j.name}"))
    print(f"xong: {len(pngs)} ảnh ở {out}")
    if not a.no_ship:
        ship.offer(ship.pack(files, out.parent / f"{out.name}_zip", out.name, int(a.zip_mb * 2**20) if a.zip_mb else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
