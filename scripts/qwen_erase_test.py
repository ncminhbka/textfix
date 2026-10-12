#!/usr/bin/env python3
"""MÁY CHỦ: thử khả năng XOÁ CHỮ (sửa ảnh) của Qwen-Image-2.1-Turbo -- cùng lời dặn xoá của FLUX ("Remove all text from this
image. Keep everything else exactly the same."), trên:
  (a) 28 nháp FLUX distill mà bước xoá klein làm hỏng (bench/base_test.json mục erase: sót chữ, đổi hình, mất vỏ, đối chứng);
  (b) --own: các nháp do chính Qwen vẽ (output/qwen_test/*.png, sau khi chạy scripts/qwen_test.py).

  !pkill -f server/app.py                         # tắt máy chủ TextFix trước
  %run scripts/qwen_erase_test.py --sig           # CHẠY TRƯỚC: in tham số của pipeline (tên tham số nhận ảnh), không nạp model
  %run scripts/qwen_erase_test.py --limit 2 --gpus 1
  %run scripts/qwen_erase_test.py                 # 28 ca, mọi GPU -> zip < 24 MB, tự tải về
  %run scripts/qwen_erase_test.py --own           # xoá nháp của chính Qwen
  %run scripts/qwen_erase_test.py --why sót --prompt "Remove all text, letters and numbers. Keep pills, cards, icons."
  %run scripts/qwen_erase_test.py --ship-only
Ra output/qwen_erase[_<tag>]/<key>.png (bản xoá, cỡ nháp) + cmp/<key>.jpg (nháp | Qwen xoá | FLUX xoá nếu còn trong
output/runs) + info.json (thời gian, VRAM, tỉ lệ điểm ảnh đổi so với nháp).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

IMAGE_ARGS = ("image", "images", "input_image", "input_images", "reference_image", "reference_images", "image_reference")


def main(argv: list[str] | None = None) -> int:
    import qwen_test as Q
    from textfix import config, multigpu, ship
    from textfix.flux import ERASE_PROMPT
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="?", type=Path, default=ROOT / "bench" / "base_test.json")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--sig", action="store_true", help="chỉ in tham số __call__ của pipeline, không nạp model")
    ap.add_argument("--own", action="store_true", help="xoá nháp của chính Qwen (--own-dir) thay cho nháp FLUX")
    ap.add_argument("--own-dir", type=Path, default=ROOT / "output" / "qwen_test")
    ap.add_argument("--only", default="", help="key, cách nhau dấu phẩy")
    ap.add_argument("--why", default="", help="chỉ các ca có nhóm lỗi chứa chuỗi này (sót, đổi hình, vỏ, đối chứng)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", default=ERASE_PROMPT, help="lời dặn xoá")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--cfg", type=float, default=-1.0)
    ap.add_argument("--image-arg", default="", help="tên tham số nhận ảnh của pipeline (mặc định: tự dò)")
    ap.add_argument("--mem", choices=("offload", "seq", "cuda"), default="offload")
    ap.add_argument("--tag", default="")
    ap.add_argument("--runs", type=Path, default=ROOT / "output" / "runs", help="nơi máy chủ giữ các lượt chạy (nháp + bản xoá FLUX)")
    ap.add_argument("--drafts", type=Path, default=ROOT / "output" / "base_test_drafts", help="nháp distill dự phòng: <key>.png")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ship-only", action="store_true")
    multigpu.add_args(ap)
    a = ap.parse_args(argv)
    config.load_env()
    model = a.model or Path(os.environ.get("TEXTFIX_QWEN_DIR") or Q.DEFAULT_MODEL)
    if not (model / "model_index.json").is_file():
        print(f"không thấy {model}/model_index.json -- truyền --model <thư mục> hoặc export TEXTFIX_QWEN_DIR")
        return 1
    if a.sig:
        import diffusers
        cls = getattr(diffusers, json.loads((model / "model_index.json").read_text(encoding="utf-8"))["_class_name"])
        for n, p in inspect.signature(cls.__call__).parameters.items():
            print(f"  {n}: {p.annotation if p.annotation is not p.empty else ''} = {p.default if p.default is not p.empty else ''}")
        print("---- README (mã mẫu) ----")
        print(Q.readme_code(model))
        return 0
    name = "qwen_erase" + ("_own" if a.own else "") + (f"_{a.tag}" if a.tag else "")
    out = a.out or ROOT / "output" / name
    (out / "cmp").mkdir(parents=True, exist_ok=True)
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    if a.own:
        cases = [{"key": p.stem, "why": "nháp Qwen", "seed": 4001, "src": str(p)} for p in sorted(a.own_dir.glob("*.png"))]
    else:
        cases = []
        for c in json.loads(a.cases.read_text(encoding="utf-8"))["erase"]:
            rd = a.runs / c["run_id"] / f"v{c['v']}"
            src = next((p for p in (rd / "draft.png", a.drafts / f"{c['key']}.png") if p.is_file()), None)
            cases.append({**c, "src": str(src) if src else None, "flux": str(rd / "plate_raw.jpg") if (rd / "plate_raw.jpg").is_file()
                          else str(rd / "plate.png") if (rd / "plate.png").is_file() else None})
        lost = [c["key"] for c in cases if not c["src"]]
        if lost and not a.shard:
            print(f"THIẾU nháp của {len(lost)} ca (không có trong {a.runs} lẫn {a.drafts}), bỏ qua: {lost}")
        cases = [c for c in cases if c["src"]]
    cases = [c for c in cases if (not only or c["key"] in only) and a.why in (c.get("why") or "")]
    if a.limit:
        cases = cases[:a.limit]
    info_f = out / "info.json"
    todo = [] if a.ship_only else [c for c in cases if a.force or not (out / f"{c['key']}.png").exists()]
    if todo and multigpu.fanout(a, __file__, argv):
        todo = []
    todo = multigpu.shard(todo, a)
    info = {} if a.shard else multigpu.merge(info_f, deep=("runs",))
    info_out = multigpu.part(info_f, a)
    if todo:
        import numpy as np
        import torch
        from PIL import Image
        code = Q.readme_code(model)
        t0 = time.time()
        mem = a.mem
        pipe = Q.load(model, mem)
        params = set(inspect.signature(pipe.__call__).parameters)
        iarg = a.image_arg or next((x for x in IMAGE_ARGS if x in params), "")
        if iarg not in params:
            print(f"pipeline {type(pipe).__name__} không có tham số nhận ảnh trong {IMAGE_ARGS}; tham số có: {sorted(params)}\n"
                  "-> chạy --sig xem tên đúng rồi truyền --image-arg <tên>")
            return 2
        cfg = a.cfg if a.cfg >= 0 else None
        print(f"nạp {type(pipe).__name__} ({mem}) {time.time() - t0:.0f}s; {len(todo)} ca xoá; ảnh vào qua '{iarg}'; lời dặn: {a.prompt!r}")
        info["model"] = {"dir": model.name, "pipeline": type(pipe).__name__, "image_arg": iarg, "steps": a.steps, "cfg": cfg,
                         "mem": mem, "prompt": a.prompt, "readme": code[:4000]}
    for k, c in enumerate(todo, 1):
        draft = Image.open(c["src"]).convert("RGB")
        W, H = draft.size
        kw = {"prompt": a.prompt, iarg: [draft] if iarg.endswith("s") else draft, "width": W // 16 * 16, "height": H // 16 * 16,
              "generator": torch.Generator("cpu").manual_seed(int(c.get("seed") or 0))}
        if a.steps:
            kw["num_inference_steps"] = a.steps
        if cfg is not None:
            kw["true_cfg_scale" if "true_cfg_scale" in params else "guidance_scale"] = cfg
            if cfg > 1:
                kw["negative_prompt"] = " "
        kw = {x: v for x, v in kw.items() if x in params}
        img = None
        for _ in range(2):
            try:
                torch.cuda.reset_peak_memory_stats()
                t1 = time.time()
                img = pipe(**kw).images[0]
                break
            except torch.cuda.OutOfMemoryError:
                if mem == "seq":
                    print(f"{c['key']}: hết VRAM cả ở chế độ seq")
                    break
                print(f"{c['key']}: hết VRAM ở chế độ {mem} -> nạp lại ở chế độ seq (chậm hơn)")
                del pipe
                torch.cuda.empty_cache()
                mem = "seq"
                pipe = Q.load(model, mem)
                info["model"]["mem"] = mem
            except Exception as e:
                print(f"{c['key']}: LỖI {type(e).__name__}: {str(e)[:400]}")
                break
        if img is None:
            continue
        s = round(time.time() - t1, 1)
        raw = img.size
        img = img.convert("RGB").resize((W, H), Image.LANCZOS) if img.size != (W, H) else img.convert("RGB")
        img.save(out / f"{c['key']}.png")
        # tỉ lệ điểm ảnh đổi rõ so với nháp (> 20 mức xám trung bình ba kênh): xoá càng "chỉ chữ" thì càng nhỏ
        d = np.abs(np.asarray(img, np.int16) - np.asarray(draft, np.int16)).mean(2)
        rec = {"why": c.get("why"), "s": s, "vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), "out_size": list(raw),
               "changed": round(float((d > 20).mean()), 4), "mean_diff": round(float(d.mean()), 2)}
        tiles = [draft, img]
        if c.get("flux"):
            tiles.append(Image.open(c["flux"]).convert("RGB").resize((W, H), Image.LANCZOS))
        z = min(1.0, 900 / H)
        tw, th = int(W * z), int(H * z)
        sheet = Image.new("RGB", (tw * len(tiles) + 8 * (len(tiles) - 1), th), (255, 255, 255))
        for i, t in enumerate(tiles):
            sheet.paste(t.resize((tw, th), Image.LANCZOS), (i * (tw + 8), 0))
        sheet.save(out / "cmp" / f"{c['key']}.jpg", quality=90)
        info.setdefault("erase", {"runs": {}})["runs"][c["key"]] = rec
        info_out.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{k}/{len(todo)}] {c['key']} ({c.get('why')}): {s}s, VRAM {rec['vram_gb']} GB, đổi {rec['changed']:.1%} điểm ảnh")
    if a.shard:
        return 0
    info_out.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    multigpu.merge(info_f, deep=("runs",))
    cmps = sorted((out / "cmp").glob("*.jpg"))
    if not cmps:
        print("chưa có ảnh nào để gói")
        return 1
    print(f"xong: {len(cmps)} ca ở {out} (cmp/<key>.jpg: nháp | Qwen xoá | FLUX xoá)")
    if not a.no_ship:
        ship.offer(ship.pack([(info_f, "info.json")] + [(p, p.name) for p in cmps], out.parent / f"{name}_zip", name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
