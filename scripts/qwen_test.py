#!/usr/bin/env python3
"""MÁY CHỦ: thử Qwen-Image-2.1-Turbo (đã tải sẵn, thư mục diffusers) vẽ NHÁP poster chữ Việt -- so với nháp FLUX distill đã có.
Vẽ lại các nháp trong bench/base_test.json (mục gen: đúng prompt + cỡ + seed của nháp distill) bằng Qwen.

  !pkill -f server/app.py                    # tắt máy chủ TextFix trước (nó giữ FLUX trên cả hai card)
  %run scripts/qwen_test.py --info           # CHẠY TRƯỚC: in model_index.json, cỡ từng phần, đoạn mã mẫu trong README (không nạp model)
  %run scripts/qwen_test.py --limit 2 --gpus 1   # thử 2 nháp trên một card xem có vừa VRAM không
  %run scripts/qwen_test.py                  # 24 nháp, mọi GPU -> một zip < 24 MB (JPG 92), tự tải về
  %run scripts/qwen_test.py --only dev3_p04_v0,dev2_d20_v0 --steps 8 --cfg 1.0
  %run scripts/qwen_test.py --prompts prompt_test.txt   # mỗi dòng một prompt tự do (1024x1024)
  %run scripts/qwen_test.py --ship-only
Model: --model, hoặc TEXTFIX_QWEN_DIR, mặc định ~/persistent-data/Qwen-Image-2.1-Turbo.
Số bước / CFG: --steps / --cfg; không cho thì lấy trong đoạn mã mẫu của README, không có thì để mặc định của pipeline.
VRAM (--mem): offload (mặc định: từng phần lên card khi cần; hết VRAM thì tự lùi về seq) | seq (từng lớp, chậm, luôn vừa) |
cuda (cả pipeline trên card).
Ra output/qwen_test/<key>.png + info.json (prompt, bước, CFG, thời gian, VRAM đỉnh, README mẫu).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

DEFAULT_MODEL = Path.home() / "persistent-data" / "Qwen-Image-2.1-Turbo"


def readme_code(model: Path) -> str:
    """Các khối mã python trong README của model (cách gọi chính chủ: lớp pipeline, số bước, CFG)."""
    f = model / "README.md"
    if not f.is_file():
        return ""
    blocks = re.findall(r"```(?:python|py)[ \t]*\r?\n(.*?)```", f.read_text(encoding="utf-8", errors="replace"), flags=re.S)
    return "\n# ----\n".join(b.strip() for b in blocks if "import" in b or "pipe" in b)


def readme_default(code: str, name: str) -> float | None:
    m = re.search(rf"\b{name}\s*[=:]\s*([0-9.]+)", code)
    return float(m.group(1)) if m else None


def show_info(model: Path) -> None:
    print(f"model: {model}")
    mi = model / "model_index.json"
    print(mi.read_text(encoding="utf-8") if mi.is_file() else "KHÔNG có model_index.json")
    for d in sorted(p for p in model.iterdir() if p.is_dir()):
        gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 2 ** 30
        print(f"  {d.name:14s} {gb:6.1f} GB  {sorted(f.name for f in d.iterdir())[:8]}")
    try:
        import diffusers
        import torch
        import transformers
        print(f"diffusers {diffusers.__version__}, transformers {transformers.__version__}, torch {torch.__version__}")
        cls = json.loads(mi.read_text(encoding="utf-8")).get("_class_name")
        print(f"lớp pipeline {cls}: {'CÓ' if hasattr(diffusers, cls) else 'KHÔNG CÓ'} trong diffusers đang cài")
        for k in range(torch.cuda.device_count()):
            free, tot = torch.cuda.mem_get_info(k)
            print(f"  cuda:{k} {torch.cuda.get_device_name(k)} trống {free / 2 ** 30:.1f} / {tot / 2 ** 30:.1f} GB")
    except Exception as e:
        print(f"không đọc được môi trường: {type(e).__name__}: {e}")
    print("---- README (mã mẫu) ----")
    print(readme_code(model) or "(README không có khối mã)")


def load(model: Path, mem: str):
    import torch
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(str(model), torch_dtype=torch.bfloat16)
    if mem == "cuda":
        pipe.to("cuda")
    elif mem == "seq":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()
    for part in ("vae",):
        m = getattr(pipe, part, None)
        for fn in ("enable_tiling", "enable_slicing"):
            if m is not None and hasattr(m, fn):
                getattr(m, fn)()
    return pipe


def main(argv: list[str] | None = None) -> int:
    from textfix import config, multigpu, ship
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="?", type=Path, default=ROOT / "bench" / "base_test.json")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--info", action="store_true", help="chỉ in thông tin model, không nạp")
    ap.add_argument("--only", default="", help="key, cách nhau dấu phẩy")
    ap.add_argument("--limit", type=int, default=0, help="chỉ vẽ N nháp đầu")
    ap.add_argument("--prompts", type=Path, default=None, help="tệp chữ, mỗi dòng một prompt tự do (thay cho bench)")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--cfg", type=float, default=-1.0)
    ap.add_argument("--neg", default=" ", help="negative prompt (chỉ dùng khi CFG > 1)")
    ap.add_argument("--mem", choices=("offload", "seq", "cuda"), default="offload")
    ap.add_argument("--tag", default="", help="hậu tố thư mục ra (so nhiều cấu hình)")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ship-only", action="store_true")
    multigpu.add_args(ap)
    a = ap.parse_args(argv)
    config.load_env()
    model = a.model or Path(os.environ.get("TEXTFIX_QWEN_DIR") or DEFAULT_MODEL)
    if not (model / "model_index.json").is_file():
        print(f"không thấy {model}/model_index.json -- truyền --model <thư mục> hoặc export TEXTFIX_QWEN_DIR")
        return 1
    if a.info:
        show_info(model)
        return 0
    name = "qwen_test" + (f"_{a.tag}" if a.tag else "")
    out = a.out or ROOT / "output" / name
    out.mkdir(parents=True, exist_ok=True)
    if a.prompts:
        lines = [x.strip() for x in a.prompts.read_text(encoding="utf-8").splitlines() if x.strip()]
        cases = [{"key": f"p{i:02d}", "prompt_en": p, "W": 1024, "H": 1024, "seed": 4001} for i, p in enumerate(lines, 1)]
    else:
        cases = json.loads(a.cases.read_text(encoding="utf-8"))["gen"]
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    cases = [c for c in cases if not only or c["key"] in only]
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
        code = readme_code(model)
        steps = a.steps or int(readme_default(code, "num_inference_steps") or 0)
        cfg = a.cfg if a.cfg >= 0 else readme_default(code, "true_cfg_scale")
        t0 = time.time()
        mem = a.mem
        pipe = load(model, mem)
        params = set(inspect.signature(pipe.__call__).parameters)
        print(f"nạp {type(pipe).__name__} ({mem}) {time.time() - t0:.0f}s; {len(todo)} nháp; bước {steps or 'mặc định'}, "
              f"CFG {cfg if cfg is not None else 'mặc định'}")
        info["model"] = {"dir": model.name, "pipeline": type(pipe).__name__, "steps": steps, "cfg": cfg, "mem": mem, "readme": code[:4000]}
    for k, c in enumerate(todo, 1):
        kw = {"prompt": c["prompt_en"], "width": c["W"] // 16 * 16, "height": c["H"] // 16 * 16,
              "generator": torch.Generator("cpu").manual_seed(int(c["seed"]))}
        if steps:
            kw["num_inference_steps"] = steps
        if cfg is not None:
            kw["true_cfg_scale" if "true_cfg_scale" in params else "guidance_scale"] = cfg
            if cfg > 1:
                kw["negative_prompt"] = a.neg
        kw = {x: v for x, v in kw.items() if x in params}
        img = None
        for attempt in range(2):
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
                pipe = load(model, mem)
                info["model"]["mem"] = mem
            except Exception as e:
                print(f"{c['key']}: LỖI {type(e).__name__}: {str(e)[:400]}")
                break
        if img is None:
            continue
        Image.fromarray(np.asarray(img.convert("RGB"))).save(out / f"{c['key']}.png")
        info.setdefault("gen", {"runs": {}})["runs"][c["key"]] = {
            "seed": c["seed"], "W": kw.get("width"), "H": kw.get("height"), "s": round(time.time() - t1, 1),
            "vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), "prompt": c["prompt_en"]}
        info_out.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{k}/{len(todo)}] {c['key']}: {time.time() - t1:.1f}s, VRAM đỉnh {info['gen']['runs'][c['key']]['vram_gb']} GB")
    if a.shard:
        return 0
    info_out.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    multigpu.merge(info_f, deep=("runs",))
    pngs = sorted(out.glob("*.png"))
    if not pngs:
        print("chưa có ảnh nào để gói")
        return 1
    from PIL import Image
    files = [(info_f, "info.json")]
    for p in pngs:
        j = out / "ship" / f"{p.stem}.jpg"
        if not j.exists() or j.stat().st_mtime < p.stat().st_mtime:
            j.parent.mkdir(parents=True, exist_ok=True)
            Image.open(p).convert("RGB").save(j, quality=92)
        files.append((j, j.name))
    print(f"xong: {len(pngs)} nháp Qwen ở {out}")
    if not a.no_ship:
        ship.offer(ship.pack(files, out.parent / f"{name}_zip", name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
