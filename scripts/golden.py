#!/usr/bin/env python3
"""DẤU VÂN TAY trạm thiết kế, để tái cấu trúc mã mà không đổi hành vi: chạy engine.design (đo dòng -> gỡ chữ -> ghép -> bố cục
-> dựng -> sửa) trên các nháp đã lưu, VLM giả trả lại ĐÚNG câu trả lời đã lưu trong plan.json (lượt hỏi lại trả rỗng), rồi băm
plan.json (bỏ thời gian), poster, nền, mặt nạ.

  python -X utf8 scripts/golden.py record build/golden.json final2_1015:final2     # trước khi sửa
  python -X utf8 scripts/golden.py check  build/golden.json final2_1015:final2     # sau khi sửa: phải "khớp hết"
  ... --only g03,g18

Cần: output/<lượt>/<run>/v0/{draft.png, ink.png, plan.json}, brief.json; Chromium; LaMa.
NHẸ MÁY: LaMa chạy MỘT lần mỗi (ảnh, mặt nạ), kết quả đệm ở build/golden_cache/ (mặt nạ vẫn được băm và so, nên sửa nhầm bước
gỡ chữ vẫn lộ); torch 2 luồng CPU. Mặc định chỉ chạy bộ ca nhỏ GOLDEN (--only all: mọi ca).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from textfix import erase

from textfix.draw import hisam  # noqa: E402
from textfix.render.browser import Browser  # noqa: E402
from textfix.engine import design  # noqa: E402


torch.set_num_threads(2)
CACHE = "build/golden_cache"
# 12 ca đại diện (thẳng / nghiêng / cong / dọc / bảng / ảnh sản phẩm / hàng nhiều cỡ / dòng bỏ sót)
GOLDEN = "g01,g03,g05,g08,g12,g14,g18,g21,g26,g30,d07,d21"


def cached_inpaint(real):
    def inpaint(img, mask, device=None):
        os.makedirs(CACHE, exist_ok=True)
        f = f"{CACHE}/{digest(img, np.asarray(mask, bool))}.png"
        if os.path.exists(f):
            return np.asarray(Image.open(f).convert("RGB"))
        out = real(img, mask, device)
        Image.fromarray(out).save(f)
        return out
    return inpaint


def raw_answer(P: dict) -> dict:
    """plan.json (đơn vị đã đọc) -> câu trả lời VLM thô như lúc gọi."""
    return {"assign": [{"lines": u["marks"], "client": [f"T{k}" for k in u["clients"]], "kind": u["kind"], "text": u["text_vlm"],
                        "role": u["role_vlm"], "split": u["split"], "why": u["why"]} for u in P.get("assign") or []],
            "styles": P.get("style") or {}, "accent": P.get("accent") or [], "notes": P.get("notes")}


def fake_vlm(P: dict):
    calls = []

    def call(system, parts, meta=None, fresh=False):
        calls.append(1)
        return raw_answer(P) if len(calls) == 1 else {"assign": []}
    return call


def digest(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()[:16]


def cases(specs: list[str], only: set):
    for spec in specs:
        setn, mfn = spec.split(":") if ":" in spec else (spec, spec)
        man = json.load(open(f"output/{setn}/{mfn}_manifest.json", encoding="utf-8"))
        bench = {c["id"]: c for c in json.load(open(f"bench/{mfn}.json", encoding="utf-8"))} if os.path.exists(f"bench/{mfn}.json") else {}
        for cid, m in sorted(man.items()):
            r = f"output/{setn}/{m.get('run_id')}/"
            if not m.get("run_id") or (only and cid not in only) or not os.path.exists(r + "v0/plan.json"):
                continue
            P = json.load(open(r + "v0/plan.json", encoding="utf-8"))
            if not P.get("assign") or not os.path.exists(r + "v0/ink.png"):
                continue
            yield f"{setn}/{cid}", r, P, ((bench.get(cid) or {}).get("request") or {}).get("image")


def run(specs: list[str], only: set) -> dict:
    from textfix.lines.ocr import read_lines
    out = {}
    with Browser() as B:
        for key, r, P, image in cases(specs, only):
            draft = np.asarray(Image.open(next(iter(glob.glob(r + "v0/draft.*")))).convert("RGB"))
            ink = hisam.from_png(r + "v0/ink.png", (draft.shape[1], draft.shape[0]))
            Bf = json.load(open(r + "brief.json", encoding="utf-8"))
            Bf["prompt_en"] = P.get("prompt_en") or Bf["prompt_en"]
            d = {"draft": draft, "ink": ink, "timing": {}, "device": "cpu"}
            if image and os.path.exists(image):
                d["product_ocr"] = [l["text"] for l in read_lines(np.asarray(Image.open(image).convert("RGB")))
                                    if sum(ch.isalnum() for ch in l["text"]) >= 2]
            res = design(d, Bf, fake_vlm(P), B.page, product=bool(image))
            plan = json.dumps(res["plan"], ensure_ascii=False, sort_keys=True, default=str)
            out[key] = {"plan": hashlib.sha256(plan.encode()).hexdigest()[:16], "poster": digest(res["poster"]),
                        "clean": digest(res["clean"]), "mask": digest(res["mask"]), "base": digest(res["base"]),
                        "ops": [(o.get("id"), o.get("text"), round(float(o.get("size") or 0), 2)) for o in res["plan"]["ops"]]}
            print(key, out[key]["plan"], out[key]["poster"], flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "check"])
    ap.add_argument("file")
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    erase.inpaint = cached_inpaint(erase.inpaint)
    only = set() if a.only == "all" else {x for x in (a.only or GOLDEN).split(",") if x}
    got = run(a.sets, only)
    if a.mode == "record":
        os.makedirs(os.path.dirname(a.file) or ".", exist_ok=True)
        json.dump(got, open(a.file, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"ghi {len(got)} ca -> {a.file}")
        return 0
    ref = json.load(open(a.file, encoding="utf-8"))
    diff = {k: [f for f in ("plan", "clean", "mask", "base", "poster") if ref.get(k, {}).get(f) != v[f]] for k, v in got.items()}
    diff = {k: v for k, v in diff.items() if v}
    for k, v in diff.items():
        print(f"KHÁC {k}: {v}")
        if "plan" in v:
            print("   ops cũ:", ref[k]["ops"][:6], "\n   ops mới:", got[k]["ops"][:6])
    lost = sorted(set(ref) - set(got))
    print(f"{len(got)} ca: {'khớp hết' if not diff and not lost else f'{len(diff)} khác, {len(lost)} thiếu {lost}'}")
    return 0 if not diff and not lost else 1


if __name__ == "__main__":
    sys.exit(main())
