#!/usr/bin/env python3
"""MÁY CHỦ: thử Hi-SAM PHÂN CẤP (dòng / từ / đoạn) trên các nháp đã lưu -- để quyết có chia dòng bằng Hi-SAM thay OCR không.

Người dùng 14/10: lỗi còn lại của chặng B dồn về bước ĐO DÒNG của OCR (gộp hai cột, một khung ôm hai hàng, đọc tick thành chữ
cái); không vá luật theo kiểu sai của OCR mà kết hợp Hi-SAM. Script này chưa nối vào dây chuyền, chỉ trả lời: Hi-SAM phân cấp
chia dòng trên chữ giun / chữ trang trí ra sao so với OCR, mất bao lâu, tốn bao nhiêu VRAM.

Tệp cần có (máy chủ không ra internet: tải ở máy khác rồi đưa lên):
  ~/persistent-data/hisam/Hi-SAM/                                         mã nguồn (đã có)
  ~/persistent-data/hisam/Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth   nền SAM ViT-L (đã có)
  ~/persistent-data/hisam/hi_sam_l.pth                                    Hi-SAM-L bản HierText (bảng "Hi-SAM-L" trong README
                                                                          của github ymy-k/Hi-SAM, ~ vài trăm MB) -- MỚI

  python scripts/hisam_hier_test.py --check                # kiểm tệp + VRAM, không nạp model
  python scripts/hisam_hier_test.py mix1 accept2           # mọi nháp của hai tập, chia mọi GPU -> một zip < 24 MB
  python scripts/hisam_hier_test.py mix1 --limit 4 --gpus 1

Chế độ tự động của tác giả (AutoMaskGenerator): rải điểm trên vùng chữ, mỗi điểm cho mặt nạ từ / dòng / đoạn, lọc trùng ->
MỖI DÒNG một mặt nạ (gồm các từ của nó) + ma trận "cùng đoạn". Ra output/hisam_hier[_<tag>]/:
  lines/<key>.json   từng dòng: khung, hình chữ nhật xoay (tâm, cỡ, góc), khung từng từ, số đoạn; + số đo so với dòng OCR
  cmp/<key>.jpg      nháp | dòng Hi-SAM (mỗi dòng một màu, khung từ mảnh) + khung dòng OCR (trắng, nét đứt) của plan.json
  info.json          thời gian / VRAM từng ảnh, tổng kết
So với OCR (dòng L của plan.json cùng lượt): OCR GỘP = một dòng OCR phủ >= 2 dòng Hi-SAM (hai hàng / hai cột); OCR CẮT = một dòng
Hi-SAM nằm trong >= 2 dòng OCR; chỉ Hi-SAM thấy / chỉ OCR thấy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

HOME = Path(os.path.expanduser(os.environ.get("TEXTFIX_HISAM", "~/persistent-data/hisam")))
COLS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (210, 245, 60), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128)]


def find_runs(sets: list[str], runs: Path) -> list[dict]:
    """Nháp + plan.json của các tập đã chạy (máy chủ: output/dev/<tập>_manifest.json; máy cá nhân: output/<tập>/<tập>_manifest.json)."""
    out = []
    for s in sets:
        for mf, base in ((ROOT / "output" / "dev" / f"{s}_manifest.json", runs), (ROOT / "output" / s / f"{s}_manifest.json", ROOT / "output" / s)):
            if mf.is_file():
                break
        else:
            print(f"không thấy manifest của tập {s}")
            continue
        for cid, m in sorted(json.loads(mf.read_text(encoding="utf-8")).items()):
            d = base / str(m.get("run_id")) / "v0"
            src = next((d / f"draft{e}" for e in (".png", ".jpg") if (d / f"draft{e}").is_file()), None)
            if src:
                out.append({"key": f"{s}_{cid}", "src": str(src), "plan": str(d / "plan.json") if (d / "plan.json").is_file() else ""})
    return out


def check(a) -> int:
    ok = True
    for name, p in (("mã nguồn Hi-SAM", a.repo / "hi_sam" / "modeling" / "auto_mask_generator.py"),
                    ("nền SAM", a.repo / "pretrained_checkpoint" / "sam_vit_l_0b3195.pth"), ("Hi-SAM-L phân cấp", a.ckpt)):
        good = p.is_file()
        ok &= good
        print(f"  {'có   ' if good else 'THIẾU'} {name}: {p}" + (f"  ({p.stat().st_size:,} byte)" if good else ""))
    for mod in ("torch", "einops", "cv2"):
        try:
            __import__(mod)
        except ImportError:
            ok = False
            print(f"  THIẾU thư viện {mod}")
    try:
        import torch
        for k in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(k)
            print(f"    card {k}: {torch.cuda.get_device_name(k)}, trống {free / 2**30:.1f} / {total / 2**30:.1f} GB")
    except Exception as ex:
        print(f"  không hỏi được GPU: {type(ex).__name__}: {ex}")
    print("ĐỦ, chạy được." if ok else "CHƯA ĐỦ -- xem các dòng THIẾU.")
    return 0 if ok else 1


def load(a, device: str):
    """Nạp Hi-SAM-L phân cấp (hier_det). Mã của tác giả đọc nền SAM theo đường dẫn tương đối -> tạm đổi thư mục làm việc."""
    sys.path.insert(0, str(a.repo))
    cwd = os.getcwd()
    os.chdir(a.repo)
    try:
        from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
        from hi_sam.modeling.build import model_registry
        ns = argparse.Namespace(model_type="vit_l", checkpoint=str(a.ckpt), device=device, hier_det=True, input_size=[1024, 1024],
                                patch_mode=False, attn_layers=1, prompt_len=12)
        model = model_registry["vit_l"](ns).eval().to(device)
    finally:
        os.chdir(cwd)
    return AutoMaskGenerator(model, efficient_hisam=False)


def lines_of(masks, affinity, thresh: float):
    """Mặt nạ dòng (n, 1, H, W) + ma trận cùng đoạn -> [{box, rect, words, para, px}] theo thứ tự trên -> dưới."""
    import cv2
    import numpy as np
    out = []
    if masks is None:
        return out
    m = masks[:, 0].astype(np.uint8)
    n = len(m)
    par = list(range(n))

    def root(i):
        while par[i] != i:
            i = par[i]
        return i
    for i, j in zip(*np.where(affinity > thresh)):
        if i != j:
            par[root(i)] = root(j)
    for i in range(n):
        ys, xs = np.nonzero(m[i])
        if len(xs) < 20:
            continue
        cnts, _ = cv2.findContours(m[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        words = [list(map(int, cv2.boundingRect(c))) for c in cnts if cv2.contourArea(c) >= 16]
        (cx, cy), (w, h), ang = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        if w < h:   # góc theo cạnh dài, trong [-45, 45]
            w, h, ang = h, w, ang - 90
        out.append({"box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], "rect": [round(cx, 1), round(cy, 1),
                    round(w, 1), round(h, 1), round(float(ang), 1)], "words": [[x, y, x + bw, y + bh] for x, y, bw, bh in words],
                    "para": root(i), "px": int(len(xs)), "_mask": m[i]})
    out.sort(key=lambda r: (r["box"][1], r["box"][0]))
    return out


def compare(H, ocr: list[dict]) -> dict:
    """Dòng Hi-SAM vs dòng OCR: phủ = phần điểm của dòng Hi-SAM nằm trong khung OCR."""
    import numpy as np
    inside = lambda r, b: float(r["_mask"][int(max(0, b[1])):int(b[3]), int(max(0, b[0])):int(b[2])].sum()) / max(1, r["px"])  # noqa: E731
    cov = np.array([[inside(r, l["box"]) for l in ocr] for r in H]) if H and ocr else np.zeros((len(H), len(ocr)))
    merged = [l["id"] for j, l in enumerate(ocr) if (cov[:, j] >= 0.6).sum() >= 2] if len(H) else []
    split = [i for i in range(len(H)) if (cov[i] >= 0.2).sum() >= 2 and cov[i].max() < 0.8] if len(ocr) else []
    only_h = [i for i in range(len(H)) if not len(ocr) or cov[i].sum() < 0.2]
    only_o = [l["id"] for j, l in enumerate(ocr) if not len(H) or cov[:, j].max() < 0.2]
    return {"hisam_lines": len(H), "ocr_lines": len(ocr), "ocr_merged": merged, "ocr_split_hisam": split, "only_hisam": only_h,
            "only_ocr": only_o}


def main(argv: list[str] | None = None) -> int:
    from textfix import multigpu, ship
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", help="tên tập đã chạy (mix1, accept2...)")
    ap.add_argument("--runs", type=Path, default=ROOT / "output" / "runs")
    ap.add_argument("--repo", type=Path, default=HOME / "Hi-SAM")
    ap.add_argument("--ckpt", type=Path, default=HOME / "hi_sam_l.pth")
    ap.add_argument("--device", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--points", type=int, default=1500, help="số điểm rải trên vùng chữ (tác giả: 1500)")
    ap.add_argument("--layout", type=float, default=0.5, help="ngưỡng cùng đoạn (tác giả: 0.5)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ship-only", action="store_true")
    multigpu.add_args(ap)
    a = ap.parse_args(argv)
    if a.check:
        return check(a)
    name = "hisam_hier" + (f"_{a.tag}" if a.tag else "")
    out = ROOT / "output" / name
    for sub in ("lines", "cmp"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cases = find_runs(a.sets, a.runs)
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    cases = [c for c in cases if not only or c["key"] in only][:a.limit or None]
    if not cases and not a.ship_only:
        print("không có nháp nào -- truyền tên tập đã chạy (mix1, accept2...)")
        return 1
    info_f = out / "info.json"
    todo = [] if a.ship_only else [c for c in cases if a.force or not (out / "lines" / f"{c['key']}.json").exists()]
    if todo and not a.shard:
        if check(a):
            return 1
        if multigpu.fanout(a, __file__, argv):
            todo = []
    todo = multigpu.shard(todo, a)
    info = {} if a.shard else multigpu.merge(info_f)
    info_out = multigpu.part(info_f, a)
    if todo:
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        torch.manual_seed(42)
        np.random.seed(42)
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        gpu = device.startswith("cuda")
        card = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] if gpu else "cpu"
        sync = torch.cuda.synchronize if gpu else (lambda: None)
        free0 = torch.cuda.mem_get_info()[0] / 2**30 if gpu else 0.0
        t0 = time.time()
        try:
            amg = load(a, device)
        except torch.cuda.OutOfMemoryError:
            print(f"HẾT VRAM khi nạp (card {card} trống {free0:.1f} GB) -- tắt máy chủ TextFix (pkill -f server/app.py) rồi chạy lại")
            return 3
        sync()
        t_load = time.time() - t0
        print(f"nạp Hi-SAM-L phân cấp trên {device} (card {card}, trống {free0:.1f} GB): {t_load:.1f}s; {len(todo)} ảnh")
        if gpu:
            torch.cuda.reset_peak_memory_stats()
        rows = {}
        for k, c in enumerate(todo, 1):
            img = np.asarray(Image.open(c["src"]).convert("RGB"))
            Hh, Ww = img.shape[:2]
            try:
                sync()
                t1 = time.time()
                with torch.no_grad():
                    amg.set_image(img)
                    sync()
                    t2 = time.time()
                    masks, scores, aff = amg.predict(from_low_res=False, fg_points_num=a.points, batch_points_num=100,
                                                     score_thresh=0.5, nms_thresh=0.5)
                sync()
                t3 = time.time()
            except torch.cuda.OutOfMemoryError:
                print(f"HẾT VRAM ở {c['key']} -- tắt máy chủ TextFix rồi chạy lại")
                return 3
            H = lines_of(masks, aff, a.layout)
            ocr = json.loads(Path(c["plan"]).read_text(encoding="utf-8")).get("slots", {}).get("L", []) if c["plan"] else []
            ocr = [{"id": l["id"], "box": l.get("ocr_box") or l["box"], "ocr": l.get("ocr")} for l in ocr]
            cmp = compare(H, ocr)
            row = {"size": [Ww, Hh], "enc": round(t2 - t1, 3), "amg": round(t3 - t2, 3), "card": card, **cmp}
            (out / "lines" / f"{c['key']}.json").write_text(json.dumps(
                {"lines": [{k2: v for k2, v in r.items() if k2 != "_mask"} for r in H], "ocr": ocr, "compare": cmp}, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
            vis = img.astype(np.float32).copy()
            for i, r in enumerate(H):
                col = np.array(COLS[r["para"] % len(COLS)], np.float32)
                mk = r["_mask"].astype(bool)
                vis[mk] = vis[mk] * 0.3 + col * 0.7
            im = Image.fromarray(vis.round().astype(np.uint8))
            dr = ImageDraw.Draw(im)
            for i, r in enumerate(H):
                dr.rectangle(r["box"], outline=COLS[r["para"] % len(COLS)], width=2)
                for w in r["words"]:
                    dr.rectangle(w, outline=(255, 255, 0), width=1)
                dr.text((r["box"][0], max(0, r["box"][1] - 12)), f"h{i}", fill=(255, 255, 255))
            for l in ocr:
                b = l["box"]
                for x in range(int(b[0]), int(b[2]), 8):
                    dr.line([(x, b[1]), (min(x + 4, b[2]), b[1])], fill=(255, 255, 255), width=2)
                    dr.line([(x, b[3]), (min(x + 4, b[2]), b[3])], fill=(255, 255, 255), width=2)
                dr.text((b[0], b[3] + 2), l["id"], fill=(255, 255, 255))
            sheet = Image.fromarray(np.hstack([img, np.asarray(im)]))
            sheet.thumbnail((1600, 900))
            sheet.save(out / "cmp" / f"{c['key']}.jpg", quality=80)
            rows[c["key"]] = row
            print(f"[{k}/{len(todo)}] {c['key']}: mã hoá {row['enc'] * 1000:.0f} ms + phân cấp {row['amg'] * 1000:.0f} ms | dòng Hi-SAM "
                  f"{cmp['hisam_lines']} / OCR {cmp['ocr_lines']} | OCR gộp {cmp['ocr_merged'] or '-'} | chỉ Hi-SAM {len(cmp['only_hisam'])} | "
                  f"chỉ OCR {cmp['only_ocr'] or '-'}")
            info.setdefault("runs", {}).update(rows)
            info.setdefault("cards", {})[card] = {"load_s": round(t_load, 1), "free_before_gb": round(free0, 1),
                                                  "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if gpu else 0.0, "images": len(rows)}
            info_out.write_text(json.dumps(info, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    if a.shard:
        return 0
    runs = list((info.get("runs") or {}).values())
    if runs:
        med = lambda v: sorted(v)[len(v) // 2]   # noqa: E731
        print("---- TỔNG KẾT Hi-SAM phân cấp ----")
        print(f"  {len(runs)} ảnh | trung vị: mã hoá {med([r['enc'] for r in runs]) * 1000:.0f} ms, phân cấp {med([r['amg'] for r in runs]) * 1000:.0f} ms")
        print(f"  dòng Hi-SAM {sum(r['hisam_lines'] for r in runs)} / dòng OCR {sum(r['ocr_lines'] for r in runs)} | OCR gộp >= 2 dòng Hi-SAM: "
              f"{sum(len(r['ocr_merged']) for r in runs)} | chỉ Hi-SAM thấy: {sum(len(r['only_hisam']) for r in runs)} | chỉ OCR thấy: "
              f"{sum(len(r['only_ocr']) for r in runs)}")
        for card, cinfo in sorted((info.get("cards") or {}).items()):
            print(f"  card {card}: nạp {cinfo['load_s']}s, trống trước {cinfo['free_before_gb']} GB, đỉnh VRAM {cinfo['peak_gb']} GB")
    if a.no_ship:
        return 0
    files = [(p, f"{name}/{p.relative_to(out).as_posix()}") for p in sorted(out.rglob("*")) if p.is_file() and ".shard" not in p.name]
    if not files:
        print("chưa có kết quả để gói")
        return 1
    ship.offer(ship.pack(files, ROOT / "output" / "runs_zip", f"{name}_{time.strftime('%m%d_%H%M%S')}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
