#!/usr/bin/env python3
"""MÁY CHỦ: thử Hi-SAM (phân đoạn NÉT CHỮ ở mức điểm ảnh) trên các nháp đã lưu, ĐO THỜI GIAN và VRAM trên mọi GPU.

Hi-SAM = SAM ViT-L (Meta) + phần tự gợi ý "cắt chữ" (github ymy-k/Hi-SAM). Vào: ảnh nháp. Ra: mặt nạ nét chữ. Chưa nối vào
dây chuyền TextFix -- script này chỉ trả lời: nó chạy được trên A30 không, mất bao lâu, tốn bao nhiêu VRAM, mặt nạ ra sao.

Tệp cần có (máy chủ không ra internet: tải ở máy khác rồi đưa lên, xem docs/SERVER.md):
  ~/persistent-data/hisam/Hi-SAM/                                   mã nguồn (giải nén zip của github)
  ~/persistent-data/hisam/Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth   nền SAM ViT-L, 1,2 GB
  ~/persistent-data/hisam/sam_tss_l_textseg.pth                     phần nét chữ (bản TextSeg), 117 MB
Thư viện: torch + einops (KHÔNG cài requirements.txt của Hi-SAM: nó ghim numpy / opencv bản cũ).

  python scripts/hisam_test.py --check                 # CHẠY TRƯỚC: kiểm tệp, thư viện, VRAM còn trống từng card; không nạp model
  python scripts/hisam_test.py accept2 --limit 4 --gpus 1
  python scripts/hisam_test.py accept2 mix1            # mọi nháp của hai tập, chia đều mọi GPU -> một zip < 24 MB
  python scripts/hisam_test.py accept2 --patch         # đo thêm chế độ cắt mảnh 512 px (chữ nhỏ), chậm hơn nhiều
  python scripts/hisam_test.py --dir <thư mục ảnh>     # ảnh bất kỳ thay cho nháp của tập
  python scripts/hisam_test.py accept2 --ship-only
Máy chủ TextFix đang chạy (mỗi A30 một bản FLUX ~17 GB) thì Hi-SAM vẫn có thể vừa phần VRAM còn lại; --check in số GB trống.
Hết bộ nhớ: tắt máy chủ (pkill -f server/app.py) rồi chạy lại.

Mỗi ảnh đo ba số (GPU đã đồng bộ trước khi bấm giờ; ảnh đầu của mỗi card chạy nháp một lượt, không tính):
  mã hoá  = SAM ViT-L nhìn ảnh (phần nặng)        giải mã = sinh mặt nạ nét chữ
  fp16    = cùng ảnh chạy nửa độ chính xác (autocast): thời gian + độ trùng (IoU) với mặt nạ fp32. Bỏ: --no-half
Ra output/hisam_test[_<tag>]/: masks/<key>.png (mặt nạ, cỡ nháp), cmp/<key>.jpg (nháp | nét chữ tô hồng), info.json (thời
gian từng ảnh, tổng kết từng card). Zip: output/runs_zip/hisam_test[_<tag>]_<giờ>_NN.zip.
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
SAM_FILE = {"vit_b": "sam_vit_b_01ec64.pth", "vit_l": "sam_vit_l_0b3195.pth", "vit_h": "sam_vit_h_4b8939.pth"}
PATCH, STRIDE = 512, 384   # cắt mảnh: đúng cấu hình của tác giả


def find_drafts(sets: list[str], runs: Path) -> list[dict]:
    """Nháp của các tập đã chạy. Máy chủ: output/dev/<tập>_manifest.json + output/runs/<run_id>/v0/draft.*; máy cá nhân (zip đã
    giải nén): output/<tập>/<tập>_manifest.json + output/<tập>/<run_id>/v0/draft.*."""
    out = []
    for s in sets:
        for mf, base in ((ROOT / "output" / "dev" / f"{s}_manifest.json", runs), (ROOT / "output" / s / f"{s}_manifest.json", ROOT / "output" / s)):
            if mf.is_file():
                break
        else:
            print(f"không thấy manifest của tập {s} (output/dev/{s}_manifest.json) -- tập này đã chạy trên máy này chưa?")
            continue
        for cid, m in sorted(json.loads(mf.read_text(encoding="utf-8")).items()):
            d = base / str(m.get("run_id")) / "v0"
            src = next((d / f"draft{e}" for e in (".png", ".jpg") if (d / f"draft{e}").is_file()), None)
            if src:
                out.append({"key": f"{s}_{cid}", "src": str(src)})
    return out


def check(a) -> int:
    """Kiểm tệp, thư viện, VRAM trống. Không nạp model."""
    ok = True
    for name, p in (("mã nguồn Hi-SAM", a.repo / "hi_sam" / "modeling" / "build.py"),
                    ("nền SAM", a.repo / "pretrained_checkpoint" / SAM_FILE[a.model_type]), ("phần nét chữ", a.ckpt)):
        good = p.is_file()
        ok &= good
        print(f"  {'có   ' if good else 'THIẾU'} {name}: {p}" + (f"  ({p.stat().st_size:,} byte)" if good else ""))
    for mod in ("torch", "einops"):
        try:
            __import__(mod)
            print(f"  có    thư viện {mod}")
        except ImportError:
            ok = False
            print(f"  THIẾU thư viện {mod}")
    try:
        import torch
        print(f"  torch {torch.__version__}, CUDA {'có' if torch.cuda.is_available() else 'KHÔNG'}, {torch.cuda.device_count()} card")
        for k in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(k)
            print(f"    card {k}: {torch.cuda.get_device_name(k)}, trống {free / 2**30:.1f} / {total / 2**30:.1f} GB")
    except Exception as ex:   # thiếu torch / trình điều khiển lỗi: đã báo ở trên
        print(f"  không hỏi được GPU: {type(ex).__name__}: {ex}")
    print("ĐỦ, chạy được." if ok else "CHƯA ĐỦ -- xem các dòng THIẾU.")
    return 0 if ok else 1


def load(a, device: str):
    """Nạp Hi-SAM. Mã của tác giả tìm nền SAM theo đường dẫn tương đối 'pretrained_checkpoint/...' -> tạm đổi thư mục làm việc."""
    sys.path.insert(0, str(a.repo))
    cwd = os.getcwd()
    os.chdir(a.repo)
    try:
        from hi_sam.modeling.build import model_registry
        from hi_sam.modeling.predictor import SamPredictor
        ns = argparse.Namespace(model_type=a.model_type, checkpoint=str(a.ckpt), device=device, hier_det=False,
                                input_size=[1024, 1024], patch_mode=False, attn_layers=1, prompt_len=12)
        model = model_registry[a.model_type](ns).eval().to(device)
    finally:
        os.chdir(cwd)
    return SamPredictor(model)


def main(argv: list[str] | None = None) -> int:
    from textfix.tools import multigpu, ship
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", help="tên tập đã chạy (accept2, mix1...)")
    ap.add_argument("--dir", type=Path, default=None, help="thư mục ảnh (png / jpg) thay cho nháp của tập")
    ap.add_argument("--runs", type=Path, default=ROOT / "output" / "runs", help="nơi máy chủ giữ các lượt chạy")
    ap.add_argument("--repo", type=Path, default=HOME / "Hi-SAM")
    ap.add_argument("--ckpt", type=Path, default=HOME / "sam_tss_l_textseg.pth")
    ap.add_argument("--model-type", default="vit_l", choices=sorted(SAM_FILE))
    ap.add_argument("--device", default="", help="mặc định: cuda nếu có, không thì cpu")
    ap.add_argument("--only", default="", help="key, cách nhau dấu phẩy (vd. accept2_h02)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--patch", action="store_true", help="đo thêm chế độ cắt mảnh 512 px")
    ap.add_argument("--no-half", action="store_true", help="không đo fp16")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ship-only", action="store_true")
    multigpu.add_args(ap)
    a = ap.parse_args(argv)
    if a.check:
        return check(a)
    name = "hisam_test" + (f"_{a.tag}" if a.tag else "")
    out = a.out or ROOT / "output" / name
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "cmp").mkdir(parents=True, exist_ok=True)
    if a.dir:
        cases = [{"key": p.stem, "src": str(p)} for p in sorted(a.dir.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    else:
        cases = find_drafts(a.sets, a.runs)
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    cases = [c for c in cases if not only or c["key"] in only]
    if a.limit:
        cases = cases[:a.limit]
    if not cases and not a.ship_only:
        print("không có ảnh nào để chạy -- truyền tên tập đã chạy (accept2, mix1...) hoặc --dir <thư mục ảnh>")
        return 1
    info_f = out / "info.json"
    todo = [] if a.ship_only else [c for c in cases if a.force or not (out / "masks" / f"{c['key']}.png").exists()]
    wall = None
    if todo and not a.shard:
        if info_f.exists() and a.force:
            info_f.unlink()
        if check(a):
            return 1
        t_wall = time.time()
        if multigpu.fanout(a, __file__, argv):
            wall = (time.time() - t_wall, len(todo))
            todo = []
    todo = multigpu.shard(todo, a)
    info = {} if a.shard else multigpu.merge(info_f)
    info_out = multigpu.part(info_f, a)
    if todo:
        import numpy as np
        import torch
        from PIL import Image
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        gpu = device.startswith("cuda")
        card = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] if gpu else "cpu"
        sync = torch.cuda.synchronize if gpu else (lambda: None)
        free0 = torch.cuda.mem_get_info()[0] / 2**30 if gpu else 0.0
        t0 = time.time()
        try:
            P = load(a, device)
        except torch.cuda.OutOfMemoryError:
            print(f"HẾT VRAM khi nạp (card {card} trống {free0:.1f} GB) -- tắt máy chủ TextFix (pkill -f server/app.py) rồi chạy lại")
            return 3
        sync()
        t_load = time.time() - t0
        vram_model = torch.cuda.memory_allocated() / 2**30 if gpu else 0.0
        print(f"nạp Hi-SAM {a.model_type} trên {device} (card {card}, trống {free0:.1f} GB): {t_load:.1f}s, model chiếm {vram_model:.2f} GB; "
              f"{len(todo)} ảnh")

        def run(img, half: bool = False):
            """-> (mặt nạ bool, giây mã hoá, giây giải mã)."""
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=half and gpu):
                sync()
                t1 = time.time()
                P.set_image(img)
                sync()
                t2 = time.time()
                _m, hr, _s, _hs = P.predict(multimask_output=False)
                sync()
                return np.asarray(hr[0]).astype(bool), t2 - t1, time.time() - t2

        def run_patch(img):
            """Cắt mảnh 512, bước 384, cộng điểm số rồi lấy ngưỡng (như demo của tác giả). -> (mặt nạ, giây, số mảnh)."""
            H, W = img.shape[:2]
            ys = sorted({min(y, max(0, H - PATCH)) for y in range(0, H, STRIDE)})
            xs = sorted({min(x, max(0, W - PATCH)) for x in range(0, W, STRIDE)})
            acc = np.zeros((H, W), np.float32)
            sync()
            t1 = time.time()
            with torch.no_grad():
                for y in ys:
                    for x in xs:
                        P.set_image(img[y:y + PATCH, x:x + PATCH])
                        _m, hr, _s, _hs = P.predict(multimask_output=False, return_logits=True)
                        acc[y:y + PATCH, x:x + PATCH] += np.asarray(hr[0], np.float32)
            sync()
            return acc > P.model.mask_threshold, time.time() - t1, len(ys) * len(xs)

        run(np.asarray(Image.open(todo[0]["src"]).convert("RGB")))   # chạy nháp: lượt đầu gánh khởi tạo CUDA / cuDNN, không tính
        if gpu:
            torch.cuda.reset_peak_memory_stats()
        rows = {}
        for k, c in enumerate(todo, 1):
            img = np.asarray(Image.open(c["src"]).convert("RGB"))
            H, W = img.shape[:2]
            try:
                mk, t_enc, t_dec = run(img)
                row = {"size": [W, H], "enc": round(t_enc, 3), "dec": round(t_dec, 3), "text_pct": round(100 * float(mk.mean()), 2),
                       "card": card}
                if not a.no_half and gpu:
                    mh, h_enc, h_dec = run(img, half=True)
                    u = float((mk | mh).sum())
                    row.update(half=round(h_enc + h_dec, 3), half_iou=round(float((mk & mh).sum()) / u, 4) if u else 1.0)
                if a.patch:
                    mp, t_p, n_p = run_patch(img)
                    row.update(patch=round(t_p, 3), patch_n=n_p, patch_pct=round(100 * float(mp.mean()), 2))
                    Image.fromarray((mp * 255).astype(np.uint8)).convert("1").save(out / "masks" / f"{c['key']}_patch.png")
            except torch.cuda.OutOfMemoryError:
                print(f"HẾT VRAM ở {c['key']} -- tắt máy chủ TextFix (pkill -f server/app.py) rồi chạy lại")
                return 3
            Image.fromarray((mk * 255).astype(np.uint8)).convert("1").save(out / "masks" / f"{c['key']}.png")
            vis = img.astype(np.float32)
            vis[mk] = vis[mk] * 0.25 + np.array([255, 0, 255], np.float32) * 0.75
            sheet = Image.fromarray(np.hstack([img, vis.round().astype(np.uint8)]))
            sheet.thumbnail((1500, 760))
            sheet.save(out / "cmp" / f"{c['key']}.jpg", quality=82)
            rows[c["key"]] = row
            print(f"[{k}/{len(todo)}] {c['key']} {W}x{H}: mã hoá {t_enc * 1000:.0f} ms + giải mã {t_dec * 1000:.0f} ms"
                  + (f" | fp16 {row['half'] * 1000:.0f} ms, IoU {row['half_iou']:.3f}" if "half" in row else "")
                  + (f" | cắt mảnh {row['patch']:.2f}s ({row['patch_n']} mảnh)" if "patch" in row else "")
                  + f" | chữ {row['text_pct']:.1f}% ảnh")
            info.setdefault("runs", {}).update(rows)
            info.setdefault("cards", {})[card] = {
                "device": torch.cuda.get_device_name(0) if gpu else "cpu", "load_s": round(t_load, 1), "free_before_gb": round(free0, 1),
                "model_gb": round(vram_model, 2), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if gpu else 0.0,
                "images": len(rows)}
            info["model"] = {"type": a.model_type, "ckpt": a.ckpt.name, "torch": torch.__version__}
            info_out.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.shard:
        return 0
    if wall:
        info["wall"] = {"seconds": round(wall[0], 1), "images": wall[1], "note": "cả lượt trên mọi card, gồm cả thời gian nạp model"}
        info_f.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    report(info)
    if a.no_ship:
        return 0
    files = [(p, f"{name}/{p.relative_to(out).as_posix()}") for p in sorted(out.rglob("*")) if p.is_file() and ".shard" not in p.name]
    if not files:
        print("chưa có kết quả để gói")
        return 1
    ship.offer(ship.pack(files, ROOT / "output" / "runs_zip", f"{name}_{time.strftime('%m%d_%H%M%S')}"))
    return 0


def report(info: dict) -> None:
    """Bảng tổng kết: từng card và chung."""
    rows = list((info.get("runs") or {}).values())
    if not rows:
        return
    med = lambda v: sorted(v)[len(v) // 2] if v else 0.0   # noqa: E731

    def line(tag: str, rs: list) -> str:
        full = [r["enc"] + r["dec"] for r in rs]
        s = f"  {tag}: {len(rs)} ảnh | fp32 trung vị {med(full) * 1000:.0f} ms (mã hoá {med([r['enc'] for r in rs]) * 1000:.0f} + giải mã " \
            f"{med([r['dec'] for r in rs]) * 1000:.0f}), chậm nhất {max(full) * 1000:.0f} ms"
        h = [r for r in rs if "half" in r]
        if h:
            s += f" | fp16 trung vị {med([r['half'] for r in h]) * 1000:.0f} ms, IoU với fp32 thấp nhất {min(r['half_iou'] for r in h):.3f}"
        p = [r for r in rs if "patch" in r]
        if p:
            s += f" | cắt mảnh trung vị {med([r['patch'] for r in p]):.2f}s"
        return s
    print("---- TỔNG KẾT Hi-SAM ----")
    for card, c in sorted((info.get("cards") or {}).items()):
        print(f"  card {card} ({c['device']}): nạp {c['load_s']}s, trống trước khi nạp {c['free_before_gb']} GB, model {c['model_gb']} GB, "
              f"đỉnh VRAM {c['peak_gb']} GB")
        print(line(f"card {card}", [r for r in rows if r.get("card") == card]))
    print(line("chung", rows))
    w = info.get("wall")
    if w:
        print(f"  cả lượt (mọi card song song, GỒM nạp model): {w['seconds']}s cho {w['images']} ảnh = {w['seconds'] / max(1, w['images']):.2f}s / ảnh")


if __name__ == "__main__":
    sys.exit(main())
