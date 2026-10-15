#!/usr/bin/env python3
"""MÁY CÁ NHÂN: đo bước ĐO DÒNG trên bộ đáp án `bench/lines_gt/` (không cần GPU, không gọi VLM).

  python scripts/lines_bench.py --extract              # giải nén nháp + mặt nạ Hi-SAM + plan.json từ các zip báo cáo ở gốc repo
  python scripts/lines_bench.py                        # dòng đã lưu trong plan.json (lượt chạy đó) = mốc
  python scripts/lines_bench.py --source current       # vùng dòng đã lưu + hướng / kiểu đường theo mã hiện tại (reorient)
  python scripts/lines_bench.py --source build         # toàn bộ build hiện tại (kết quả bộ dò OCR đệm ở output/lines_data)

Ghép dòng dự đoán <-> dòng đáp án theo NÉT: tập điểm nét Hi-SAM trong vùng mỗi dòng, IoU >= MATCH (khung lệch vài px không ảnh
hưởng). Chỉ số (dòng `eval` của đáp án, loại poster + object):
  - phủ (recall), đúng (precision), F1 theo dòng;
  - TÁCH: một dòng đáp án bị chia cho >= 2 dòng dự đoán (mỗi phần >= PART nét của nó);
  - GỘP: một dòng dự đoán ôm >= 2 dòng đáp án (mỗi dòng >= PART nét của dòng đó);
  - KIỂU: dòng ghép được mà kiểu đường (ngang / nghiêng / cong / dọc) khác đáp án.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GT_DIR = ROOT / "bench" / "lines_gt"
DATA = ROOT / "output" / "lines_data"
MATCH = 0.5   # IoU nét để coi là cùng một dòng
PART = 0.2    # phần nét tối thiểu để tính là "ôm" / "bị chia"


def extract() -> None:
    """Các zip `<tập>_report_*.zip` ở gốc repo -> output/lines_data/<tập>/runs/<run>/v<i>/{draft.jpg, ink.png, plan.json}."""
    want = {json.loads(p.read_text(encoding="utf-8"))["src"] for p in GT_DIR.glob("*.json")}
    got = set()
    for z in sorted(ROOT.glob("*_report_*.zip")):
        name = z.name.split("_report_")[0]
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                parts = m.split("/")
                if len(parts) >= 4 and parts[-1] in ("draft.jpg", "ink.png", "plan.json"):
                    src = f"{name}/{'/'.join(parts[-4:-1])}"
                    if src in want:
                        out = DATA / src / parts[-1]
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(zf.read(m))
                        got.add(src)
    print(f"giải nén {len(got)} / {len(want)} nháp vào {DATA}")


def shape_of(l: dict) -> str:
    from textfix.lines.detect import shape_of
    return shape_of(l)


def region(l: dict, H: int, W: int) -> np.ndarray:
    """Vùng một dòng dự đoán: theo đa giác / đường chân (erase.line_region)."""
    from textfix.erase import line_region
    return line_region(l, H, W, 0.1)


def box_mask(b, H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
    return m


def own(masks: list[np.ndarray], cc: np.ndarray, n: int) -> list[np.ndarray]:
    """Mỗi mảnh nét thuộc về MỘT vùng: vùng chứa nhiều điểm của mảnh nhất (>= nửa mảnh); vùng chồng nhau không chia đôi nét."""
    area = np.bincount(cc.ravel(), minlength=n).astype(float)
    hit = np.stack([np.bincount(cc[m], minlength=n) for m in masks]) if masks else np.zeros((0, n))
    best = hit.argmax(0) if len(masks) else np.zeros(n, int)
    ok = (hit.max(0) >= 0.5 * area) if len(masks) else np.zeros(n, bool)
    ok[0] = False
    return [np.isin(cc, np.nonzero(ok & (best == i))[0]) for i in range(len(masks))]


def score(gt: list[dict], pred: list[dict], ink: np.ndarray, kinds: tuple = ("poster", "object")) -> Counter:
    import cv2
    H, W = ink.shape
    n, cc = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    gs = own([box_mask(g["box"], H, W) for g in gt], cc, n)
    G = [(g, s) for g, s in zip(gt, gs) if g.get("eval", True) and g["kind"] in kinds and s.sum() >= 20]
    # nét thuộc dòng đáp án "rác" / không tính điểm: dự đoán nằm chủ yếu ở đó không tính đúng / sai
    skip = np.zeros(ink.shape, bool)
    for g, s in zip(gt, gs):
        if not g.get("eval", True) or g["kind"] not in kinds:
            skip |= s
    ps = own([region(p, H, W) for p in pred], cc, n)
    P = [(p, s) for p, s in zip(pred, ps) if s.sum() >= 20 and (s & skip).sum() < 0.5 * s.sum()]
    c = Counter(gt=len(G), pred=len(P))
    inter = np.array([[float((gs & ps).sum()) for _, ps in P] for _, gs in G]) if G and P else np.zeros((len(G), len(P)))
    gsum = np.array([float(s.sum()) for _, s in G])
    psum = np.array([float(s.sum()) for _, s in P])
    iou = inter / np.maximum(1.0, gsum[:, None] + psum[None, :] - inter) if inter.size else inter
    used_p = set()
    for gi in np.argsort(-(iou.max(1) if iou.size else np.zeros(len(G)))):
        if not iou.size:
            break
        pj = int(np.argmax(np.where([j in used_p for j in range(len(P))], -1, iou[gi])))
        if iou[gi, pj] >= MATCH and pj not in used_p:
            used_p.add(pj)
            c["match"] += 1
            if shape_of(P[pj][0]) != G[gi][0]["shape"]:
                c["shape_wrong"] += 1
                c[f"shape {G[gi][0]['shape']}->{shape_of(P[pj][0])}"] += 1
    if inter.size:
        part = inter >= PART * gsum[:, None]
        c["split"] = int((part.sum(1) >= 2).sum())
        c["merge"] = int((part.sum(0) >= 2).sum())
        c.details = ([("tách", G[i][0].get("id"), [P[j][0].get("id") for j in np.nonzero(part[i])[0]]) for i in np.nonzero(part.sum(1) >= 2)[0]]
                     + [("gộp", P[j][0].get("id"), [G[i][0].get("id") for i in np.nonzero(part[:, j])[0]]) for j in np.nonzero(part.sum(0) >= 2)[0]])
    return c


def predict(src_dir: Path, source: str, ink: np.ndarray) -> list[dict]:
    from PIL import Image
    from textfix.lines.page import build
    from textfix.lines.detect import reorient
    if source == "build":
        with ocr_cache(src_dir):
            return build(np.asarray(Image.open(src_dir / "draft.jpg").convert("RGB")), ink)["L"]
    L = json.loads((src_dir / "plan.json").read_text(encoding="utf-8"))["slots"]["L"]
    if source == "current":
        reorient(np.asarray(Image.open(src_dir / "draft.jpg").convert("RGB")), ink, L)
    return L


@contextmanager
def ocr_cache(src_dir: Path):
    """Đệm ra đĩa hai lời gọi chậm của bộ dò (dòng OCR, bản đồ vùng chữ) cho một nháp: đo lại nhiều lần không chạy lại OCR."""
    from textfix.lines import ocr
    read0, reg0 = ocr._read, ocr.regions
    fl, fr = src_dir / "ocr_lines.json", src_dir / "ocr_regions.npz"

    def read(img, multi):
        if not fl.exists():
            fl.write_text(json.dumps(read0(img, multi), ensure_ascii=False), encoding="utf-8")
        return json.loads(fl.read_text(encoding="utf-8"))

    def regions(img):
        if not fr.exists():
            R = reg0(img)
            np.savez_compressed(fr, shape=np.array(img.shape[:2]), pads=np.array([r["pad"] for r in R]),
                                areas=np.array([r["area"] for r in R]), masks=np.array([r["mask"] for r in R], bool).reshape(len(R), *img.shape[:2]))
            return R
        z = np.load(fr)
        if tuple(z["shape"]) != img.shape[:2]:   # ảnh khác (thu nhỏ): không đệm
            return reg0(img)
        return [{"mask": m, "pad": float(p), "area": int(a)} for m, p, a in zip(z["masks"], z["pads"], z["areas"])]
    ocr._read, ocr.regions = read, regions
    try:
        yield
    finally:
        ocr._read, ocr.regions = read0, reg0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--source", choices=["saved", "current", "build"], default="saved")
    ap.add_argument("--only", default="", help="các khoá, cách nhau dấu phẩy")
    ap.add_argument("--kind", choices=["all", "poster"], default="all", help="poster: chỉ chấm chữ poster (chữ được viết lại)")
    ap.add_argument("-v", action="store_true", help="in từng nháp")
    a = ap.parse_args()
    if a.extract:
        extract()
        return
    from PIL import Image
    tot = Counter()
    for f in sorted(GT_DIR.glob("*.json")):
        gt = json.loads(f.read_text(encoding="utf-8"))
        if a.only and gt["key"] not in a.only.split(","):
            continue
        d = DATA / gt["src"]
        if not (d / "ink.png").exists():
            print("thiếu dữ liệu (chạy --extract):", gt["src"])
            continue
        ink = np.asarray(Image.open(d / "ink.png").convert("L")) > 127
        c = score(gt["lines"], predict(d, a.source, ink), ink, ("poster",) if a.kind == "poster" else ("poster", "object"))
        tot += c
        if a.v:
            print(f"{gt['key']}: đáp án {c['gt']}, dự đoán {c['pred']}, khớp {c['match']}, tách {c['split']}, gộp {c['merge']}, "
                  f"sai kiểu {c['shape_wrong']} {getattr(c, 'details', '')}")
    rec = tot["match"] / max(1, tot["gt"])
    prec = tot["match"] / max(1, tot["pred"])
    print(f"[{a.source}] dòng đáp án {tot['gt']}, dự đoán {tot['pred']} | phủ {rec:.3f}, đúng {prec:.3f}, "
          f"F1 {2 * rec * prec / max(1e-9, rec + prec):.3f} | tách {tot['split']}, gộp {tot['merge']}, sai kiểu {tot['shape_wrong']}")
    for k, v in sorted(tot.items()):
        if k.startswith("shape ") and "->" in k:
            print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
