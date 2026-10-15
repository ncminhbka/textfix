"""GỠ CHỮ GIUN khỏi nháp: mặt nạ từ nét chữ Hi-SAM, nền dưới chữ do LaMa vẽ lại -- MỘT lượt LaMa cho cả poster.

  ink = hisam.strokes(draft)                        # nét chữ (trạm vẽ)
  clean, mask = erase(draft, ink, M)                # gỡ MỌI dòng của M["L"] -> nền VLM nhìn
  base, mask = settle(draft, clean, ink, M, plan)   # dòng VLM chọn "keep" trở về điểm ảnh nháp (không gọi LaMa lần nữa)

Chất lượng gỡ chữ do mặt nạ quyết: LaMa sạch khi mặt nạ ôm sát nét, hỏng khi mặt nạ trùm cả nút / icon. Mỗi dòng cùng một cách:
  1. NÉT    điểm nét Hi-SAM trong vùng dòng, cộng trọn các mảng nét chạm lõi dòng mà phần lớn nằm trong vùng (đuôi chữ, dấu thanh
            thò khỏi khung OCR). Hi-SAM đã trả lời "có phải chữ không", nên vòng con dấu, viền nút, hoạ tiết không bị kéo vào.
  2. DỰ PHÒNG  Hi-SAM thấy quá ít ở dòng (chữ rất nhỏ): thêm các mảng cùng màu mực cỡ một chữ cái; vẫn không có thì lấy lõi dòng.
  3. QUẦNG  Hi-SAM chỉ thấy thân nét. Trong dải sát nét, điểm khác nền của dòng là viền dày / bóng đổ / quầng sáng; nền phẳng thì
            lan tiếp theo đuôi mờ. Quầng phải dính vào nét (icon đứng cách chữ một khe nền thì không), không lấn sang dòng khác.
  4. DẤU CÂU  mảng đậm nhỏ nằm gọn trong lõi dòng, giữa nét đầu và nét cuối (gạch nối, dấu chấm Hi-SAM bỏ qua).
Nét của dòng giữ (chữ in trên sản phẩm) không bao giờ vào mặt nạ. Còn sót quầng mờ / nét nhỏ thì chữ dựng mới đè lên.
Số đo và các hướng đã thử, đã bỏ: docs/TUNING.md (mục gỡ chữ), công cụ đo scripts/erase_bench.py.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np

from .color import hex_lab, lab

# --- vùng của dòng (theo cao dòng)
GROW_HOLD = 0.3     # vùng "thuộc dòng" (giữ / đếm chồng lấn)
GROW_SEARCH = 0.6   # vùng tìm nét / quầng
GROW_CORE = 0.1     # lõi dòng
HEIGHT_CAP = 2.2    # cỡ dòng <= 2,2 lần cao mảng nét cao (khung OCR trùm hai hàng thì "cỡ" phình, dải quầng nuốt icon)
# --- 1. nét
WHOLE = 0.5         # mảng nét chạm lõi dòng thuộc trọn về dòng khi >= nửa diện tích nằm trong vùng dòng
MASK_GROW = float(os.environ.get("TEXTFIX_MASK_GROW", "0.12"))   # nới quanh nét (viền khử răng cưa còn lại thì LaMa vẽ lại hình chữ)
# --- 2. dự phòng
WEAK = 0.08         # Hi-SAM phủ < 8% lõi dòng = thấy quá ít (dòng thường 21-30%)
INK_SAME = 25.0     # cùng màu mực: <= 25 ΔE
INK_IN = 0.7        # mảng cùng màu mực là nét khi >= 70% nằm trong vùng dòng (vòng con dấu, viền nút phần lớn ở ngoài)
GLYPH_MAX = 2.0     # ... và cạnh dài <= 2 cao dòng (một chữ cái)
# --- 3. quầng
HALO_REACH = 0.35   # dải sát nét rộng 0,35 cao dòng
HALO_DE = 14.0      # điểm trong dải khác nền của dòng > 14 ΔE là quầng (nền ước từ vành ngay ngoài dải, trong vùng dòng: chữ
                    # trong nút thì vành là lòng nút, lòng nút không thành quầng)
HALO_LO = 7.0       # nền phẳng: quầng lan tiếp qua điểm 7-14 ΔE liền với nó (đuôi bóng mờ dần); điểm đậm hơn ngoài dải là hình khác
HALO_FLAT = 8.0     # nền phẳng = vành lệch trung vị <= 8 ΔE; nền có vân / ảnh chụp thì không lan
HALO_LINE = 0.25    # dòng "có quầng" (nhật ký) khi quầng > 25% phần nét đã nới

LAMA_PATH = os.path.expanduser(os.environ.get("TEXTFIX_LAMA", "~/persistent-data/lama/big-lama.pt"))


# ---------------------------------------------------------------- API

def erase(draft: np.ndarray, ink: np.ndarray, M: dict, device: str | None = None,
          log: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Gỡ mọi dòng của M["L"]: LaMa lấp mặt nạ, ngoài mặt nạ giữ nguyên điểm ảnh nháp. -> (nền sạch chữ, mặt nạ).
    device: card chạy LaMa (nên là card đã vẽ nháp, để chia tải). Thiếu big-lama.pt: báo lỗi và dừng."""
    if not lama_ready():
        raise RuntimeError(f"thiếu tệp LaMa {LAMA_PATH} (docs/SERVER.md, mục LaMa; đổi chỗ: export TEXTFIX_LAMA=<đường dẫn>)")
    mask = erase_mask(draft, ink, M, set(), log)
    if not mask.any():
        return draft.copy(), mask
    return _blend(inpaint(draft, mask, device), draft, mask), mask


def settle(draft: np.ndarray, clean: np.ndarray, ink: np.ndarray, M: dict, P: dict | None = None,
           log: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Nền dựng của một kế hoạch P: nền sạch chữ trong mặt nạ của các dòng KHÔNG giữ, nháp gốc ở mọi chỗ khác. Chỗ hai mặt nạ
    chồng nhau thuộc dòng bị gỡ. Không gọi LaMa. -> (nền, mặt nạ)."""
    kept = kept_of(P)
    mask = erase_mask(draft, ink, M, kept, log)
    return (clean if not kept else _blend(clean, draft, mask)), mask


def kept_of(P: dict | None) -> set:
    """Các dòng lệnh "keep" của kế hoạch (giữ nguyên điểm ảnh nháp)."""
    return {m for op in (P or {}).get("ops") or [] if op.get("kind") == "keep" for m in op.get("marks") or op.get("slots") or []}


def line_region(l: dict, H: int, W: int, grow: float) -> np.ndarray:
    """Vùng một dòng (bool cỡ ảnh): đa giác OCR, hoặc dải dọc đường chân với dòng cong; nới `grow` cao dòng, dọc nhiều hơn ngang
    (dấu thanh tiếng Việt nhô khỏi khung OCR)."""
    import cv2
    m = np.zeros((H, W), np.uint8)
    if l.get("path"):
        sz = float(l.get("size") or 1.0)
        for (ax, ay), (bx, by) in zip(l["path"][:-1], l["path"][1:]):
            ln = float(np.hypot(bx - ax, by - ay)) or 1.0
            nx, ny = (by - ay) / ln, -(bx - ax) / ln
            cv2.fillConvexPoly(m, np.int32([[ax - nx * 0.3 * sz, ay - ny * 0.3 * sz], [bx - nx * 0.3 * sz, by - ny * 0.3 * sz],
                                            [bx + nx * 1.2 * sz, by + ny * 1.2 * sz], [ax + nx * 1.2 * sz, ay + ny * 1.2 * sz]]), 1)
        h = sz
    else:
        b = l["box"]
        poly = l.get("poly") or [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        cv2.fillPoly(m, [np.asarray(poly, np.int32)], 1)
        h = float(min(b[3] - b[1], b[2] - b[0])) if abs(l.get("angle") or 0) < 2 else float(l.get("size") or (b[3] - b[1]))
    k = max(1, int(round(grow * h)))
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * int(1.6 * k) + 1))).astype(bool)


# ---------------------------------------------------------------- mặt nạ

@dataclass
class _Page:
    """Những gì mọi dòng dùng chung: ảnh Lab, nét Hi-SAM và các mảng nét, vùng từng dòng, vùng giữ, số dòng phủ mỗi điểm."""
    H: int
    W: int
    lab: np.ndarray
    ink: np.ndarray
    n_cc: int
    cc: np.ndarray
    stats: np.ndarray
    regions: dict
    count: np.ndarray
    hold: np.ndarray


@dataclass
class _Window:
    """Một dòng, cắt theo khung của vùng tìm: s = lát cắt; r2 vùng tìm, r vùng dòng, r0 lõi; hgt cao dòng."""
    s: tuple
    r2: np.ndarray
    r: np.ndarray
    r0: np.ndarray
    hgt: float


def erase_mask(draft: np.ndarray, ink: np.ndarray, M: dict, kept: set, log: list | None = None) -> np.ndarray:
    """Mặt nạ gỡ chữ của mọi dòng không giữ (bool cỡ ảnh) -- các bước ở đầu tệp."""
    pg = _page(draft, ink, M["L"], kept)
    mask = np.zeros((pg.H, pg.W), bool)
    sources, n_halo = {"nét": 0, "màu mực": 0, "lõi dòng": 0}, 0
    for l in M["L"]:
        if l["id"] in kept:
            continue
        w = _window(pg, l)
        if w is None:
            continue
        got, src = _strokes(pg, l, w)
        got &= ~pg.hold[w.s]
        grown, halo = _halo(pg, w, got)
        mask[w.s] |= grown | halo
        sources[src] += 1
        n_halo += bool(halo.sum() > HALO_LINE * max(1.0, float(grown.sum())))
    mask &= ~(pg.ink & pg.hold)
    if log is not None:
        free = pg.ink & ~mask & ~pg.hold
        log.append("nền = nháp trừ chữ (Hi-SAM + LaMa): mặt nạ %.1f%% ảnh; dòng theo %s; %d dòng có quầng; %d dòng giữ nguyên; nét "
                   "không thuộc dòng nào %.2f%% ảnh" % (100 * mask.mean(), ", ".join(f"{k} {v}" for k, v in sources.items() if v),
                                                       n_halo, len(kept), 100 * free.mean()))
    return mask


def _page(draft: np.ndarray, ink: np.ndarray, lines: list[dict], kept: set) -> _Page:
    import cv2
    H, W = draft.shape[:2]
    ink = np.asarray(ink, bool)
    n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    regions = {l["id"]: line_region(l, H, W, GROW_HOLD) for l in lines}
    count, hold = np.zeros((H, W), np.uint8), np.zeros((H, W), bool)
    for l in lines:
        count += regions[l["id"]]
        if l["id"] in kept:
            hold |= regions[l["id"]]
    return _Page(H, W, lab(draft), ink, n_cc, cc, stats, regions, count, hold)


def _window(pg: _Page, l: dict) -> _Window | None:
    """Cắt theo vùng tìm của dòng; cao dòng chặn trên bởi HEIGHT_CAP lần cao các mảng nét cao (phân vị 90: thân chữ hoa, không
    phải trung vị -- dấu thanh rời kéo trung vị xuống)."""
    import cv2
    R2 = line_region(l, pg.H, pg.W, GROW_SEARCH)
    ys, xs = np.nonzero(R2)
    if not len(ys):
        return None
    s = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
    r = pg.regions[l["id"]][s]
    hgt = float(l.get("size") or (l["box"][3] - l["box"][1]))
    tall = [i for i in np.unique(pg.cc[s][pg.ink[s] & r]) if i > 0 and pg.stats[i, cv2.CC_STAT_HEIGHT] >= 4]
    if len(tall) >= 3:
        hgt = min(hgt, HEIGHT_CAP * float(np.percentile(pg.stats[tall, cv2.CC_STAT_HEIGHT], 90)))
    return _Window(s, R2[s], r, line_region(l, pg.H, pg.W, GROW_CORE)[s], hgt)


def _strokes(pg: _Page, l: dict, w: _Window) -> tuple[np.ndarray, str]:
    """Bước 1 + 2: nét của dòng -> (mặt nạ trong cửa sổ, nguồn: "nét" / "màu mực" / "lõi dòng")."""
    import cv2
    ik, c_ = pg.ink[w.s], pg.cc[w.s]
    area = pg.stats[:, cv2.CC_STAT_AREA]
    got = ik & w.r
    inside = np.bincount(c_[w.r], minlength=pg.n_cc)
    whole = [i for i in np.unique(c_[ik & w.r0]) if i > 0 and inside[i] >= WHOLE * area[i]]
    if whole:
        got |= np.isin(c_, whole) & w.r2
    core = max(1.0, float(w.r0.sum()))
    if float((got & w.r0).sum()) >= WEAK * core:
        return got, "nét"
    if l.get("color"):
        cand = w.r2 & (np.linalg.norm(pg.lab[w.s] - hex_lab(l["color"]), axis=2) <= INK_SAME)
        k, c2, s2, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
        touch, ins = np.bincount(c2[w.r0], minlength=k), np.bincount(c2[w.r], minlength=k)
        ok = [i for i in range(1, k) if touch[i] and ins[i] >= INK_IN * s2[i, cv2.CC_STAT_AREA]
              and max(s2[i, cv2.CC_STAT_WIDTH], s2[i, cv2.CC_STAT_HEIGHT]) <= GLYPH_MAX * w.hgt]
        if ok:
            got |= np.isin(c2, ok)
            if float((got & w.r0).sum()) >= WEAK * core:
                return got, "màu mực"
    return got | w.r0, "lõi dòng"


def _disk(radius: int) -> np.ndarray:
    import cv2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(m: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.dilate(m.astype(np.uint8), kernel).astype(bool)


def _halo(pg: _Page, w: _Window, got: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bước 3 + 4: (nét đã nới, quầng + dấu câu) trong cửa sổ."""
    import cv2
    grown = _dilate(got, _disk(max(3, int(round(MASK_GROW * w.hgt))))) & w.r2
    near = _dilate(got, _disk(max(4, int(round(HALO_REACH * w.hgt))))) & w.r2
    ring = (w.r2 & ~near & ~pg.ink[w.s]).astype(np.float32)
    if float(ring.sum()) <= 50:
        return grown, np.zeros(got.shape, bool)
    L = pg.lab[w.s]
    # nền của dòng: trung bình có trọng số của vành ngay ngoài dải sát nét, làm mượt theo cao dòng
    sg = max(6.0, 0.6 * w.hgt)
    den = cv2.GaussianBlur(ring, (0, 0), sg)
    bg = cv2.GaussianBlur(L * ring[..., None], (0, 0), sg) / np.maximum(den[..., None], 1e-3)
    d = np.where(den > 0.02, np.linalg.norm(L - bg, axis=2), 0.0)
    halo = near & ~got & (d > HALO_DE)
    halo = cv2.morphologyEx(halo.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    halo &= _touching(halo | grown, grown)   # quầng dính vào nét; icon cách chữ một khe nền thì không
    if halo.any() and float(np.median(d[ring > 0])) <= HALO_FLAT:
        tail = ((d > HALO_LO) & (d <= HALO_DE) & w.r2 & ~got) | halo   # nền phẳng: lan theo đuôi mờ (chỉ điểm nhạt)
        halo = _touching(tail, halo)
    halo |= _punctuation(w, d, got, grown)
    others = ((pg.count[w.s] - w.r.astype(np.uint8)) > 0) & ~w.r0   # vùng của dòng khác: dòng kia tự lo phần của nó
    halo = _dilate(halo & ~others, np.ones((5, 5), np.uint8)) & w.r2 & ~others & ~pg.hold[w.s]
    return grown, halo


def _touching(m: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Các mảng liên thông (8 hướng) của m có chạm seed."""
    import cv2
    _, c = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    ids = np.unique(c[seed])
    return np.isin(c, ids[ids > 0])


def _punctuation(w: _Window, d: np.ndarray, got: np.ndarray, grown: np.ndarray) -> np.ndarray:
    """Mảng đậm nhỏ (<= cao dòng) nằm gọn trong lõi dòng (không chạm mép lõi), giữa nét đầu và nét cuối của dòng: dấu câu / gạch
    nối Hi-SAM không coi là nét. Thứ đứng trước chữ là icon đầu dòng: để nguyên."""
    import cv2
    n, c, s, _ = cv2.connectedComponentsWithStats(((d > HALO_DE) & w.r0 & ~grown).astype(np.uint8), 8)
    rim = w.r0 & (cv2.erode(w.r0.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
    edge = np.bincount(c[rim], minlength=n)
    gx = np.nonzero(got.any(axis=0))[0]
    lo, hi = (int(gx.min()), int(gx.max())) if len(gx) else (0, -1)
    small = [i for i in range(1, n) if not edge[i] and max(s[i, cv2.CC_STAT_WIDTH], s[i, cv2.CC_STAT_HEIGHT]) <= w.hgt
             and s[i, cv2.CC_STAT_LEFT] >= lo and s[i, cv2.CC_STAT_LEFT] + s[i, cv2.CC_STAT_WIDTH] - 1 <= hi]
    return np.isin(c, small) if small else np.zeros(got.shape, bool)


# ---------------------------------------------------------------- LaMa

_LAMA: dict = {}
_LAMA_LOCK = threading.Lock()


def lama_ready() -> bool:
    return os.path.exists(LAMA_PATH)


def _lama(device: str | None):
    """Một bản LaMa (TorchScript) mỗi thiết bị, nạp lần đầu dùng (có khoá: nhiều luồng thiết kế gọi cùng lúc) -> (model, khoá)."""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with _LAMA_LOCK:
        if dev not in _LAMA:
            _LAMA[dev] = (torch.jit.load(LAMA_PATH, map_location=dev).eval(), threading.Lock())
        return dev, *_LAMA[dev]


def inpaint(img: np.ndarray, mask: np.ndarray, device: str | None = None) -> np.ndarray:
    """LaMa vẽ lại phần trong mặt nạ theo nền xung quanh (một lượt, không khuếch tán). Ảnh đệm phản chiếu tới bội số 8."""
    import torch
    dev, net, lock = _lama(device)
    H, W = img.shape[:2]
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    im = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    mk = np.pad(mask.astype(np.float32), ((0, ph), (0, pw)))
    with lock, torch.no_grad():
        o = net(torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255).to(dev), torch.from_numpy(mk)[None, None].to(dev))
    return (o[0].permute(1, 2, 0).float().cpu().numpy()[:H, :W] * 255).round().clip(0, 255).astype(np.uint8)


def _blend(inside: np.ndarray, outside: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ghép theo mặt nạ, mép mềm 1 px (không lộ đường cắt)."""
    import cv2
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)[..., None]
    return (inside * a + outside * (1 - a)).round().clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------- ảnh xem

def hisam_image(draft: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Đầu ra thô của Hi-SAM: nháp nhạt đi, nét chữ tô HỒNG."""
    vis = draft.astype(np.float32) * 0.55 + 255 * 0.45
    ink = np.asarray(ink, bool)
    vis[ink] = vis[ink] * 0.15 + np.asarray((235, 0, 200), np.float32) * 0.85
    return vis.round().clip(0, 255).astype(np.uint8)


def ink_image(draft: np.ndarray, ink: np.ndarray, mask: np.ndarray, M: dict, kept: set) -> np.ndarray:
    """Bước gỡ chữ: nét bị gỡ XANH LÁ, phần nới / quầng VÀNG, nét của dòng giữ XANH DƯƠNG, nét không thuộc dòng nào HỒNG."""
    H, W = draft.shape[:2]
    hold = np.zeros((H, W), bool)
    for l in M["L"]:
        if l["id"] in kept:
            hold |= line_region(l, H, W, GROW_HOLD)
    ink = np.asarray(ink, bool)
    vis = draft.astype(np.float32) * 0.55 + 255 * 0.45
    for sel, col, a in ((mask & ~ink, (255, 210, 0), 0.55), (mask & ink, (0, 170, 60), 0.85), (ink & ~mask & hold, (30, 110, 240), 0.85),
                        (ink & ~mask & ~hold, (235, 0, 200), 0.85)):
        vis[sel] = vis[sel] * (1 - a) + np.asarray(col, np.float32) * a
    return vis.round().clip(0, 255).astype(np.uint8)
