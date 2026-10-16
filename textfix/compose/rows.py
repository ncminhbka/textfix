"""HÀNG CỦA DANH SÁCH / BẢNG trong nháp (hình học thuần) và HÀNG THỪA. Người dùng chốt 16/10: FLUX vẽ nhiều hàng hơn số câu
khách -> GỠ CẢ HÀNG (mọi ô của hàng: tên món lẫn giá) và DÀN ĐỀU các hàng còn lại trên đúng khoảng nháp -- không chữ lấp
bịa (món, giá, bước), không lỗ trống, không gọi lại VLM.

  G = lists(M)                   # [{"rows": [[id dòng], ...], "cols": n}] -- cột dòng thẳng lề, cùng cỡ, đều nhịp; cột cùng hàng -> bảng
  row_of(G)                      # {id dòng: (danh sách, hàng)}
  log = surplus(U, M, G)         # hàng không câu khách / không dòng giữ nào -> skip, g["surplus"]
  base, log = respace(base, U, M)  # dời dải nền của các hàng còn lại (kèm chấm đầu dòng, icon) + khung chữ theo

Dàn đều dời ĐIỂM ẢNH nền theo dòng ngang: bỏ trọn chu kỳ của hàng thừa (cả pill / chấm đầu dòng của nó), nhân bản nền ở chỗ cắt
giữa các hàng còn lại; chỉ làm khi không đường nối nào lộ (respace); không thì chỉ gỡ chữ hàng thừa.
"""

from __future__ import annotations

import numpy as np

from .geometry import CENTER_TOL, EDGE_TOL

SIZE_RATIO = 1.25   # các hàng một cột: cỡ nháp hai hàng liền nhau lệch <= 1,25 lần (như tập cùng vai, compose.layout.SET_RATIO)
ROW_GAP = 2.0       # ... khe dọc giữa hai hàng liền nhau <= 2 cỡ chữ
PITCH_BREAK = 1.6   # ... nhịp (khoảng cách tâm) > 1,6 lần nhịp trung vị của cột: hết danh sách, bắt đầu khối khác
LIST_MIN = 3        # một cột >= 3 hàng mới là danh sách (hai dòng xếp chồng thường là tiêu đề + phụ đề); bảng >= 2 hàng
ROW_OVERLAP = 0.5   # hai dòng cùng hàng: chồng dọc >= nửa cao dòng thấp hơn
LEAD = 1.5          # vùng dời kéo sang trái 1,5 cỡ chữ: chấm đầu dòng / icon trước hàng đi theo hàng ...
SIDE_MAX = 4.0      # ... và nới thêm tối đa 4 cỡ mỗi bên tới khi mép vùng nối liền phần ảnh không dời (ôm trọn pill / thẻ)
PLAIN_DE = 12.0     # hai dải điểm ảnh NỐI LIỀN được / dòng chèn là NỀN TRƠN: lệch nhau <= 12 ΔE ...
PLAIN_FRAC = 0.03   # ... trừ <= 3% số điểm (vật vắt qua chỗ nối -- vạch kẻ bảng, vệt nước, viền pill -- thì không dời được)
OUTSIDE_FRAC = 0.1  # dòng nhân bản ở chỗ cắt khớp màu nền hai bên vùng danh sách, trừ <= 10% số điểm


def _straight(l: dict) -> bool:
    return not (l.get("path") or abs(l.get("angle") or 0) >= 2 or l.get("vertical") or l.get("orphan"))


def _aligned(A: list, B: list, s: float) -> bool:
    """Hai khung thẳng một lề: trái / phải lệch <= EDGE_TOL cỡ, hoặc tâm lệch <= CENTER_TOL cỡ."""
    return (abs(A[0] - B[0]) <= EDGE_TOL * s or abs(A[2] - B[2]) <= EDGE_TOL * s
            or abs((A[0] + A[2]) / 2 - (B[0] + B[2]) / 2) <= CENTER_TOL * s)


def _same_row(A: list, B: list) -> bool:
    return min(A[3], B[3]) - max(A[1], B[1]) >= ROW_OVERLAP * min(A[3] - A[1], B[3] - B[1])


def columns(L: list[dict]) -> list[list[dict]]:
    """Các CỘT: chuỗi dòng thẳng, thẳng lề, cỡ gần nhau, mỗi dòng nối với dòng gần nhất ngay dưới (khe <= ROW_GAP cỡ); chuỗi
    cắt ở nhịp lạc (> PITCH_BREAK nhịp trung vị). -> các cột >= 2 dòng, dòng trên xuống."""
    Ls = sorted((l for l in L if _straight(l)), key=lambda l: (l["box"][1] + l["box"][3]) / 2)
    nxt, prev = {}, {}
    for i, a in enumerate(Ls):
        A, sa = a["box"], a["size"]
        best = None
        for b in Ls[i + 1:]:
            B, s = b["box"], max(sa, b["size"])
            if B[1] < A[1] + 0.5 * (A[3] - A[1]) or max(sa, b["size"]) > SIZE_RATIO * min(sa, b["size"]):
                continue
            if B[1] - A[3] > ROW_GAP * s or not _aligned(A, B, s):
                continue
            if best is None or B[1] < best["box"][1]:
                best = b
        if best is not None:
            j = best["id"]
            # một dòng chỉ nhận một dòng ngay trên: dòng trên gần hơn thắng
            if j not in prev or A[3] > prev[j]["box"][3]:
                if j in prev:
                    nxt.pop(prev[j]["id"], None)
                prev[j], nxt[a["id"]] = a, best
    out = []
    starts = [l for l in Ls if l["id"] not in {b["id"] for b in nxt.values()}]
    for s0 in starts:
        ch = [s0]
        while ch[-1]["id"] in nxt:
            ch.append(nxt[ch[-1]["id"]])
        if len(ch) < 2:
            continue
        cy = [(l["box"][1] + l["box"][3]) / 2 for l in ch]
        pitch = np.diff(cy)
        med = float(np.median(pitch))
        cur = [ch[0]]
        for l, p in zip(ch[1:], pitch):
            if p > PITCH_BREAK * med:
                if len(cur) >= 2:
                    out.append(cur)
                cur = []
            cur.append(l)
        if len(cur) >= 2:
            out.append(cur)
    return out


def lists(M: dict) -> list[dict]:
    """DANH SÁCH / BẢNG: cột >= LIST_MIN hàng, hoặc >= 2 cột không chồng ngang có >= 2 cặp dòng cùng hàng (tên món | giá) ghép
    thành bảng >= 2 hàng. Hàng = các dòng cùng hàng (chồng dọc >= ROW_OVERLAP), trên xuống. -> [{"rows", "cols", "lines"}]."""
    C = columns(M["L"])
    parent = list(range(len(C)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i, a in enumerate(C):
        for j in range(i + 1, len(C)):
            b = C[j]
            ax = (min(l["box"][0] for l in a), max(l["box"][2] for l in a))
            bx = (min(l["box"][0] for l in b), max(l["box"][2] for l in b))
            if min(ax[1], bx[1]) > max(ax[0], bx[0]):   # chồng ngang: hai cột cùng chỗ, không phải hai cột của một bảng
                continue
            if sum(any(_same_row(x["box"], y["box"]) for y in b) for x in a) >= 2:
                parent[root(j)] = root(i)
    G = {}
    for i, c in enumerate(C):
        G.setdefault(root(i), []).append(c)
    out = []
    for cols in G.values():
        Ls = sorted({l["id"]: l for c in cols for l in c}.values(), key=lambda l: (l["box"][1] + l["box"][3]) / 2)
        rows = []
        for l in Ls:
            if rows and any(_same_row(l["box"], o["box"]) for o in rows[-1]):
                rows[-1].append(l)
            else:
                rows.append([l])
        if len(rows) >= LIST_MIN or (len(cols) >= 2 and len(rows) >= 2):
            out.append({"rows": [[l["id"] for l in sorted(r, key=lambda l: l["box"][0])] for r in rows], "cols": len(cols),
                        "lines": [l["id"] for l in Ls]})
    return out


def row_of(G: list[dict]) -> dict:
    """{id dòng: (chỉ số danh sách, chỉ số hàng)}."""
    return {m: (gi, ri) for gi, g in enumerate(G) for ri, r in enumerate(g["rows"]) for m in r}


def surplus(U: list[dict], M: dict, G: list[dict]) -> list[str]:
    """HÀNG THỪA: hàng của danh sách không có dòng nào viết câu khách hay giữ nguyên (chữ lấp / câu trùng / rác) -> mọi dòng của
    hàng thành skip (u["surplus"]), g["surplus"] = các hàng đó. Cả danh sách đều thừa: chỉ gỡ, không dàn. -> nhật ký."""
    owner = {m: u for u in U for m in u["marks"]}
    real = lambda m: m in owner and ((owner[m]["kind"] == "text" and owner[m]["clients"]) or owner[m]["kind"] == "keep")  # noqa: E731
    log = []
    for gi, g in enumerate(G):
        g["surplus"] = [ri for ri, r in enumerate(g["rows"]) if not any(real(m) for m in r)]
        if not g["surplus"]:
            continue
        for ri in g["surplus"]:
            for m in g["rows"][ri]:
                u = owner.get(m)
                if u is None:
                    continue
                u["marks"] = [x for x in u["marks"] if x != m]
                U.append({"id": f"r_{m}", "kind": "skip", "marks": [m], "clients": [], "split": None, "text_vlm": "",
                          "role_vlm": None, "surplus": gi, "why": "hàng thừa của danh sách: gỡ cả hàng, dàn đều các hàng còn lại"})
        log.append(f"danh sách #{gi} ({len(g['rows'])} hàng, {g['cols']} cột): hàng thừa {[g['rows'][ri] for ri in g['surplus']]} "
                   "-> gỡ cả hàng")
    U[:] = [u for u in U if u["marks"]]
    return log


def respace(base: np.ndarray, U: list[dict], M: dict) -> tuple[np.ndarray, list[str]]:
    """DÀN ĐỀU các hàng còn lại của mỗi danh sách có hàng thừa, bằng cách dời ĐIỂM ẢNH theo dòng ngang:
      1. mỗi hàng là một CHU KỲ dọc: từ giữa khe chữ phía trên tới giữa khe chữ phía dưới (hàng đầu / cuối: nửa khe trung vị ra
         ngoài) -- chu kỳ ôm trọn pill / thẻ / chấm đầu dòng / icon của hàng khi chúng cân quanh chữ;
      2. gỡ hàng thừa = bỏ trọn chu kỳ của nó (cả pill của nó); các hàng còn lại giữ nguyên điểm ảnh;
      3. dàn đều = chèn bản sao dòng điểm ảnh ở chỗ cắt (nền giữa hai hàng) vào mỗi khe giữa các hàng còn lại cho đủ lại khoảng
         nháp (còn một hàng: chèn đều trên / dưới) -- pill, viền không bị kéo méo;
      4. CHỈ LÀM khi không đường nối nào lộ (_seam): hai mép chỗ gỡ khớp nhau, dòng nhân bản là nền trơn, mép trái / phải vùng dời
         nối liền phần ảnh không dời (vùng tự nới ngang tối đa SIDE_MAX cỡ để ôm trọn pill), vùng không chứa dòng chữ nào khác.
         Không được (pill sát nhau, vạch kẻ bảng, vật vắt qua): chỉ gỡ chữ hàng thừa, nhật ký ghi lý do.
    Khung chữ các đơn vị dời theo hàng của nó. -> (nền mới, nhật ký)."""
    from ..color import lab
    G = M.get("lists") or []
    if not any(g.get("surplus") for g in G):
        return base, []
    by, out, log = M["by"], base.copy(), []
    L = lab(base)
    owner = {m: u for u in U for m in u["marks"]}
    for gi, g in enumerate(G):
        if not g.get("surplus") or len(g["surplus"]) == len(g["rows"]):
            continue
        p = _plan(g, by, owner, L)
        if isinstance(p, str):
            log.append(f"danh sách #{gi}: không dàn đều ({p}) -> chỉ gỡ hàng thừa")
            continue
        why = _blocked(g, p, U, M, owner, L)
        if why:
            log.append(f"danh sách #{gi}: không dàn đều ({why}) -> chỉ gỡ hàng thừa")
            continue
        c0, c1, xa, xb = p["c0"], p["c1"], p["xa"], p["xb"]
        out[c0:c1, xa:xb] = base[p["src"], xa:xb]
        moved = []
        for b in p["keep"]:
            if not b["dy"]:
                continue
            for u in {id(owner[m]): owner[m] for ri in b["rows"] for m in g["rows"][ri] if m in owner}.values():
                _shift(u, b["dy"])
                moved.append(f"{u['id']} {b['dy']:+d}px")
        log.append(f"danh sách #{gi}: gỡ {len(p['bands']) - len(p['keep'])} chu kỳ hàng, dàn đều {len(p['keep'])} hàng trên y "
                   f"{c0}-{c1}, x {xa}-{xb}: {', '.join(moved) or 'không hàng nào phải dời'}")
    return out, log


def _plan(g: dict, by: dict, owner: dict, L: np.ndarray) -> dict | str:
    """Chu kỳ các hàng, dãy dòng nguồn của vùng mới, vùng ngang -> dict; không được -> lý do."""
    H, W = L.shape[:2]
    ids = set(g["lines"])
    s = float(np.median([by[m]["size"] for m in ids]))
    bands = []   # hàng dính nhau qua một đơn vị nhiều dòng -> một chu kỳ
    for ri, r in enumerate(g["rows"]):
        us = {id(owner[m]) for m in r if m in owner}
        if bands and us & bands[-1]["us"]:
            bands[-1]["rows"].append(ri)
            bands[-1]["us"] |= us
        else:
            bands.append({"rows": [ri], "us": us})
    for b in bands:
        ls = [by[m] for ri in b["rows"] for m in g["rows"][ri]]
        b["top"], b["bot"] = min(l["box"][1] for l in ls), max(l["box"][3] for l in ls)
        b["gone"] = all(ri in g["surplus"] for ri in b["rows"])
    keep = [b for b in bands if not b["gone"]]
    if len(keep) == len(bands):
        return "không hàng nào gỡ trọn được"
    gaps = [b["top"] - a["bot"] for a, b in zip(bands, bands[1:])]
    h = float(np.median(gaps)) if gaps else s
    if h < 2:
        return "các hàng sát nhau, không có khe để cắt"
    cut = [int(round(bands[0]["top"] - h / 2))] + [int(round((a["bot"] + b["top"]) / 2)) for a, b in zip(bands, bands[1:])] \
        + [int(round(bands[-1]["bot"] + h / 2))]
    if cut[0] < 1 or cut[-1] > H - 1 or any(b <= a for a, b in zip(cut, cut[1:])):
        return "chu kỳ hàng chạm mép ảnh / chồng nhau"
    for i, b in enumerate(bands):
        b["c0"], b["c1"] = cut[i], cut[i + 1]
    D = sum(b["c1"] - b["c0"] for b in bands if b["gone"])
    ins = []   # dòng được nhân bản (chỗ cắt, nền giữa hai hàng)
    src = []
    if len(keep) == 1:
        top, bot = [cut[0]] * (D // 2), [cut[-1] - 1] * (D - D // 2)
        ins = [cut[0], cut[-1] - 1]
        keep[0]["new"] = cut[0] + len(top)
        src = top + list(range(keep[0]["c0"], keep[0]["c1"])) + bot
    else:
        extra = np.diff(np.round(np.linspace(0, D, len(keep))).astype(int))
        for k, b in enumerate(keep):
            b["new"] = cut[0] + len(src)
            src += list(range(b["c0"], b["c1"]))
            if k + 1 < len(keep):
                src += [b["c1"]] * int(extra[k])
                ins.append(b["c1"])
    for b in keep:
        b["dy"] = b["new"] - b["c0"]
    src = np.asarray(src, int)
    # vùng ngang: chữ + LEAD cỡ bên trái (chấm đầu dòng / icon), nới thêm tới khi mép vùng nối liền phần ảnh không dời
    x0 = min(by[m]["box"][0] for m in ids) - LEAD * s
    x1 = max(by[m]["box"][2] for m in ids) + 0.3 * s
    c0, c1 = cut[0], cut[-1]
    xa = next((x for x in range(int(max(0, x0)), int(max(-1, x0 - SIDE_MAX * s)), -1)
               if x == 0 or _match(L[src, x], L[c0:c1, x - 1])), None)
    xb = next((x for x in range(int(min(W, x1)), int(min(W + 1, x1 + SIDE_MAX * s))) if x == W or _match(L[src, x - 1], L[c0:c1, x])), None)
    if xa is None or xb is None:
        return "mép trái / phải vùng danh sách không nối liền được (vật vắt qua mép)"
    return {"bands": bands, "keep": keep, "src": src, "ins": ins, "c0": c0, "c1": c1, "xa": xa, "xb": xb, "size": s}


def _blocked(g: dict, p: dict, U: list, M: dict, owner: dict, L: np.ndarray) -> str | None:
    """Lý do không dời được vùng danh sách, hoặc None: dòng chữ khác trong vùng; đơn vị vắt ra ngoài danh sách; đường nối lộ
    (chỗ gỡ hai mép không khớp, dòng nhân bản không phải nền trơn)."""
    xa, xb, c0, c1 = p["xa"], p["xb"], p["c0"], p["c1"]
    ids = set(g["lines"])
    for l in M["L"]:
        if l["id"] in ids:
            continue
        b = l["box"]
        w, h = min(b[2], xb) - max(b[0], xa), min(b[3], c1) - max(b[1], c0)
        if w > 0 and h > 0 and w * h > 0.2 * max(1.0, (b[2] - b[0]) * (b[3] - b[1])):
            return f"dòng {l['id']} nằm trong vùng danh sách"
    for m in ids:
        u = owner.get(m)
        if u and any(x not in ids for x in u["marks"]):
            return f"đơn vị {u['id']} vắt ra ngoài danh sách"
    seq = [c0 - 1] + p["src"].tolist() + [c1]
    for a, b in zip(seq, seq[1:]):
        if b != a + 1 and b != a and not _match(L[a, xa:xb], L[b, xa:xb]):
            return f"chỗ gỡ y {a}|{b} hai mép không khớp"
    for r in p["ins"]:
        if not _plain(L[r - 1:r + 2, xa:xb]) or not _outside(L, r, xa, xb):
            return f"chỗ cắt y {r} không phải nền (viền pill / vạch kẻ / vật)"
    return None


def _outside(L: np.ndarray, r: int, xa: int, xb: int, k: int = 6) -> bool:
    """Dòng r trong vùng [xa, xb) là NỀN NGOÀI danh sách: khớp màu nền ngay hai bên vùng (nội suy trái -> phải, nền chuyển màu
    vẫn qua) -- dòng ở chỗ cắt mà là viền / thân pill (pill sát nhau) thì không nhân bản được."""
    W = L.shape[1]
    lt, rt = L[r, max(0, xa - k):xa], L[r, xb:min(W, xb + k)]
    if not len(lt) and not len(rt):
        return True
    cl = np.median(lt if len(lt) else rt, axis=0)
    cr = np.median(rt if len(rt) else lt, axis=0)
    t = np.linspace(0, 1, xb - xa)[:, None]
    return bool((np.linalg.norm(L[r, xa:xb] - (cl * (1 - t) + cr * t), axis=-1) > PLAIN_DE).mean() <= OUTSIDE_FRAC)


def _match(a: np.ndarray, b: np.ndarray) -> bool:
    """Hai dải điểm ảnh (Lab) nối liền được: số điểm lệch > PLAIN_DE (ΔE) <= PLAIN_FRAC."""
    return float((np.linalg.norm(a - b, axis=-1) > PLAIN_DE).mean()) <= PLAIN_FRAC if a.size else True


def _plain(lab_strip: np.ndarray) -> bool:
    """Dải nền trơn: số cột điểm ảnh biến thiên dọc > PLAIN_DE (ΔE) <= PLAIN_FRAC."""
    if lab_strip.shape[0] < 2 or lab_strip.shape[1] < 1:
        return True
    span = np.linalg.norm(lab_strip.max(axis=0) - lab_strip.min(axis=0), axis=-1)
    return float((span > PLAIN_DE).mean()) <= PLAIN_FRAC


def _shift(u: dict, dy: float) -> None:
    """Dời khung chữ của đơn vị theo chiều dọc (khung đơn vị, khung nét từng dòng, phần tử dựng)."""
    mv = lambda b: [b[0], b[1] + dy, b[2], b[3] + dy]  # noqa: E731
    if u.get("box"):
        u["box"] = mv(u["box"])
    u["ib"] = [mv(b) for b in u.get("ib") or []]
    for e in u.get("els") or []:
        e["box"] = mv(e["box"])
