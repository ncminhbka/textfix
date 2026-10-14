"""DỰNG các đơn vị chữ (textfix/fill.py) lên nền (nháp đã gỡ chữ) trong Chromium; báo lỗi khách quan cho bước sửa bằng code.

  poster, res, log = render_units(page, base, U, M)   # mỗi phần tử = một div tại khung nét nháp, chữ nguyên văn câu khách
  errs = check_units(U, res)                           # nội dung chồng nhau, tràn, phải co quá nhỏ, chìm nền
  U2, log = autofix_units(U, errs, res)                # màu tương phản, nén sâu hơn, cắt khung chồng

Dòng model vẽ nghiêng: div dựng thẳng rồi xoay đúng góc; dòng cong: SVG textPath. TẬP cùng vai (fill.layout): đo bề rộng tự
nhiên mọi mục ở cỡ chung, rồi CẢ TẬP một mức nén ngang + một cỡ theo mục chật nhất (không mục nào co riêng).
"""

from __future__ import annotations

import base64
import io
import re

import numpy as np

OVERLAP = 0.1   # chồng nhau > 10% diện tích phần tử nhỏ hơn = lỗi báo cho vòng sửa
CONDENSE = 0.85   # chữ dài hơn ô theo chiều ngang: nén ngang (scaleX) tới 0.85 TRƯỚC khi co cỡ -- chữ giữ chiều cao như nháp
ROT_MIN = 2.0   # độ: dòng model vẽ nghiêng >= 2 độ thì dựng nghiêng đúng góc đó (dưới 2 độ là nhiễu đo của OCR)
EMOJI = re.compile("[\U0001F000-\U0001FAFF️‍]")   # emoji (không có font emoji trong bộ font)
def to_data_uri(arr: np.ndarray) -> str:
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def plain(h: str) -> str:
    """Chữ trần của HTML (bỏ thẻ, gộp khoảng trắng)."""
    import html as _html
    return " ".join(_html.unescape(re.sub(r"<[^>]+>", " ", h or "")).split())


# ---------------------------------------------------------------------------------------------------- khung
def _ids(op: dict) -> list[str]:
    ids = op.get("marks") or []
    if op.get("kind") == "text" and any(m.startswith("L") for m in ids):
        ids = [m for m in ids if m.startswith("L")]
    return ids


def op_geom(op: dict, M: dict) -> dict | None:
    """Khung DỰNG của lệnh = hợp khung các ô. Dòng NGHIÊNG (trung vị góc OCR các dòng L >= ROT_MIN): khung trong hệ trục xoay
    theo góc đó (ôm sát dòng nghiêng), div dựng thẳng rồi xoay. -> {rect (chưa xoay), angle, poly (4 góc thật), aabb}.
    Lệnh có fix.box (autofix cắt / thu khung): rect = fix.box (hệ trục của dòng), góc giữ như đo."""
    g = _geom(op, M)
    fb = (op.get("fix") or {}).get("box")
    if g is None or not fb:
        return g
    t = np.radians(g["angle"])
    c, sn = np.cos(t), np.sin(t)
    cx, cy, w, h = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2, fb[2] - fb[0], fb[3] - fb[1]
    corners = [[cx + u * c - v * sn, cy + u * sn + v * c] for u, v in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    W, H = M["size"]
    return {"rect": list(fb), "angle": g["angle"], "poly": corners,
            "aabb": [max(0.0, min(xs)), max(0.0, min(ys)), min(float(W), max(xs)), min(float(H), max(ys))]}


def _geom(op: dict, M: dict) -> dict | None:
    by, (W, H) = M["by"], M["size"]
    ids = [m for m in _ids(op) if m in by]
    if not ids:
        return None
    Ls = [by[m] for m in ids if m.startswith("L")]
    ang = 0.0
    if Ls:
        # GÓC CỦA KHỐI = góc của dòng DÀI nhất so với cao chữ (13/10 dev3 g13: "ĂN" hai chữ cái OCR đo -4 độ, "NGON" bên dưới
        # -19 độ, cùng một nhãn; trung vị ra -11, sai cả hai). Góc đo trên dòng càng dài càng chắc; dòng vài chữ cái là nhiễu.
        def reach(m):
            pl = m.get("poly")
            if pl and len(pl) == 4:
                e = sorted(float(np.hypot(pl[i][0] - pl[(i + 1) % 4][0], pl[i][1] - pl[(i + 1) % 4][1])) for i in range(4))
                return e[-1] / max(1.0, e[0])
            return (m["box"][2] - m["box"][0]) / max(1.0, m["box"][3] - m["box"][1])
        tilted = [m for m in Ls if abs(m.get("angle") or 0.0) >= ROT_MIN]
        if len(tilted) * 2 >= len(Ls):   # đa số dòng nghiêng thì khối nghiêng
            ang = float(max(tilted, key=reach).get("angle") or 0.0)
    if abs(ang) < ROT_MIN:
        bs = [by[m]["box"] for m in ids]
        r = [max(0.0, min(b[0] for b in bs)), max(0.0, min(b[1] for b in bs)), min(float(W), max(b[2] for b in bs)),
             min(float(H), max(b[3] for b in bs))]
        if r[2] - r[0] < 2 or r[3] - r[1] < 2:
            return None
        return {"rect": r, "angle": 0.0, "poly": [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], "aabb": r}
    t = np.radians(ang)
    c, sn = np.cos(t), np.sin(t)
    pts = []
    for m in ids:
        mk = by[m]
        pts += mk["poly"] if "poly" in mk else [[mk["box"][0], mk["box"][1]], [mk["box"][2], mk["box"][1]],
                                               [mk["box"][2], mk["box"][3]], [mk["box"][0], mk["box"][3]]]
    P_ = np.asarray(pts, float)
    u, v = P_[:, 0] * c + P_[:, 1] * sn, -P_[:, 0] * sn + P_[:, 1] * c   # sang hệ trục của dòng
    u0, u1, v0, v1 = u.min(), u.max(), v.min(), v.max()
    uc, vc, w, h = (u0 + u1) / 2, (v0 + v1) / 2, u1 - u0, v1 - v0
    cx, cy = uc * c - vc * sn, uc * sn + vc * c
    corners = [[uu * c - vv * sn, uu * sn + vv * c] for uu, vv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    return {"rect": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], "angle": ang, "poly": corners,
            "aabb": [max(0.0, min(xs)), max(0.0, min(ys)), min(float(W), max(xs)), min(float(H), max(ys))]}


def path_of(op: dict, M: dict) -> list | None:
    """Đường chân của lệnh chữ viết vào ô cung / lượn (textfix/geo.py) -> các điểm, hướng đọc trái -> phải; lệnh thường: None.
    Lệnh chữ chỉ dựng theo các ô L của nó (_ids), nên ô S / I đi kèm không đổi gì. Lệnh có ô cong KHÔNG BAO GIỜ lùi về khối chữ
    thẳng trong khung bao (13/10 run ba93: chữ thẳng đè lên hình giữa con dấu): ô cong đi một mình;
    lệnh chưa qua validate mà gộp nhiều ô L thì viết theo đường cong dài nhất."""
    if op.get("kind") != "text":
        return None
    pts = [M["by"][m]["path"] for m in op.get("marks") or [] if m.startswith("L") and m in M["by"] and len(M["by"][m].get("path") or []) >= 2]
    if not pts:
        return None
    return max(pts, key=lambda pt: sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(pt[:-1], pt[1:])))


def _path_polys(path: list, size: float) -> list:
    """Dải chữ dọc đường chân: mỗi khúc một tứ giác lồi (chân chữ -> đỉnh chữ hoa, phía trên của hướng đọc)."""
    out, hh = [], 0.8 * size
    for (ax, ay), (bx, by) in zip(path[:-1], path[1:]):
        dx, dy = bx - ax, by - ay
        ln = float(np.hypot(dx, dy)) or 1.0
        nx, ny = dy / ln, -dx / ln
        out.append([[ax, ay], [bx, by], [bx + nx * hh, by + ny * hh], [ax + nx * hh, ay + ny * hh]])
    return out


def _multi_ov(A: list, B: list) -> float:
    """Phần giao hai HỢP đa giác lồi (chữ theo đường = nhiều khúc) / diện tích hợp nhỏ hơn."""
    import cv2
    pa, pb = [np.asarray(x, np.float32) for x in A], [np.asarray(x, np.float32) for x in B]
    sm = min(sum(abs(cv2.contourArea(x)) for x in pa), sum(abs(cv2.contourArea(x)) for x in pb))
    if sm <= 0:
        return 0.0
    return float(sum(cv2.intersectConvexConvex(x, y)[0] for x in pa for y in pb)) / sm


def op_box(op: dict, M: dict) -> list[float] | None:
    """Khung thẳng ôm khung dựng của lệnh (để kiểm, cắt ảnh lỗi, vẽ debug)."""
    g = op_geom(op, M)
    return g["aabb"] if g else None


# ---------------------------------------------------------------------------------------------------- dựng
def _attr(tag: str, name: str, default=None):
    m = re.search(rf"""{name}\s*=\s*(["'])(.*?)\1""", tag)
    return m.group(2) if m else default


def font_key(k) -> str | None:
    """Khoá font VLM ghi -> khoá CATALOG: đúng khoá, tên họ font ("Dancing Script"), hoặc gần nhất ("bevietsans" -> bevietnam)."""
    from difflib import get_close_matches
    from .fonts import CATALOG
    if not isinstance(k, str) or not k.strip():
        return None
    if k in CATALOG:
        return k
    norm = lambda x: re.sub(r"[^a-z]", "", x.lower())   # noqa: E731
    names = {norm(v[0]): key for key, v in CATALOG.items()} | {norm(key): key for key in CATALOG}
    m = get_close_matches(norm(k), list(names), n=1, cutoff=0.6)
    return names[m[0]] if m else None


WEIGHT = lambda m: (300 if m < 0.07 else 400 if m < 0.11 else 700 if m < 0.16 else 900)   # noqa: E731  nét / cỡ chữ -> độ đậm (như slots._look)
HEAD = 1.4   # dòng cao >= 1.4 cỡ chữ trung vị của poster = tiêu đề -> mặc định style.display_font


def _clip_text_shadow(h: str) -> str:
    """Chữ tô gradient (background-clip:text, màu chữ trong suốt): text-shadow vẽ DƯỚI lớp tô và lộ qua chữ trong suốt -> chữ
    vàng thành nâu (08/10 h08_s1). Đổi bóng sang filter:drop-shadow (bóng của cả khối chữ đã tô), viền -webkit-text-stroke
    vẽ sau phần tô (paint-order) để viền chỉ ở mép ngoài."""
    def fix(m):
        st = m.group(2)
        if not re.search(r"background-clip\s*:\s*text", st, flags=re.I):
            return m.group(0)
        sh = re.search(r"(?<![-\w])text-shadow\s*:\s*([^;]+);?", st, flags=re.I)
        if sh:
            parts = [x.strip() for x in re.split(r",(?![^(]*\))", sh.group(1)) if x.strip() and x.strip() != "none"]
            st = st.replace(sh.group(0), "") + (";filter:" + " ".join(f"drop-shadow({x})" for x in parts) if parts else "")
        if "text-stroke" in st:
            st += ";paint-order:stroke fill"
        return f"style={m.group(1)}{st}{m.group(1)}"
    return re.sub(r"""style=(["'])(.*?)\1""", fix, h, flags=re.S)


def expand(h: str, fonts_used: set) -> tuple[str, list[str]]:
    """HTML của VLM -> HTML dựng được: an toàn (bỏ script / on* / URL / img) + thay thẻ i-font. TextFix CHỈ SỬA CHỮ (14/10): icon,
    hàng sao, hình vẽ là điểm ảnh của nháp -- thẻ <i-icon>, <i-stars>, <svg> VLM lỡ viết bị gỡ, ghi log, không vẽ."""
    from .fonts import family
    log = []
    h = re.sub(r"<script.*?</script>", "", h or "", flags=re.S | re.I)
    h = re.sub(r"\son\w+\s*=\s*([\"']).*?\1", "", h, flags=re.I)
    h = re.sub(r"url\([^)]*\)", "none", h, flags=re.I)
    h = re.sub(r"<img[^>]*>", "", h, flags=re.I)
    for pat in (r"<i-icon\b[^>]*?(/>|>\s*</i-icon>|>)", r"<i-stars\b[^>]*?(/>|>\s*</i-stars>|>)", r"<svg\b.*?</svg>"):
        h2 = re.sub(pat, "", h, flags=re.I | re.S)
        if h2 != h:
            log.append("bỏ icon / hình VLM viết (TextFix chỉ sửa chữ)")
            h = h2

    def font(m):
        t = m.group(0)
        raw, wt = _attr(t, "key", ""), _attr(t, "weight", "")
        key = font_key(raw)
        if key is None:
            log.append(f"font '{raw}' không có -> font mặc định")
            return f"<span style=\"{'font-weight:' + wt + ';' if wt else ''}\">"
        if key != raw:
            log.append(f"font '{raw}' -> '{key}'")
        fonts_used.add(key)
        return f"<span style=\"font-family:{family(key)};{'font-weight:' + wt + ';' if wt else ''}\">"

    h = re.sub(r"<i-font\b[^>]*>", font, h, flags=re.I)
    h = re.sub(r"</i-font>", "</span>", h, flags=re.I)
    h = _clip_text_shadow(h)
    # font ghi thẳng bằng CSS (font-family:'Dancing Script') cũng phải được nạp (07/10 h08: rơi về font dự phòng)
    for fam in re.findall(r"font-family\s*:\s*([^;\"]+)", h, flags=re.I):
        k = font_key(fam.split(",")[0].strip(" '\""))
        if k:
            fonts_used.add(k)
    return h, log


RENDER_JS = r"""
async (o) => {
  const c = document.getElementById('c');
  // nạp HẾT font khai báo trước khi đo: fonts.ready xong ngay khi chưa có chữ nào dùng font (chưa bắt đầu tải) -> lần dựng đầu
  // của trang đo bằng font dự phòng (cỡ / tràn sai), các lần sau đúng nhờ bộ nhớ đệm
  await Promise.all([...document.fonts].map(f => f.load().catch(() => null)));
  await document.fonts.ready;
  const out = [];
  for (const e of o.els) {
    // CHỮ THEO ĐƯỜNG (ô cung / lượn của textfix/geo.py: e.path = các điểm ĐƯỜNG CHÂN, hướng đọc trái -> phải): dựng bằng SVG
    // textPath, chữ căn giữa đường. Kiểu chữ lấy từ chính lệnh của VLM (đo trên một khối ẩn), như mọi lệnh khác. Chữ dài hơn
    // đường: nén ngang tới o.condense rồi mới co cỡ -- cùng thứ tự với dòng thẳng.
    if (e.path && e.path.length >= 2) {
      const c0 = document.getElementById('c'), R0 = c0.getBoundingClientRect();
      const h = document.createElement('div');
      h.style.cssText = `position:absolute;visibility:hidden;white-space:nowrap;line-height:1.15;font-family:${e.font};color:${e.color};` +
                        `font-size:${e.size}px;` + (e.style || '');
      h.innerHTML = e.html;
      c0.appendChild(h);
      let el = h, best = 0;
      const wk = document.createTreeWalker(h, NodeFilter.SHOW_TEXT);
      for (let n = wk.nextNode(); n; n = wk.nextNode()) { const k = n.textContent.trim().length; if (k > best) { best = k; el = n.parentElement; } }
      const cs = getComputedStyle(el);
      let txt = h.textContent.replace(/\s+/g, ' ').trim();
      if (cs.textTransform === 'uppercase') txt = txt.toUpperCase(); else if (cs.textTransform === 'lowercase') txt = txt.toLowerCase();
      // chữ tô dải màu (background-clip:text, ruột chữ trong suốt): SVG không tô được nền vào chữ -> lấy màu chữ của lệnh / màu đo
      const clear = v => !v || v === 'transparent' || /rgba\([^)]*,\s*0\)$/.test(v);
      const grad = /text/.test(cs.webkitBackgroundClip || cs.backgroundClip || '') ? (cs.backgroundImage.match(/rgba?\([^)]*\)/) || [])[0] : null;
      const fill = [cs.webkitTextFillColor, grad, cs.color, e.color].find(v => !clear(v)) || '#111';
      const sty = {ff: cs.fontFamily, fw: cs.fontWeight, fi: cs.fontStyle, fg: fill, ls: cs.letterSpacing,
                   sw: parseFloat(cs.webkitTextStrokeWidth) || 0, sc: cs.webkitTextStrokeColor, sh: cs.textShadow,
                   k: (parseFloat(cs.fontSize) || e.size) / e.size};
      h.remove();
      const NS = 'http://www.w3.org/2000/svg', mk = t => document.createElementNS(NS, t);
      const svg = mk('svg'), p = mk('path'), t = mk('text'), tp = mk('textPath');
      svg.setAttribute('width', R0.width); svg.setAttribute('height', R0.height);
      svg.setAttribute('viewBox', `0 0 ${R0.width} ${R0.height}`);
      svg.style.cssText = 'position:absolute;left:0;top:0;overflow:visible;pointer-events:none';
      if (e.cls) svg.setAttribute('class', e.cls);
      p.setAttribute('id', 'tp_' + e.id); p.setAttribute('fill', 'none');
      p.setAttribute('d', 'M' + e.path.map(q => q[0].toFixed(1) + ',' + q[1].toFixed(1)).join(' L'));
      let s = e.size * sty.k;
      t.style.cssText = `font-family:${sty.ff};font-weight:${sty.fw};font-style:${sty.fi};font-size:${s}px;fill:${sty.fg};` +
                        `letter-spacing:${sty.ls};` + (sty.sh && sty.sh !== 'none' ? `text-shadow:${sty.sh};` : '') +
                        (sty.sw > 0 ? `stroke:${sty.sc};stroke-width:${sty.sw}px;stroke-linejoin:round;` +
                                      (/stroke/.test(cs.paintOrder || '') ? `paint-order:${cs.paintOrder};` : '') : '');   // như -webkit-text-stroke
      tp.setAttribute('href', '#tp_' + e.id); tp.setAttribute('startOffset', '50%'); tp.setAttribute('text-anchor', 'middle');
      tp.textContent = txt;
      t.appendChild(tp); svg.appendChild(p); svg.appendChild(t); c0.appendChild(svg);
      const span = p.getTotalLength();
      let w = t.getComputedTextLength(), sx = 1, k = 0;
      const w0 = w;
      if (w > span) {   // nén ngang (giãn chữ + nét) rồi mới co cỡ
        sx = Math.max(e.condense || o.condense, span / w);
        tp.setAttribute('lengthAdjust', 'spacingAndGlyphs'); tp.setAttribute('textLength', (w * sx).toFixed(1));
        while (w * sx > span + 1 && s > o.floor && k < 80) {
          s *= 0.95; t.style.fontSize = s + 'px'; tp.removeAttribute('textLength'); w = t.getComputedTextLength();
          tp.setAttribute('textLength', (w * sx).toFixed(1)); k++;
        }
      }
      const used = w * sx, a0 = Math.max(0, (span - used) / 2), N = 8, polys = [];
      // nét mực = dải chữ dọc đường (từng khúc là một tứ giác lồi: chân chữ -> đỉnh chữ hoa, phía "trên" của hướng đọc)
      for (let i = 0; i < N; i++) {
        const u0 = a0 + used * i / N, u1 = a0 + used * (i + 1) / N;
        const A = p.getPointAtLength(u0), B = p.getPointAtLength(u1);
        const dx = B.x - A.x, dy = B.y - A.y, L = Math.hypot(dx, dy) || 1, nx = dy / L, ny = -dx / L, hh = 0.8 * s;
        polys.push([[A.x, A.y], [B.x, B.y], [B.x + nx * hh, B.y + ny * hh], [A.x + nx * hh, A.y + ny * hh]]);
      }
      const xs = polys.flat().map(q => q[0]), ys = polys.flat().map(q => q[1]);
      const bb = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
      out.push({id: e.id, size: s / sty.k, shrink: e.size ? s / sty.k / e.size : 1, over: Math.max(0, used - span - 1), o0: [Math.max(0, w0 - span), 0],
                fg: sty.fg, fx: sty.sw > 0 || (sty.sh && sty.sh !== 'none'), fs: s, ff: sty.ff, fw: parseInt(sty.fw), fi: sty.fi, ls: sty.ls,
                tt: cs.textTransform, sx: +sx.toFixed(2), ink0: null, angle: 0, runs: [], path: true, polys, ink: bb, box: bb});
      continue;
    }
    const d = document.createElement('div');
    d.style.cssText = `position:absolute;left:${e.box[0]}px;top:${e.box[1]}px;box-sizing:border-box;display:flex;align-items:center;` +
      `justify-content:center;text-align:center;line-height:1.15;font-synthesis:style;font-family:${e.font};color:${e.color};font-size:${e.size}px;` +
      (e.fit === 'grow' ? `min-width:${e.box[2] - e.box[0]}px;min-height:${e.box[3] - e.box[1]}px;width:max-content;` :
                          `width:${e.box[2] - e.box[0]}px;height:${e.box[3] - e.box[1]}px;`) + (e.style || '');
    d.innerHTML = e.html;
    if (e.cls) d.className = e.cls;
    // CHỮ TRẦN LẪN THẺ ngay trong khung (12/10 form1 f09: câu trích 'chữ <span>đậm</span> chữ <span>đậm</span> chữ' không bọc
    // thẻ nào): khung là flex nên mỗi mẩu chữ / mỗi span thành một CỘT xếp ngang -> tràn, co còn 0.32. Có mẩu chữ trần ở cấp
    // ngoài = nội dung chảy theo dòng -> bọc một span (một con flex). Bố cục cột thật (các span, không chữ trần) giữ nguyên.
    if (d.childNodes.length > 1 && [...d.childNodes].some(n => n.nodeType === 3 && n.textContent.trim()) &&
        ![...d.children].some(x => /display\s*:\s*(inline-)?flex/.test(x.getAttribute('style') || ''))) {
      const w = document.createElement('span');
      while (d.firstChild) w.appendChild(d.firstChild);
      d.appendChild(w);
    }
    // CỘT FLEX trong thẻ font: VLM viết <i-font><span style="flex:1">tên</span><span style="flex:0 0 30%">giá</span></i-font> --
    // span bọc font (inline) là con flex duy nhất, tên và giá dính liền "Bún bò tái40.000đ" (09/10 dev2 d05) -> bọc font
    // display:contents, các span flex thành con flex thật của khung (vẫn thừa hưởng font)
    // (icon / sao là SVG có "flex:none" trong style -- không phải cột flex: 13/10 accept1 a04, ba dòng có icon đầu dòng bị trải
    // thành ba cột hẹp)
    for (const x of d.querySelectorAll('[style*="flex"]')) {
      if (x instanceof SVGElement) continue;
      // (con trực tiếp của khung flex bị "blockify": display tính ra là block, không phải inline -- xét theo thẻ span)
      for (let p = x.parentElement; p && p !== d && p.tagName === 'SPAN' && !/flex|grid/.test(getComputedStyle(p).display);
           p = p.parentElement)
        p.style.display = 'contents';
    }
    // dòng nghiêng: renderer tự xoay đúng góc đo (bên dưới) -- transform VLM ghi ở khung (rotate(-14deg)) bỏ trước khi đo,
    // không thì "đo trước khi xoay" là đo khối đã xoay (tràn / nét sai)
    if (e.angle) {
      d.style.transform = 'none';
      // ... và transform xoay / xiên VLM ghi ở các thẻ BÊN TRONG (13/10 dev3 g05, g13: span rotate(31deg) + renderer xoay 31 độ
      // = chữ xoay hai lần, dựng đứng)
      for (const x of d.querySelectorAll('[style]')) if (/rotate|skew|matrix/.test(x.style.transform)) x.style.transform = 'none';
    }
    c.appendChild(d);
    const R0 = c.getBoundingClientRect();
    // TRÀN = nội dung thật (mọi con, kể cả SVG) vượt khung ở BẤT KỲ phía nào (nội dung căn giữa tràn cả trái -- scrollWidth chỉ
    // đếm phía phải: p20 hàng sao thò ra ngoài huy hiệu bên trái)
    // KHUNG Ô ôm NÉT MỰC (cả dấu thanh), còn hộp chữ của trình duyệt = ascent + descent của font (đệm trên / dưới nét): so
    // thẳng thì mọi dòng "tràn dọc" và bị co ~0.6-0.8 (07/10: 80% lỗi too_small, chữ nhỏ / nhạt hơn nháp). Trừ phần đệm đó --
    // đo bằng measureText trên chính chữ của khối, theo font của PHẦN CHỮ CHIẾM NHIỀU NHẤT (i-font / span bên trong có font, cỡ
    // riêng -- đo theo font của khung thì sai: 08/10 h02 khung Plus Jakarta, chữ Baloo dấu cao -> "tràn dọc" oan, co còn 0.6),
    // tính theo đơn vị cỡ của khung (tỉ lệ theo cỡ, đo một lần ở 100px).
    const pad = (() => {
      const t = d.textContent.trim();
      if (!t) return {top: 0, bot: 0};
      let el = d, best = 0;
      const w = document.createTreeWalker(d, NodeFilter.SHOW_TEXT);
      for (let n = w.nextNode(); n; n = w.nextNode()) {
        const k = n.textContent.trim().length;
        if (k > best) { best = k; el = n.parentElement; }
      }
      const cs = getComputedStyle(el), cx = document.createElement('canvas').getContext('2d');
      const r = parseFloat(cs.fontSize) / (parseFloat(getComputedStyle(d).fontSize) || 1);
      cx.font = `${cs.fontStyle} ${cs.fontWeight} 100px ${cs.fontFamily}`;
      const mt = cx.measureText(t);
      return {top: r * Math.max(0, mt.fontBoundingBoxAscent - mt.actualBoundingBoxAscent) / 100,
              bot: r * Math.max(0, mt.fontBoundingBoxDescent - mt.actualBoundingBoxDescent) / 100};
    })();
    let inner = d;   // phần đo nội dung (sau khi nén ngang: bên trong span nén -- hộp span cao theo line-height, đo nó là tràn dọc oan)
    const over = () => {
      const rr = document.createRange(); rr.selectNodeContents(inner);
      const q = rr.getBoundingClientRect(), b = d.getBoundingClientRect();
      const fs = parseFloat(d.style.fontSize) || 0;
      // DÒNG ĐƠN (e.hfit): chỉ kiểm tràn NGANG -- khung ô = khung OCR, không ôm dấu thanh (12/10 bỏ bước nắn khung); nền đã xoá
      // sạch nên dấu / đuôi chữ nhô khỏi khung trên dưới không đè vào đâu, co chữ vì nó là co oan
      const vy = e.hfit ? 0 : Math.max(b.top - q.top - pad.top * fs, q.bottom - b.bottom - pad.bot * fs);
      return {q, v: Math.max(0, b.left - q.left, q.right - b.right, vy)};
    };
    // ô KHÔNG có nội dung (vỏ chỉ có CSS): không đo tràn -- khung nội dung rỗng nằm ở (0,0) cho số tràn ảo (07/10 p11: 429 / 916 px)
    if (!d.textContent.trim() && !d.querySelector('svg')) {
      const b0 = d.getBoundingClientRect();
      out.push({id: e.id, size: e.size, shrink: 1, over: 0, empty: true,
                ink: [b0.left - R0.left, b0.top - R0.top, b0.right - R0.left, b0.bottom - R0.top],
                box: [b0.left - R0.left, b0.top - R0.top, b0.right - R0.left, b0.bottom - R0.top]});
      continue;
    }
    let s = e.size, k = 0, m = over(), sx = 1;
    const m0q = m.q, m0b = d.getBoundingClientRect(), fs0 = parseFloat(d.style.fontSize) || 0;   // tràn lúc CHƯA co: ngang / dọc
    let o0 = [Math.max(0, m0b.left - m0q.left, m0q.right - m0b.right),
              Math.max(0, m0b.top - m0q.top - pad.top * fs0, m0q.bottom - m0b.bottom - pad.bot * fs0)];
    // NGẮT LẠI: khối nhiều dòng (<br> chép theo dòng nháp) tràn ngang -> dòng dài tự gãy thêm, chữ mồ côi ("Khách hàng nói / gì",
    // "Tóc mềm mượt / sau": 09/10 dev3 p01, f02) -> thử bỏ <br>, trình duyệt tự ngắt; GIỮ nếu vừa khung ở nguyên cỡ
    if (o0[0] > 1 && /<br/i.test(d.innerHTML) && !d.querySelector('div,p,ul,ol,li,table,section,[style*="flex"]')) {
      const keep = d.innerHTML, ws = d.style.whiteSpace;
      d.innerHTML = keep.replace(/\s*<br\s*\/?>\s*/gi, ' ');
      d.style.whiteSpace = 'normal'; d.style.textWrap = 'pretty';
      const m2 = over();
      if (m2.v <= 1) { m = m2; o0 = [0, 0]; }
      else { d.innerHTML = keep; d.style.whiteSpace = ws; d.style.textWrap = ''; }
    }
    // NÉN NGANG trước khi co: chữ dài hơn ô (tràn ngang) -> scaleX tới o.condense, giữ cỡ chữ (chiều cao) như nháp -- co cỡ
    // làm chữ bé hẳn so với nháp (08/10 server: too_small 33 / 27 ảnh). Chỉ khối chữ dòng (không div / p / li bên trong: bọc
    // lại sẽ đổi bố cục flex của VLM); tâm nén theo căn lề của khung.
    if (((e.fit !== 'grow' && e.fit !== 'none') || e.even) && o0[0] > 1 && !d.querySelector('div,p,ul,ol,li,table,section,[style*="flex"]')) {
      sx = Math.max(e.condense || o.condense, Math.min(1, m0b.width / Math.max(1, m0q.width)));
      const j = getComputedStyle(d).justifyContent;
      const org = /start|left/.test(j) ? 'left' : /end|right/.test(j) ? 'right' : 'center';
      d.innerHTML = `<span style="display:inline-block;flex:none;white-space:inherit;transform:scaleX(${sx});` +
                    `transform-origin:${org} center">${d.innerHTML}</span>`;
      inner = d.firstElementChild;
      m = over();
    }
    if (e.fit !== 'grow' && e.fit !== 'none') {   // co nội dung cho vừa khung (em theo font-size của khung)
      while (m.v > 1 && s > o.floor && k < 80) { s *= 0.95; d.style.fontSize = s + 'px'; k++; m = over(); }
    }
    const m0 = m.v;   // tràn đo khi CHƯA xoay (sau khi xoay, khung thẳng ôm ngoài lệch so với nội dung)
    let ink0 = null;   // chữ xoay: nét mực CHƯA xoay + tâm xoay -> đa giác nét thật (hộp thẳng ôm chữ xoay chồng oan dòng kề)
    if (e.angle) {
      const q0 = m.q, b0 = d.getBoundingClientRect(), f0 = parseFloat(d.style.fontSize) || 0;
      ink0 = {r: [q0.left - R0.left, q0.top - R0.top + Math.min(pad.top * f0, q0.height / 3),
                  q0.right - R0.left, q0.bottom - R0.top - Math.min(pad.bot * f0, q0.height / 3)],
              c: [(b0.left + b0.right) / 2 - R0.left, (b0.top + b0.bottom) / 2 - R0.top]};
    }
    if (e.angle) { d.style.transformOrigin = '50% 50%'; d.style.transform = `rotate(${e.angle}deg)`; m = over(); m.v = m0; }
    const q = m.q, b = d.getBoundingClientRect();
    // NÉT MỰC = hộp chữ trừ phần đệm trên / dưới của font (dòng chồng sát nhau như FLASH / SALE thì hộp font chồng 0.2-0.3 dù
    // nét không chạm: 08/10 server 15/27 ảnh báo content_overlap oan, vòng sửa chạy vì nó). Chữ xoay: giữ hộp ngoài.
    const fsn = parseFloat(d.style.fontSize) || 0;
    const it = e.angle ? 0 : Math.min(pad.top * fsn, (q.bottom - q.top) / 3), ib = e.angle ? 0 : Math.min(pad.bot * fsn, (q.bottom - q.top) / 3);
    // MÀU CHỮ thật (màu tính được của phần tử chứa nhiều chữ nhất) + có hiệu ứng tách nền không (bóng / viền / chữ tô gradient)
    let fg = null, fx = false, best = 0, fs = null, ff = null, fw = null, fi = null, ls = null, tt = null;
    const runs = [];   // từng đoạn chữ (màu / hiệu ứng riêng): khối hai màu -- "Giảm" trắng + "40%" cam trên nền đỏ (09/10 dev2 d11)
    const tw = document.createTreeWalker(d, NodeFilter.SHOW_TEXT);
    for (let t = tw.nextNode(); t; t = tw.nextNode()) {
      const n = t.textContent.trim().length, cs = getComputedStyle(t.parentElement);
      const tfx = cs.textShadow !== 'none' || parseFloat(cs.webkitTextStrokeWidth) > 0 || cs.backgroundClip === 'text' ||
          cs.webkitBackgroundClip === 'text';
      if (tfx) fx = true;
      if (n > best) { best = n; fg = cs.webkitTextFillColor || cs.color; fs = parseFloat(cs.fontSize); ff = cs.fontFamily; fw = parseInt(cs.fontWeight);
                      fi = cs.fontStyle; ls = cs.letterSpacing; tt = cs.textTransform; }
      if (n >= 2) {
        const rg = document.createRange(); rg.selectNodeContents(t);
        const rc = rg.getBoundingClientRect();
        runs.push({n, fg: cs.webkitTextFillColor || cs.color, fx: tfx,
                   ink: [rc.left - R0.left, rc.top - R0.top, rc.right - R0.left, rc.bottom - R0.top]});
      }
    }
    out.push({id: e.id, size: s, shrink: e.size ? s / e.size : 1, over: m.v, o0, fg, fx, fs, ff, fw, fi, ls, tt, sx: +sx.toFixed(2), ink0, angle: e.angle, runs,
              ink: [q.left - R0.left, q.top - R0.top + it, q.right - R0.left, q.bottom - R0.top - ib],
              box: [b.left - R0.left, b.top - R0.top, b.right - R0.left, b.bottom - R0.top]});
  }
  return out;
}
"""


JUST = {"left": "justify-content:flex-start;text-align:left", "right": "justify-content:flex-end;text-align:right",
        "center": "justify-content:center;text-align:center"}


def _el(u: dict, e: dict, k: int, force: list, size: float | None = None, sx: float | None = None, fit: str | None = None) -> dict:
    """Phần tử của đơn vị -> tham số RENDER_JS. fix (autofix_units): màu / viền tương phản, nén sâu hơn, khung cắt."""
    from .fonts import family
    fix = u.get("fix") or {}
    h = EMOJI.sub("", e["html"])   # không font nào trong bộ font có hình emoji -> ô vuông
    if sx:
        h = (f'<span style="display:inline-block;flex:none;white-space:inherit;transform:scaleX({sx:.3f});'
             f'transform-origin:{u["align"]} center">{h}</span>')
    cls = None
    if fix.get("color"):   # chữ chìm nền -> màu tương phản (+ quầng) cưỡng cho mọi phần tử trong khối
        cls = f"fx{k}"
        force.append(f"#c>.{cls},#c>.{cls} *{{color:{fix['color']}!important;-webkit-text-fill-color:{fix['color']}!important;"
                     + (f"text-shadow:0 0 .08em {fix['halo']},0 0 .16em {fix['halo']},0 0 .3em {fix['halo']}!important;" if fix.get("halo") else "")
                     + "}")
    elif fix.get("outline"):   # chữ cỡ TIÊU ĐỀ chìm nền: GIỮ màu chữ, thêm viền + quầng màu tương phản
        cls = f"fx{k}"
        c = fix["outline"]
        ring = ",".join(f"{dx:.3f}em {dy:.3f}em 0 {c}" for dx, dy in
                        ((.04, 0), (-.04, 0), (0, .04), (0, -.04), (.028, .028), (-.028, .028), (.028, -.028), (-.028, -.028)))
        force.append(f"#c>.{cls},#c>.{cls} *{{text-shadow:{ring},0 0 .18em {c}!important;}}")
    return {"id": e["id"], "box": (fix.get("boxes") or {}).get(e["id"]) or e["box"], "html": h,
            "style": u["css"] + ";" + JUST[u["align"]] + ";" + ("" if e["path"] else "white-space:nowrap;"),
            "size": size or e["size"], "color": u.get("color") or "#161616", "font": family(u["font"]),
            "fit": fit or fix.get("fit") or "shrink", "angle": e["angle"], "cls": cls, "condense": fix.get("condense"),
            "nl": len(e["lines"]), "hfit": not e["multi"], "path": e["path"], "lead": False}


def render_units(page, plate: np.ndarray, U: list[dict], M: dict, floor: float | None = None) -> tuple[np.ndarray, list, list]:
    """Dựng các đơn vị chữ (fill.layout) lên NỀN (lonly.settle) -> (ảnh, kết quả từng phần tử {id, size, shrink, over, ink, ...}, log).
    Một dòng / một khối: nén ngang <= CONDENSE rồi co cỡ (RENDER_JS). TẬP: một mức nén + một cỡ cho mọi mục theo mục chật nhất."""
    from PIL import Image
    from .fonts import faces_for
    H, W = plate.shape[:2]
    T = [u for u in U if u["kind"] == "text" and u.get("els")]
    used = tuple(sorted({u["font"] for u in T} | {"bevietnam"}))
    fl = floor or 0.012 * min(W, H)
    log = []
    page.set_viewport_size({"width": W, "height": H})

    def run(els, force):
        page.set_content(f"<html><head><style>{faces_for(used)} *{{margin:0;padding:0;box-sizing:border-box}} #c{{position:relative;"
                         f"width:{W}px;height:{H}px;overflow:hidden;background:url('{to_data_uri(plate)}') 0 0/{W}px {H}px no-repeat}}"
                         f"{''.join(force)}</style></head><body><div id=c></div></body></html>")
        return page.evaluate(RENDER_JS, {"els": els, "floor": fl, "condense": CONDENSE})
    # TẬP: bề rộng tự nhiên của từng mục ở cỡ chung (khung giãn theo chữ) -> mục chật nhất quyết mức nén / cỡ của cả tập
    fitset = {}
    for name in sorted({u["set"] for u in T if u.get("set")}):
        us = [u for u in T if u.get("set") == name and not (u.get("fix") or {}).get("boxes")]
        if not us:
            continue
        nat = {r["id"]: r for r in run([_el(u, e, 0, [], fit="grow") for u in us for e in u["els"]], [])}
        ratio = min(1.0, min((e["box"][2] - e["box"][0]) / max(1.0, nat[e["id"]]["box"][2] - nat[e["id"]]["box"][0])
                             for u in us for e in u["els"] if e["id"] in nat))
        sx = max(CONDENSE, ratio)
        for u in us:
            fitset[u["id"]] = (ratio / sx, sx if sx < 0.999 else None)
        log.append(f"tập '{name}': mục chật nhất cần {ratio:.2f} -> cả tập nén {sx:.2f}, cỡ x{ratio / sx:.2f}")
    els, force, k = [], [], 0
    for u in T:
        for e in u["els"]:
            if u["id"] in fitset:
                kk, sx = fitset[u["id"]]
                els.append(_el(u, e, k, force, size=e["size"] * kk, sx=sx, fit="none"))
            else:
                els.append(_el(u, e, k, force))
            k += 1
    res = run(els, force)
    size0 = {e["id"]: e["size"] for u in T for e in u["els"]}
    for r in res:
        uid = r["id"].split("#")[0]
        r["unit"] = uid
        if uid in fitset:   # co theo tập: chỉ mục chật nhất là "phải co" -- too_small không tính các mục kéo theo
            r["shrink"], r["sx"], r["evened"] = fitset[uid][0], fitset[uid][1] or 1, True
        elif size0.get(r["id"]):
            r["shrink"] = r["size"] / size0[r["id"]]
        if r["shrink"] < 0.7:
            log.append(f"{r['id']}: co còn {r['shrink']:.2f} để vừa khung")
    img = np.asarray(Image.open(io.BytesIO(page.locator("#c").screenshot())).convert("RGB"))
    # NỀN THẬT sau chữ (chữ trong suốt) -> tỉ lệ tương phản chữ / nền (check_units: low_contrast)
    page.evaluate("""() => { for (const x of document.querySelectorAll('#c *')) { const s = x.style;
        s.setProperty('color', 'transparent', 'important'); s.setProperty('-webkit-text-fill-color', 'transparent', 'important');
        s.setProperty('text-shadow', 'none', 'important'); s.setProperty('-webkit-text-stroke-width', '0', 'important'); } }""")
    bg = np.asarray(Image.open(io.BytesIO(page.locator("#c").screenshot())).convert("RGB"))
    color0 = {e["id"]: u.get("color") for u in T for e in u["els"]}
    bad = lambda r: (r["low"] or 0) > (0.6 if r.get("px_based") else CONTRAST_FRAC)   # noqa: E731
    for r in res:
        r["low"] = _low_contrast(r, bg, img)
        if len(r.get("runs") or []) > 1 and not bad(r):   # khối nhiều màu: một đoạn chìm là lỗi dù cả khối trung bình đọc được
            for q in r["runs"]:
                q["low"] = _low_contrast(q, bg, img)
                if bad(q):
                    r["low"], r["px_based"] = q["low"], q.get("px_based", False)
                    break
    pal = list(dict.fromkeys(r["fg"] for r in res if r.get("fg") and not bad(r)))   # màu chữ đọc tốt trên poster
    fs_med = float(np.median([r["fs"] for r in res if r.get("fs")] or [0]))
    for r in res:
        if bad(r):
            r["recolor"] = _recolor(r, bg, color0.get(r["id"]), pal, big=bool(r.get("fs") and (r["fs"] >= OUTLINE_HEAD * fs_med or r["fs"] >= OUTLINE_ABS * min(W, H))))
    return img, res, log


OUTLINE_HEAD = 1.4   # chữ cỡ >= 1.4 lần cỡ trung vị poster (tiêu đề) chìm nền: giữ màu + viền thay vì đổi màu ...
OUTLINE_ABS = 0.06   # ... hoặc cỡ >= 6% cạnh ngắn ảnh (poster nhiều chữ to: trung vị cao -- dev3 g12 114 / 95 px)
CONTRAST_PX = 2.0   # điểm nền mà màu chữ trên đó có tỉ lệ tương phản WCAG < 2 = chữ chìm (chữ thân cần >= 4.5, chữ lớn >= 3)
CONTRAST_FRAC = 0.25   # > 25% nền sau khối chữ là điểm chìm = lỗi low_contrast (khối chữ không có bóng / viền)


def _lum(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def _low_contrast(r: dict, bg: np.ndarray, img: np.ndarray | None = None) -> float | None:
    """Phần nền (trong khung nét chữ của lệnh) mà màu chữ của lệnh chìm trên đó. Chữ có bóng / viền / tô gradient (màu chữ
    không đo được một màu, 08/10 h08_s1: tiêu đề vàng nhạt chìm hẳn vào nền hồng nhạt mà không bị bắt): đo trên ĐIỂM ẢNH THẬT
    -- điểm khác nền (bản dựng so với nền không chữ) là nét chữ; phần nét chữ có tương phản < 1.5 với nền ngay dưới nó."""
    if r.get("empty"):
        return None
    H, W = bg.shape[:2]
    x0, y0, x1, y1 = r["ink"]
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    m = re.match(r"rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?", r.get("fg") or "")
    if r.get("fx") or not m or (m.group(4) is not None and float(m.group(4)) < 0.5):
        if img is None:
            return None
        a, b = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32), bg[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        ink = np.abs(a - b).max(1) > 30
        if ink.sum() < 20:
            return None
        la, lb = _lum(a[ink]), _lum(b[ink])
        ratio = (np.maximum(la, lb) + 0.05) / (np.minimum(la, lb) + 0.05)
        r["px_based"] = True
        return round(float((ratio < 1.5).mean()), 2)
    lf = float(_lum(np.array([[float(m.group(i)) for i in (1, 2, 3)]])[0]))
    lb = _lum(bg[y0:y1, x0:x1].reshape(-1, 3))
    ratio = (np.maximum(lf, lb) + 0.05) / (np.minimum(lf, lb) + 0.05)
    return round(float((ratio < CONTRAST_PX).mean()), 2)


def _rgb(c: str | None) -> tuple | None:
    if not c:
        return None
    m = re.match(r"#([0-9a-f]{6})$", c.strip(), flags=re.I)
    if m:
        return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    m = re.match(r"rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)", c)
    return tuple(float(m.group(i)) for i in (1, 2, 3)) if m else None


def _recolor(r: dict, bg: np.ndarray, draft_color: str | None, pal: list | None = None, big: bool = False) -> dict | None:
    """Chữ chìm nền: màu chữ thay thế, ưu tiên HỢP BẢNG MÀU (08/10 replay: tiêu đề vàng nhạt -> gần đen, đọc được nhưng thô) --
    màu các chữ đọc tốt khác trên poster, rồi màu nháp tối / sáng dần, cuối cùng trắng / gần đen: màu ĐẦU TIÊN có <= 10% nền
    chìm (tương phản WCAG < CONTRAST_PX). Không màu nào đạt: màu ít chìm nhất + quầng màu ngược độ sáng."""
    H, W = bg.shape[:2]
    x0, y0, x1, y1 = r["ink"]
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    lb = _lum(bg[y0:y1, x0:x1].reshape(-1, 3))
    if big and _rgb(r.get("fg")):
        # TIÊU ĐỀ: đổi hẳn màu làm mất thiết kế (09/10 dev3 g11: cam viền trắng -> nâu ô liu, g12: trắng -> xám) -> giữ màu chữ,
        # viền màu ngược độ sáng của chữ (chữ sáng viền tối, chữ tối viền trắng) -- như nháp hay vẽ chữ có viền
        lf = float(_lum(np.array([_rgb(r["fg"])], np.float32))[0])
        return {"outline": "#161616" if lf > 0.35 else "#ffffff", "frac": None}
    cands = list(pal or [])   # màu chữ khác của chính poster trước: hợp bảng màu (màu nháp pha đen ra xám đục -- 08/10 h08_s1)
    d = _rgb(draft_color) or _rgb(r.get("fg"))
    if d:
        for t in (0.35, 0.6, 0.8):
            for to in (0, 255):
                cands.append("#%02x%02x%02x" % tuple(int(round(v + (to - v) * t)) for v in d))
    cands += ["#ffffff", "#161616"]
    best = None
    for c in dict.fromkeys(cands):
        rgb = _rgb(c)
        if rgb is None:
            continue
        lf = float(_lum(np.array([rgb], np.float32))[0])
        f = float(((np.maximum(lf, lb) + 0.05) / (np.minimum(lf, lb) + 0.05) < CONTRAST_PX).mean())
        if f <= 0.1:
            best = (c, f, lf)
            break
        if best is None or f < best[1] - 0.02:
            best = (c, f, lf)
    c, f, lf = best
    return {"color": c, "frac": round(f, 2), "halo": ("rgba(0,0,0,.55)" if lf > 0.4 else "rgba(255,255,255,.6)")
            if f > CONTRAST_FRAC else None}


def _ov(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    s = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / s if w > 0 and h > 0 and s > 0 else 0.0


def _ink_poly(r: dict) -> list:
    """Đa giác nét mực của kết quả dựng: chữ xoay = nét chưa xoay xoay quanh tâm khung; chữ thẳng = khung nét."""
    i0 = r.get("ink0")
    if not i0 or not r.get("angle"):
        x0, y0, x1, y1 = r["ink"]
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    t = np.radians(r["angle"])
    c, sn = np.cos(t), np.sin(t)
    (x0, y0, x1, y1), (cx, cy) = i0["r"], i0["c"]
    return [[cx + (x - cx) * c - (y - cy) * sn, cy + (x - cx) * sn + (y - cy) * c] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


def _poly_ov(a, b) -> float:
    """Phần giao hai đa giác lồi (khung có thể nghiêng) / diện tích đa giác nhỏ hơn."""
    import cv2
    pa, pb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    s = min(cv2.contourArea(pa), cv2.contourArea(pb))
    if s <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    return float(inter) / s


SEVERE = ("empty_plan", "content_overlap", "overflow", "low_contrast")


def check_units(U: list[dict], res: list | None, shrink_min: float | None = 0.7) -> list[dict]:
    """LỖI KHÁCH QUAN sau khi dựng (không cần mắt): nội dung hai đơn vị chồng nhau, tràn khung dù đã co tới sàn, phải co dưới
    shrink_min (chữ dài so với chỗ nháp vẽ; mục của tập kéo theo không tính), chữ chìm nền."""
    errs = []
    fixed = {u["id"]: u.get("fix") or {} for u in U}
    res = res or []
    for i, a in enumerate(res):
        for b in res[i + 1:]:
            if a.get("empty") or b.get("empty") or a["unit"] == b["unit"]:
                continue
            if a.get("polys") or b.get("polys"):
                o = _multi_ov(a.get("polys") or [_ink_poly(a)], b.get("polys") or [_ink_poly(b)])
            else:
                o = _poly_ov(_ink_poly(a), _ink_poly(b)) if a.get("angle") or b.get("angle") else _ov(a["ink"], b["ink"])
            if o > OVERLAP:
                errs.append({"type": "content_overlap", "ops": [a["id"], b["id"]], "frac": round(o, 2)})
        if a["over"] > 1:
            errs.append({"type": "overflow", "ops": [a["id"]], "px": round(a["over"])})
        elif shrink_min and a.get("shrink", 1) < shrink_min and not a.get("evened"):
            errs.append({"type": "too_small", "ops": [a["id"]], "shrink": round(a["shrink"], 2)})
        if (a.get("low") or 0) > (0.6 if a.get("px_based") else CONTRAST_FRAC) and not fixed.get(a["unit"], {}).get("outline"):
            errs.append({"type": "low_contrast", "ops": [a["id"]], "frac": a["low"], "color": a.get("fg")})
    return errs


FIX_ROUNDS = 2   # sửa bằng code: sửa -> dựng -> kiểm, tối đa 2 lượt (lượt 2 sửa lỗi lượt 1 làm lộ ra)


def _trim(A: list, B: list) -> list | None:
    """Khung A cắt bớt MỘT phía để không còn giao B -- phía giữ được nhiều diện tích A nhất; None khi cắt mất quá nửa."""
    w, h = A[2] - A[0], A[3] - A[1]
    c = [[A[0], A[1], min(A[2], B[0]), A[3]], [max(A[0], B[2]), A[1], A[2], A[3]],
         [A[0], A[1], A[2], min(A[3], B[1])], [A[0], max(A[1], B[3]), A[2], A[3]]]
    c = [x for x in c if x[2] - x[0] > 2 and x[3] - x[1] > 2]
    if not c or w <= 0 or h <= 0:
        return None
    best = max(c, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
    return best if (best[2] - best[0]) * (best[3] - best[1]) >= 0.5 * w * h else None


def autofix_units(U: list[dict], errs: list[dict], res: list[dict]) -> tuple[list[dict], list[str]]:
    """SỬA LỖI NẶNG BẰNG CODE: chữ và kiểu giữ nguyên, chỉ sửa phần đo được.
      low_contrast    -- màu tương phản nhất với nền thật ({màu nháp, trắng, gần đen}), nền loang thêm quầng; tiêu đề: viền
      overflow        -- nén ngang sâu hơn (0.75)
      content_overlap -- phần tử thò khỏi khung nhiều hơn (hoặc khung to hơn) cắt khung một phía cho hết giao, co trong khung mới;
                         phần tử xoay / theo đường cong không cắt.
    -> (đơn vị mới, log). Không lỗi nào sửa được: y nguyên."""
    import copy
    U2 = copy.deepcopy(U)
    by = {u["id"]: u for u in U2}
    R = {r["id"]: r for r in res}
    el = {e["id"]: (u, e) for u in U2 if u["kind"] == "text" for e in u.get("els") or []}
    log = []
    fx = lambda i: by[i.split("#")[0]].setdefault("fix", {})   # noqa: E731
    for e in errs:
        t, ids = e["type"], [i for i in e.get("ops") or [] if i in el]
        if t == "low_contrast" and ids and (R.get(ids[0]) or {}).get("recolor"):
            rc = R[ids[0]]["recolor"]
            if rc.get("outline"):
                fx(ids[0]).update(outline=rc["outline"])
                log.append(f"{ids[0]}: chìm nền {e['frac']} -> giữ màu, viền {rc['outline']}")
            else:
                fx(ids[0]).update(color=rc["color"], halo=rc["halo"])
                log.append(f"{ids[0]}: chìm nền {e['frac']} -> màu {rc['color']}" + (" + quầng" if rc["halo"] else ""))
        elif t == "overflow" and ids and not el[ids[0]][0].get("set"):
            fx(ids[0])["condense"] = 0.75
            log.append(f"{ids[0]}: tràn {e['px']}px -> nén ngang tới 0.75")
        elif t == "content_overlap" and len(ids) == 2:
            cand = [i for i in ids if not el[i][1]["path"] and not el[i][1]["angle"]]
            if not cand:
                continue

            def spill(i):   # phần nét chữ thò ra ngoài khung của chính phần tử
                r, b = R.get(i), el[i][1]["box"]
                if not r:
                    return 0.0
                a = r["ink"]
                area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
                return 1 - _ov(a, b) * min(area, (b[2] - b[0]) * (b[3] - b[1])) / area
            area = lambda i: (el[i][1]["box"][2] - el[i][1]["box"][0]) * (el[i][1]["box"][3] - el[i][1]["box"][1])   # noqa: E731
            i = max(cand, key=lambda i: (round(spill(i), 2), area(i)))
            j = ids[1] if i == ids[0] else ids[0]
            nb = _trim(el[i][1]["box"], R[j]["ink"] if R.get(j) else el[j][1]["box"])
            if nb is None:
                log.append(f"{i}: chồng {j} nhưng cắt khung mất quá nửa -> để nguyên")
                continue
            f = fx(i)
            f.setdefault("boxes", {})[i] = [round(v, 1) for v in nb]
            f["fit"] = "shrink"
            log.append(f"{i}: chồng {j} ({e.get('frac')}) -> cắt khung còn {[round(v) for v in nb]}")
    return U2, log


KIND_COL = {"text": (230, 30, 30), "keep": (0, 160, 80), "skip": (170, 170, 170)}


SET_COLS = [(30, 110, 230), (200, 120, 0), (150, 40, 200), (0, 150, 150), (200, 40, 120), (90, 130, 20)]


def ops_image(img, M, P):
    """Nháp mờ + khung từng đơn vị chữ (fill.public) + nhãn: vai / tập và chữ sẽ viết. Màu: mỗi TẬP cùng vai một màu (các mục
    cùng màu là được dựng giống hệt nhau); đơn vị lẻ đỏ; giữ xanh lá; xoá xám."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray((img * 0.55 + 115).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("arial.ttf", max(11, img.shape[1] // 90))
    except OSError:
        f = ImageFont.load_default()
    sets = sorted({op["set"] for op in P["ops"] if op.get("set")})
    for op in P["ops"]:
        b = op_box(op, M)
        if not b:
            continue
        col = SET_COLS[sets.index(op["set"]) % len(SET_COLS)] if op.get("set") else KIND_COL.get(op.get("kind"), (0, 0, 0))
        b = [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])]
        d.rectangle(b, outline=col, width=5 if op.get("set") else 3)
        tag = op.get("set") or op.get("role") or op.get("kind")
        lab = f"{tag}: {plain(op.get('html', ''))[:40]}"
        d.text((b[0] + 3, b[1] + 2), lab, fill=col, font=f)
    return im
