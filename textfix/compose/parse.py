"""Câu trả lời VLM -> đơn vị thô {kind, marks, clients, split, text_vlm, role_vlm, why}; mục sai khuôn bị bỏ.

Hai khuôn: "lines" (16/10: mỗi L# một mục, khuôn JSON bắt buộc đủ dòng -- compose.vlm.schema) và "assign" (mỗi mục một nhóm
dòng: plan.json cũ, scripts/replay.py, scripts/golden.py).
"""

from __future__ import annotations

from .call import _obj, _tid
from .geometry import SPLIT_RATIO, _reading, _stacked


def parse(raw, M: dict | None = None) -> dict:
    """Câu trả lời VLM -> {"assign": [đơn vị thô], "styles", "accent", "missing", "notes"}. Mục sai khuôn bị bỏ, ghi "dropped".
    M: các dòng của poster (khuôn "lines" cần hình học để gộp các mẩu của một câu)."""
    P = _obj(raw)
    bad = []
    if isinstance(P.get("lines"), dict):
        entries, bad = from_lines(P["lines"], M)
    else:
        entries = P.get("assign") or []
    out = []
    for i, a in enumerate(entries):
        a = _obj(a) if isinstance(a, str) else a
        if not isinstance(a, dict):
            bad.append(str(a)[:80])
            continue
        ln = a.get("lines") or a.get("slots") or []
        ln = [x for x in (ln if isinstance(ln, list) else [ln]) if isinstance(x, str)]
        cl = a.get("client")
        cl = cl if isinstance(cl, list) else [] if cl is None else [cl]
        sp = a.get("split")
        sp = [str(x).strip() for x in sp] if isinstance(sp, list) and sp and all(isinstance(x, (str, int, float)) for x in sp) else None
        out.append({"id": f"u{i + 1}", "kind": a.get("kind") if a.get("kind") in ("keep", "skip") else "text", "marks": ln,
                    "clients": [k for k in (_tid(c) for c in cl) if k is not None], "split": sp,
                    "text_vlm": str(a.get("text") or "").strip(), "role_vlm": str(a.get("role") or "").strip() or None,
                    "why": str(a.get("why") or "")[:160]})
    st = P.get("styles") if isinstance(P.get("styles"), dict) else {}
    return {"assign": out, "styles": {str(k): v for k, v in st.items() if isinstance(v, dict)},
            "accent": [x for x in P.get("accent") or [] if isinstance(x, dict) and str(x.get("text") or "").strip()],
            "missing": [k for k in (_tid(x) for x in P.get("missing") or []) if k is not None],
            "notes": P.get("notes"), **({"dropped": bad} if bad else {})}


def from_lines(lines: dict, M: dict | None) -> tuple[list[dict], list[str]]:
    """Khuôn "lines" (mỗi L# một mục) -> mục ghép kiểu "assign". Các dòng VLM ghi CÙNG câu khách kèm "part" (mẩu câu dòng đó
    mang) là MỘT câu chia nhiều dòng -> một mục, "split" = các mẩu theo thứ tự đọc. Các dòng đó xếp chồng thành khối, cùng cỡ
    (lệch <= SPLIT_RATIO), không cong / nghiêng: bỏ split -- code tự ngắt câu theo sức chứa (compose.layout), như khuôn cũ chỉ ghi
    split cho dòng khác cỡ / ở chỗ khác nhau. Câu ghép cho hai dòng mà không kèm part: hai mục (compose.assign xử câu ghép
    hai lần). -> (mục, mục sai khuôn)."""
    by = (M or {}).get("by") or {}
    ok, bad = {}, []
    for lid, e in lines.items():
        if isinstance(e, dict):
            ok[str(lid)] = e
        else:
            bad.append(f"{lid}: {str(e)[:60]}")
    order = sorted(ok, key=lambda i: (int(i[1:]) if i[1:].isdigit() else 1 << 30, i))
    clients = lambda e: tuple(e.get("client") or []) if isinstance(e.get("client"), list) else (e.get("client"),) if e.get("client") else ()  # noqa: E731
    part = lambda e: str(e.get("part") or "").strip()  # noqa: E731
    groups = {}
    for i in order:
        e = ok[i]
        if e.get("kind", "text") == "text" and clients(e) and part(e):
            groups.setdefault(clients(e), []).append(i)
    out, done = [], set()
    for i in order:
        if i in done:
            continue
        e = ok[i]
        g = groups.get(clients(e)) if e.get("kind", "text") == "text" and clients(e) and part(e) else None
        if g and len(g) >= 2:
            if all(m in by for m in g):
                g = [g[k] for k in _reading([by[m]["box"] for m in g])]
            split = [part(ok[m]) for m in g]
            Ls = [by[m] for m in g if m in by]
            if len(Ls) == len(g) and _block(Ls):
                split = None
            out.append({"lines": g, "client": list(clients(e)), "kind": "text", "split": split, "text": None,
                        "role": e.get("role"), "why": e.get("why")})
            done.update(g)
            continue
        out.append({"lines": [i], "client": list(clients(e)), "kind": e.get("kind"), "split": None, "text": e.get("text"),
                    "role": e.get("role"), "why": e.get("why")})
        done.add(i)
    return out, bad


def _block(Ls: list[dict]) -> bool:
    """Các dòng (thứ tự đọc) là một khối chữ thường: thẳng, xếp chồng, cùng cỡ."""
    if any(l.get("path") or abs(l.get("angle") or 0) >= 2 or l.get("vertical") for l in Ls):
        return False
    szs = [l["size"] for l in Ls]
    return _stacked(Ls) and max(szs) / max(1e-6, min(szs)) <= SPLIT_RATIO
