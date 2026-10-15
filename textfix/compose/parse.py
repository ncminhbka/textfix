"""Câu trả lời VLM -> đơn vị thô {kind, marks, clients, split, text_vlm, role_vlm, why}; mục sai khuôn bị bỏ.
"""

from __future__ import annotations

from .call import _obj, _tid


def parse(raw) -> dict:
    """Câu trả lời VLM -> {"assign": [đơn vị thô], "styles", "accent", "missing", "notes"}. Mục sai khuôn bị bỏ, ghi "dropped"."""
    P = _obj(raw)
    out, bad = [], []
    for i, a in enumerate(P.get("assign") or []):
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
