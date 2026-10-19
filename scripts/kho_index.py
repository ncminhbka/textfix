"""DỰNG KHO tham chiếu từ ảnh ứng viên đã chọn: ảnh (cạnh dài 1024, JPEG) + index.json, kiểm bằng kho.problems().

  python scripts/kho_index.py <thư mục tạm> <picks.json> <thư mục kho ra>

picks.json: một ảnh mỗi khái niệm (khái niệm không có ảnh đạt thì không ghi id đó)
  {"<id>": <số ô trong contact sheet>, ...}                                  ảnh ứng viên của kho_fetch.py
  {"<id>": {"file": "<đường dẫn ảnh>", "source": "<giấy phép · tác giả · link>"}}   ảnh người dùng tự tìm / chụp
aliases / tags lấy từ CONCEPTS của scripts/kho_fetch.py (chỉ ghi bản có dấu: kho.match so khớp đã bỏ dấu, bản không dấu tự khớp).
source = giấy phép · tác giả · link trang ảnh. Mỗi mục ghi sha1 của ảnh để máy chủ khỏi tải lại ảnh không đổi.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kho_fetch import CONCEPTS  # noqa: E402

from textfix import kho  # noqa: E402
from textfix.kho_sync import sha1  # noqa: E402

LONG = 1024


def jpeg(src: Path) -> bytes:
    im = Image.open(src).convert("RGB")
    im.thumbnail((LONG, LONG), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88, optimize=True)
    return buf.getvalue()


def entry(cid: str, meta: dict, data: bytes) -> dict:
    _, aliases, tags = CONCEPTS[cid]
    src = " · ".join(x for x in (meta.get("license"), meta.get("artist"), meta.get("page")) if x)
    return {"id": cid, "file": f"img/{cid}.jpg", "tags": tags, "aliases": aliases, "source": meta.get("source") or src,
            "sha1": sha1(data)}


def build(staging: Path, picks: dict[str, int | dict], out: Path) -> list[dict]:
    (out / "img").mkdir(parents=True, exist_ok=True)
    entries = []
    for cid, k in picks.items():
        if isinstance(k, dict):   # ảnh người dùng tự tìm / chụp
            meta, src = {"source": k.get("source", "")}, Path(k["file"])
        else:
            meta = json.loads((staging / cid / "meta.json").read_text(encoding="utf-8"))[k]
            src = staging / cid / meta["file"]
        data = jpeg(src)
        (out / "img" / f"{cid}.jpg").write_bytes(data)
        entries.append(entry(cid, meta, data))
    doc = {"_huong_dan": "docs/KHO.md. Mỗi mục: id, file, tags (tiếng Anh), aliases (cụm tiếng Việt), source, sha1.",
           "entries": entries}
    (out / "index.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return entries


if __name__ == "__main__":
    staging, picks, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")), Path(sys.argv[3])
    entries = build(staging, picks, out)
    bad = kho.problems(entries, out)
    size = sum(p.stat().st_size for p in (out / "img").iterdir())
    print(f"{len(entries)} mục, ảnh {size / 1e6:.1f} MB ->", out)
    print("\n".join(bad) or "mục lục ổn")
    sys.exit(1 if bad else 0)
