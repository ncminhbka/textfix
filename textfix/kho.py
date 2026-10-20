"""KHO ẢNH THAM CHIẾU: ảnh mẫu cho món / trang phục / vật Tết riêng của Việt Nam (bún thang, áo dài, mâm ngũ quả...) để FLUX
vẽ đúng (không có ảnh, "bún thang" thành phở bò).

  <kho>/index.json   mục lục: mỗi mục {id, file, tags, aliases, source}
      tags     tên tiếng Anh của món (khớp cả khi người dùng viết tiếng Anh)
      aliases  cụm tiếng Việt khách có thể gõ, có hay không dấu đều được
  <kho>/<file>       ảnh mẫu (JPG / PNG), MỘT ảnh mỗi khái niệm. Chỉ ảnh có giấy phép rõ ràng (ghi ở "source")
  <kho> = biến môi trường TEXTFIX_KHO_DIR, mặc định output/kho (ngoài git): thư mục đệm máy chủ đồng bộ từ kho đối tượng
          (kho_sync.py). Ảnh không vào git; cách dựng và đẩy kho: docs/KHO.md

  match_detail -> ỨNG VIÊN: khớp ĐÚNG TỪ KHOÁ (nguyên từ), không embedding: "phở" không kéo ảnh phở bò. Nhiều món: theo thứ
           tự xuất hiện. Từ khách gõ CÓ DẤU phải trùng cả dấu ("bánh Tết" không phải "bánh tét", "cơm" không phải "cốm"); từ gõ
           KHÔNG DẤU so với dạng bỏ dấu của cụm ("banh tet" khớp "bánh tét").
           Mỗi ứng viên có độ CHẮC (20/10): cụm >= 2 từ khách gõ có dấu, trùng nguyên văn kể cả dấu = chắc, code dùng luôn;
           còn lại (alias một từ "cốm" trong "trà sữa cốm", gõ không dấu "com tam" / "banh tet") LLM đầu luồng quyết
           (brief/__init__.py)
  numbered -> số ảnh của từng mục kho trong lượt vẽ: ảnh sản phẩm khách tải (nếu có) luôn là image 1
  refs_for_draw -> ảnh sản phẩm trước rồi ảnh kho; FLUX.2 klein (4B và 9B) nhận tối đa MAX_REFS ảnh
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_REFS = 4   # FLUX.2 klein 4B / 9B nhận tối đa 4 ảnh tham chiếu, kể cả ảnh sản phẩm khách tải


def kho_dir() -> Path:
    return Path(os.environ.get("TEXTFIX_KHO_DIR") or ROOT / "output" / "kho")


def _norm(s: str) -> str:
    """Dạng so khớp: chữ thường, bỏ dấu, ký tự khác thành khoảng trắng, đệm hai đầu -> khớp nguyên từ."""
    s = unicodedata.normalize("NFD", s.casefold().replace("đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " " + re.sub(r"[^a-z0-9]+", " ", s).strip() + " "


def load(root: Path | None = None) -> list[dict]:
    """Mục lục của kho; không có hoặc hỏng -> kho rỗng (vẽ như không có kho)."""
    path = (root or kho_dir()) / "index.json"
    try:
        entries = json.loads(path.read_text(encoding="utf-8")).get("entries", [])
    except (OSError, ValueError, AttributeError):
        return []
    return [e for e in entries if isinstance(e, dict)]


def _terms(e: dict) -> list[str]:
    return [t for t in [*(e.get("aliases") or []), *(e.get("tags") or [])] if str(t).strip()]


def problems(entries: list[dict], root: Path | None = None) -> list[str]:
    """Lỗi của mục lục (để kiểm trước khi đóng gói): thiếu trường, file không có, trùng id, một cụm trỏ hai mục."""
    root = root or kho_dir()
    out, ids, owner = [], set(), {}
    for i, e in enumerate(entries):
        name = e.get("id") or f"mục {i + 1}"
        if not e.get("id") or not e.get("file"):
            out.append(f"{name}: thiếu id hoặc file")
            continue
        if e["id"] in ids:
            out.append(f"{name}: trùng id")
        ids.add(e["id"])
        if not (root / e["file"]).is_file():
            out.append(f"{name}: không có file {e['file']}")
        if not str(e.get("source") or "").strip():
            out.append(f"{name}: thiếu source (giấy phép / tác giả)")
        if not _terms(e):
            out.append(f"{name}: chưa có aliases / tags, không bao giờ được khớp")
        elif all(len(_words(t)) < SURE_WORDS for t in _terms(e)):
            out.append(f"{name}: chỉ có cụm một từ -- không bao giờ khớp chắc (LLM quyết từng lần); thêm cụm >= 2 từ "
                       "như 'cốm làng vòng'")
        for t in _terms(e):
            k = _norm(t).strip()
            if k in owner and owner[k] != e["id"]:
                out.append(f"cụm '{t}' trỏ vào hai mục: {owner[k]} và {e['id']}")
            owner.setdefault(k, e["id"])
    return out


def _words(s: str) -> list[str]:
    """Từ của s: chữ thường, GIỮ dấu (NFC), tách ở mọi ký tự không phải chữ / số."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", s.casefold()))


def _word_match(user: str, term: str) -> bool:
    """Một từ khách gõ khớp một từ của cụm: có dấu thì phải trùng hẳn; không dấu thì so với dạng bỏ dấu của cụm."""
    return user == term if not user.isascii() else user == _norm(term).strip()


SURE_WORDS = 2   # khớp CHẮC: cụm >= 2 từ, khách gõ CÓ DẤU, mọi từ trùng nguyên văn kể cả dấu ("bún thang", "cốm làng Vòng")


def _first_hit(e: dict, words: list[str]) -> dict | None:
    """Lần khớp của mục trong lời khách: {pos: chỉ số từ sớm nhất, sure, word: chữ khách gõ ở chỗ khớp}; None nếu không khớp.
    Có cụm khớp chắc thì mục là chắc (vị trí vẫn là lần khớp sớm nhất)."""
    best, sure = None, False
    for term in _terms(e):
        tw = _words(term)
        for i in range(len(words) - len(tw) + 1):
            if all(_word_match(words[i + k], w) for k, w in enumerate(tw)):
                # chắc = trùng nguyên văn VÀ khách có gõ dấu: "banh tet" toàn ASCII vẫn mơ hồ (bánh Tết / bánh tét)
                got = words[i:i + len(tw)]
                exact = len(tw) >= SURE_WORDS and got == tw and not "".join(got).isascii()
                sure = sure or exact
                if best is None or i < best["pos"]:
                    best = {"pos": i, "word": " ".join(words[i:i + len(tw)])}
                break
    return None if best is None else {**best, "sure": sure}


def match_detail(text: str, entries: list[dict] | None = None, root: Path | None = None) -> list[dict]:
    """Ứng viên kho trong text, theo THỨ TỰ XUẤT HIỆN: [{id, pos, sure, word}]. Bỏ mục không có file ảnh; cắt ở MAX_REFS.
    sure = khớp chắc (code dùng luôn); không chắc (alias một từ "cốm" trong "trà sữa cốm", gõ không dấu "com tam") thì LLM
    đầu luồng quyết (brief: nhắc "image N" thì dùng, không nhắc thì bỏ)."""
    root = root or kho_dir()
    entries = load(root) if entries is None else entries
    words = _words(text or "")
    hits = [(h["pos"], i, {"id": e["id"], **h}) for i, e in enumerate(entries) if e.get("id") and e.get("file")
            for h in [_first_hit(e, words)] if h is not None and (root / e["file"]).is_file()]
    return [h for _, _, h in sorted(hits, key=lambda x: x[:2])][:MAX_REFS]


def match(text: str, entries: list[dict] | None = None, root: Path | None = None) -> list[str]:
    """Id các ứng viên kho trong text (match_detail), theo thứ tự xuất hiện."""
    return [h["id"] for h in match_detail(text, entries, root)]


def numbered(ids: list[str], product: bool, entries: list[dict] | None = None) -> list[tuple[int, dict]]:
    """(số ảnh tham chiếu, mục) cho các mục kho sẽ được vẽ cùng: ảnh sản phẩm khách là image 1, kho tiếp theo; cắt ở MAX_REFS."""
    entries = load() if entries is None else entries
    by_id = {e["id"]: e for e in entries if e.get("id")}
    first = 2 if product else 1
    keep = [by_id[i] for i in ids if i in by_id][:MAX_REFS - first + 1]
    return [(first + k, e) for k, e in enumerate(keep)]


def name_of(e: dict) -> str:
    """Tên gọi món trong prompt tiếng Anh: tag đầu (tiếng Anh), không có thì alias đầu."""
    return str((e.get("tags") or e.get("aliases") or [e.get("id")])[0])


def images(ids: list[str], entries: list[dict] | None = None, root: Path | None = None) -> list[np.ndarray]:
    from PIL import Image
    root = root or kho_dir()
    entries = load(root) if entries is None else entries
    by_id = {e["id"]: e for e in entries if e.get("id")}
    return [np.asarray(Image.open(root / by_id[i]["file"]).convert("RGB")) for i in ids if i in by_id]


def refs_for_draw(B: dict, product: np.ndarray | None, root: Path | None = None) -> list[np.ndarray]:
    """Ảnh tham chiếu của lượt vẽ: ảnh sản phẩm khách tải trước (luôn image 1), phần còn lại cho ảnh kho theo B["kho"].
    Ảnh kho hỏng / mất thì bỏ qua: vẽ thiếu tham chiếu còn hơn hỏng cả lượt."""
    refs = [product] if product is not None else []
    out = []
    for cid in (B.get("kho") or [])[:MAX_REFS - len(refs)]:
        try:
            out += images([cid], root=root)
        except Exception:
            pass
    return refs + out
