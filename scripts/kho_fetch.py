"""Tải ẢNH ỨNG VIÊN cho kho tham chiếu từ Pixabay (Pixabay Content License: dùng tự do, không bắt ghi nguồn -- vẫn ghi).

  PIXABAY_KEY=<khoá> python scripts/kho_fetch.py <thư mục tạm> [id ...]     # mặc định: mọi khái niệm trong CONCEPTS

Mỗi khái niệm: <thư mục tạm>/<id>/<n>.jpg (cạnh dài 1280) + meta.json (link trang, tác giả, giấy phép). Khoá API chỉ đọc từ biến
môi trường, không ghi vào repo. Ảnh tạm KHÔNG vào git: chọn MỘT ảnh mỗi khái niệm (scripts/kho_sheet.py xem, kho_index.py dựng),
khái niệm không có ảnh đạt thì người dùng tự tìm / chụp. Pixabay cấm dùng link ảnh lâu dài (hotlink): tải về rồi lưu lên kho
đối tượng như ở đây là đúng luật của họ. Chạy ở máy có internet (laptop); máy chủ chỉ đọc kho trong mạng nội bộ.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://pixabay.com/api/"
UA = "TextFixKho/0.2 (reference images for poster generation research)"
PER = 8                # số ứng viên giữ lại mỗi khái niệm (để chọn MỘT)
LICENSE = "Pixabay Content License"

# id -> (truy vấn Pixabay, aliases tiếng Việt, tags tiếng Anh). Khái niệm model gần như chắc chắn KHÔNG biết (không phở, bánh mì).
# Khớp đúng từ khoá (không embedding) nên cụm phải đủ riêng: không "bò", "cơm", "phở", "đèn lồng" trơn.
# Tag đầu là tên LLM gọi món trong prompt ("the vietnamese dog meat from image 2").
CONCEPTS: dict[str, tuple[list[str], list[str], list[str]]] = {
    "thit-cho":      (["dog meat vietnam", "thịt chó"], ["thịt chó", "thịt cầy", "rựa mận"], ["vietnamese dog meat"]),
    "mam-tom":       (["shrimp paste vietnam", "mắm tôm"], ["mắm tôm"], ["vietnamese fermented shrimp paste"]),
    "bun-dau":       (["bun dau mam tom", "bún đậu"], ["bún đậu mắm tôm", "bún đậu"], ["bun dau mam tom"]),
    "bun-thang":     (["bun thang", "bún thang"], ["bún thang"], ["bun thang"]),
    "banh-duc":      (["banh duc", "bánh đúc"], ["bánh đúc"], ["banh duc"]),
    "banh-beo":      (["banh beo", "bánh bèo"], ["bánh bèo"], ["banh beo"]),
    "banh-khot":     (["banh khot", "bánh khọt"], ["bánh khọt"], ["banh khot"]),
    "banh-bot-loc":  (["banh bot loc", "bánh bột lọc"], ["bánh bột lọc"], ["banh bot loc"]),
    "nem-chua":      (["nem chua", "vietnamese fermented pork"], ["nem chua"], ["nem chua"]),
    "cha-ca-la-vong": (["cha ca la vong", "chả cá hà nội"], ["chả cá lã vọng"], ["cha ca la vong"]),
    "hot-vit-lon":   (["balut egg", "hột vịt lộn"], ["hột vịt lộn", "trứng vịt lộn"], ["balut"]),
    "com-lang-vong": (["com green rice flakes vietnam", "cốm"], ["cốm làng vòng", "cốm"], ["young green rice flakes"]),
    "xoi-xeo":       (["xoi xeo", "xôi xéo"], ["xôi xéo"], ["xoi xeo"]),
    "banh-gio":      (["banh gio", "bánh giò"], ["bánh giò"], ["banh gio"]),
    "ao-tu-than":    (["ao tu than", "non quai thao", "áo tứ thân"], ["áo tứ thân", "nón quai thao"], ["ao tu than"]),
    "khan-ran":      (["khan ran scarf", "khăn rằn"], ["khăn rằn"], ["khan ran checkered scarf"]),
    "mam-ngu-qua":   (["mam ngu qua tet", "mâm ngũ quả"], ["mâm ngũ quả"], ["tet five fruit tray"]),
    "tranh-dong-ho": (["dong ho painting", "tranh đông hồ"], ["tranh đông hồ"], ["dong ho folk painting"]),
    "banh-trang-tron": ([], ["bánh tráng trộn"], ["banh trang tron"]),
    "banh-trang-nuong": ([], ["bánh tráng nướng"], ["banh trang nuong"]),
    "che-troi-nuoc": ([], ["chè trôi nước"], ["che troi nuoc"]),
    "banh-da-lon":   ([], ["bánh da lợn"], ["banh da lon"]),
    "banh-chung":    ([], ["bánh chưng"], ["banh chung"]),
    "banh-tet":      ([], ["bánh tét"], ["banh tet"]),
    "ao-ba-ba":      ([], ["áo bà ba"], ["ao ba ba"]),
    "ao-yem":        ([], ["áo yếm"], ["ao yem"]),
    "bun-rieu":      ([], ["bún riêu"], ["bun rieu"]),
    "banh-it-la-gai": ([], ["bánh ít lá gai"], ["banh it la gai"]),
    "ca-kho-to":     ([], ["cá kho tộ"], ["ca kho to"]),
    "thit-dong":     ([], ["thịt đông", "thịt nấu đông"], ["vietnamese pork aspic"]),
}


def _open(url: str, timeout: int = 60, tries: int = 4) -> bytes:
    """Tải một URL; 429 / 503 thì chờ rồi thử lại (Pixabay: 100 lời gọi / 60 giây)."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or k == tries - 1:
                raise
            time.sleep(float(e.headers.get("Retry-After") or 0) or 15 * 2 ** k)
    raise RuntimeError("unreachable")


def candidates(query: str, key: str) -> list[dict]:
    """Ảnh Pixabay khớp truy vấn (ảnh chụp, đủ lớn), xếp theo độ liên quan của Pixabay."""
    lang = "vi" if not query.isascii() else "en"
    q = urllib.parse.urlencode({"key": key, "q": query, "lang": lang, "image_type": "photo", "per_page": 20,
                                "min_width": 640, "safesearch": "false"})
    hits = json.loads(_open(f"{API}?{q}", timeout=30).decode("utf-8")).get("hits", [])
    return [{"title": h.get("tags", ""), "thumb": h["largeImageURL"], "page": h.get("pageURL"), "license": LICENSE,
             "artist": str(h.get("user", ""))[:120], "query": query} for h in hits if h.get("largeImageURL")]


def fetch(cid: str, out: Path, key: str) -> int:
    seen, picked = set(), []
    for q in CONCEPTS[cid][0]:
        for c in candidates(q, key):
            if c["page"] not in seen and len(picked) < PER:
                seen.add(c["page"])
                picked.append(c)
        time.sleep(1.0)
    d = out / cid
    d.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(picked):
        (d / f"{i}.jpg").write_bytes(_open(c["thumb"]))
        c["file"] = f"{i}.jpg"
        time.sleep(0.5)
    (d / "meta.json").write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(picked)


if __name__ == "__main__":
    key = os.environ.get("PIXABAY_KEY") or sys.exit("thiếu biến môi trường PIXABAY_KEY")
    root = Path(sys.argv[1])
    for cid in sys.argv[2:] or CONCEPTS:
        try:
            print(cid, fetch(cid, root, key), flush=True)
        except Exception as e:  # một khái niệm lỗi mạng không dừng cả lượt
            print(cid, "LỖI", type(e).__name__, flush=True)   # không in e: thông báo lỗi có thể chứa URL kèm khoá
