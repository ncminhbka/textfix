"""ĐẨY KHO tham chiếu lên kho đối tượng S3 (MinIO nội bộ). Chạy trên notebook máy chủ (laptop không tới được MinIO).

  python scripts/kho_push.py --probe <host>        # dò cổng API S3 (web console thường KHÔNG phải cổng API)
  python scripts/kho_push.py <thư mục kho>         # thư mục có index.json + img/ (giải nén từ zip do kho_index.py dựng)

Cùng biến môi trường với máy chủ (textfix/kho_sync.py): TEXTFIX_KHO_ENDPOINT, TEXTFIX_KHO_BUCKET, TEXTFIX_KHO_PREFIX
(mặc định textfix-kho/), TEXTFIX_KHO_ACCESS_KEY, TEXTFIX_KHO_SECRET_KEY. Không ghi giá trị nào vào repo.
Ảnh đẩy trước, index.json SAU CÙNG: máy chủ đồng bộ giữa chừng không bao giờ thấy mục lục trỏ vào ảnh chưa có. Đẩy xong đọc
lại bằng chính bộ đồng bộ của máy chủ vào một thư mục tạm để chắc chắn máy chủ dùng được.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from textfix import kho  # noqa: E402
from textfix.kho_sync import s3_getter, sync  # noqa: E402

PORTS = sorted({80, 443, 7000, 9443, 10000, 19000, *range(9000, 9011), *range(8000, 8101)})


def _open_ports(host: str, ports, timeout: float = 1.5) -> list[int]:
    """Cổng TCP đang mở (dò song song)."""
    import socket
    from concurrent.futures import ThreadPoolExecutor

    def ok(port):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return port
        except OSError:
            return None
    with ThreadPoolExecutor(32) as ex:
        return [p for p in ex.map(ok, ports) if p]


def probe(host: str) -> None:
    """Cổng API S3 trả XML lỗi S3 cho GET không ký. Web console MinIO cũng trả XML, nhưng với câu "S3 API Requests must be made
    to API port" -> không phải cổng API."""
    found = _open_ports(host, PORTS)
    print("cổng mở:", found or "không có")
    for port in found:
        for scheme in ("http", "https"):
            url = f"{scheme}://{host}:{port}/"
            try:
                with urllib.request.urlopen(url, timeout=4) as r:
                    head, body = dict(r.headers), r.read(600).decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                head, body = dict(e.headers), e.read(600).decode("utf-8", "replace")
            except Exception:
                continue
            if "API port" in body:
                kind = "web console (KHÔNG phải cổng API)"
            elif "<Error>" in body or "AmazonS3" in body or "MinIO" in head.get("Server", ""):
                kind = "API S3  <-- dùng cổng này"
            else:
                kind = "không phải S3"
            print(f"{url:34} {kind}  Server={head.get('Server', '')}")
            break


def push(src: Path) -> None:
    from minio import Minio
    env = os.environ
    endpoint, bucket = env["TEXTFIX_KHO_ENDPOINT"], env["TEXTFIX_KHO_BUCKET"]
    prefix = env.get("TEXTFIX_KHO_PREFIX", "textfix-kho/").rstrip("/") + "/"
    entries = kho.load(src)
    bad = kho.problems(entries, src)
    if bad or not entries:
        sys.exit("mục lục chưa ổn:\n" + "\n".join(bad or ["kho rỗng"]))
    client = Minio(endpoint.split("://", 1)[-1].rstrip("/"), access_key=env.get("TEXTFIX_KHO_ACCESS_KEY", ""),
                   secret_key=env.get("TEXTFIX_KHO_SECRET_KEY", ""), secure=endpoint.startswith("https://"))

    def put(rel: str, ctype: str) -> None:
        data = (src / rel).read_bytes()
        client.put_object(bucket, prefix + rel, io.BytesIO(data), len(data), content_type=ctype)

    for e in entries:
        put(e["file"], "image/jpeg" if e["file"].lower().endswith((".jpg", ".jpeg")) else "image/png")
    put("index.json", "application/json")
    print(f"đã đẩy {len(entries)} ảnh + index.json lên {bucket}/{prefix}")
    with tempfile.TemporaryDirectory() as tmp:   # đọc lại như máy chủ
        r = sync(s3_getter(endpoint, bucket, env.get("TEXTFIX_KHO_ACCESS_KEY", ""), env.get("TEXTFIX_KHO_SECRET_KEY", "")),
                 prefix, Path(tmp))
        print("đọc lại như máy chủ:", json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--probe"]:
        probe(sys.argv[2])
    else:
        push(Path(sys.argv[1]))
