"""ĐỒNG BỘ KHO ẢNH THAM CHIẾU từ kho đối tượng S3 (MinIO nội bộ) về thư mục đệm mà textfix/kho.py đọc.

  report = sync_from_env()          # máy chủ gọi lúc khởi động; không bao giờ ném lỗi

Biến môi trường (không ghi giá trị vào repo):
  TEXTFIX_KHO_ENDPOINT     địa chỉ API S3, "http://host:cổng" hoặc "host:cổng" (https nếu ghi https://). Trống -> tắt đồng bộ
  TEXTFIX_KHO_BUCKET       tên bucket
  TEXTFIX_KHO_PREFIX       thư mục trong bucket (mặc định "textfix-kho/")
  TEXTFIX_KHO_ACCESS_KEY / TEXTFIX_KHO_SECRET_KEY
  TEXTFIX_KHO_DIR          thư mục đệm (kho.kho_dir(); nên đặt ở chỗ ghi được, ví dụ output/kho)

Trên kho đối tượng: <prefix>index.json + <prefix><file> của từng mục (scripts/kho_push.py đẩy, index.json sau cùng).
Mỗi mục có "sha1" của ảnh (scripts/kho_index.py ghi): ảnh đệm trùng sha1 thì không tải lại.
Hỏng ở bất kỳ đâu -> giữ nguyên đệm cũ (hoặc kho rỗng): vẽ như khi chưa có kho, không bao giờ làm hỏng luồng chính.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Callable

from . import kho

Getter = Callable[[str], bytes]   # khoá đầy đủ trong bucket -> nội dung


def sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def safe_rel(file: str) -> bool:
    """Đường dẫn ảnh trong mục lục chỉ được trỏ vào bên trong thư mục kho (không tuyệt đối, không "..")."""
    p = PurePosixPath(str(file))
    return bool(file) and not p.is_absolute() and ".." not in p.parts and "\\" not in str(file) and ":" not in str(file)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _fresh(path: Path, want: str | None) -> bool:
    """Ảnh đệm còn dùng được: có file và (nếu mục lục ghi sha1) đúng sha1."""
    return path.is_file() and (not want or sha1(path.read_bytes()) == want)


def sync(get: Getter, prefix: str, cache: Path) -> dict:
    """Tải mục lục + ảnh còn thiếu / đã đổi về cache. Mục tải hỏng bị bỏ khỏi mục lục đệm. Mục lục ghi SAU CÙNG.
    -> {status: ok | error, entries, downloaded, dropped, error?}"""
    prefix = prefix.rstrip("/") + "/" if prefix.strip("/") else ""
    try:
        doc = json.loads(get(prefix + "index.json").decode("utf-8"))
        entries = doc.get("entries")
        if not isinstance(entries, list):
            raise ValueError("index.json không có danh sách entries")
    except Exception as e:   # không đọc được mục lục: giữ đệm cũ
        return {"status": "error", "error": f"index.json: {type(e).__name__}: {e}", "entries": len(kho.load(cache))}
    keep, got, dropped = [], 0, []
    for e in entries:
        f = e.get("file") if isinstance(e, dict) else None
        if not (isinstance(e, dict) and e.get("id") and isinstance(f, str) and safe_rel(f)):
            dropped.append(str(e.get("id") if isinstance(e, dict) else e))
            continue
        path = cache / f
        try:
            if not _fresh(path, e.get("sha1")):
                data = get(prefix + f)
                if e.get("sha1") and sha1(data) != e["sha1"]:
                    raise ValueError("sai sha1")
                _write_atomic(path, data)
                got += 1
            keep.append(e)
        except Exception:
            dropped.append(e["id"])
    _write_atomic(cache / "index.json", json.dumps({**doc, "entries": keep}, ensure_ascii=False, indent=1).encode("utf-8"))
    return {"status": "ok", "entries": len(keep), "downloaded": got, "dropped": dropped}


def s3_getter(endpoint: str, bucket: str, access: str, secret: str, timeout: float = 20) -> Getter:
    """Hàm đọc object của bucket qua thư viện minio (chỉ cần khi bật đồng bộ)."""
    import urllib3
    from minio import Minio
    secure = endpoint.startswith("https://")
    host = endpoint.split("://", 1)[-1].rstrip("/")
    http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=5, read=timeout), retries=urllib3.Retry(total=2, backoff_factor=0.5))
    client = Minio(host, access_key=access, secret_key=secret, secure=secure, http_client=http)

    def get(key: str) -> bytes:
        r = client.get_object(bucket, key)
        try:
            return r.read()
        finally:
            r.close()
            r.release_conn()
    return get


def sync_from_env(env: dict | None = None) -> dict:
    """Đồng bộ theo biến môi trường. Không cấu hình -> {status: off}. Không ném lỗi."""
    env = os.environ if env is None else env
    endpoint, bucket = env.get("TEXTFIX_KHO_ENDPOINT", "").strip(), env.get("TEXTFIX_KHO_BUCKET", "").strip()
    cache = kho.kho_dir()
    if not endpoint or not bucket:
        return {"status": "off", "entries": len(kho.load(cache))}
    try:
        get = s3_getter(endpoint, bucket, env.get("TEXTFIX_KHO_ACCESS_KEY", ""), env.get("TEXTFIX_KHO_SECRET_KEY", ""))
        return sync(get, env.get("TEXTFIX_KHO_PREFIX", "textfix-kho/"), cache)
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "entries": len(kho.load(cache))}
