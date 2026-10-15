"""FLUX ở MÁY KHÁC (API của Black Forest Labs, khuôn api.bfl.ai): thay cho flux.Flux khi máy chủ không tự nạp FLUX.

  F = RemoteFlux(device="cuda:0")                       # TEXTFIX_FLUX_URL, TEXTFIX_FLUX_API_KEY (docs/SERVER.md)
  draft = F.generate(prompt_en, W, H, seed, refs=[product])   # cùng chữ ký với flux.Flux.generate

Một lời gọi: POST {prompt, width, height, seed, output_format: png, input_image (base64)} kèm header x-key -> {id, polling_url};
hỏi polling_url tới khi status "Ready" -> result.sample (URL ký, sống 10 phút) -> tải ảnh. Mặc định nên dùng điểm cuối cố định
`/v1/flux-2-klein-9b` (không phải `-preview`): cùng seed -> cùng ảnh, chạy lại được.
`device`: card chạy Hi-SAM / LaMa cho nháp này (FLUX ở xa nên đây chỉ là chỗ đặt hai model gỡ chữ, engine.draw đọc dev_dit).
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np

POLL = 0.5          # giây giữa hai lần hỏi kết quả (theo tài liệu BFL)
BUSY_WAIT = 2.0     # 429 (quá số việc đang chạy của tài khoản): chờ rồi gửi lại
REF_PIXELS = 2048 ** 2   # ảnh tham chiếu lớn hơn thì thu nhỏ trước khi gửi (thân yêu cầu gọn, FLUX vẫn đủ chi tiết)


class FluxError(RuntimeError):
    """FLUX ở xa hỏng (gọi lỗi, hết giờ, trả lỗi)."""


class FluxModerated(FluxError):
    """Máy FLUX từ chối prompt hoặc ảnh (bộ lọc nội dung phía nhà cung cấp, không phải của TextFix)."""


def enabled() -> bool:
    return bool(os.environ.get("TEXTFIX_FLUX_URL"))


def concurrency() -> int:
    """Số nháp gửi sang máy FLUX cùng lúc (= số luồng VẼ của máy chủ). BFL giới hạn 24 việc đang chạy mỗi tài khoản."""
    return max(1, int(os.environ.get("TEXTFIX_FLUX_CONCURRENCY", "2")))


class RemoteFlux:
    def __init__(self, url: str | None = None, key: str | None = None, device: str | None = None,
                 timeout: float | None = None, safety: int | None = None):
        self.url = url or os.environ.get("TEXTFIX_FLUX_URL", "")
        self.key = key if key is not None else os.environ.get("TEXTFIX_FLUX_API_KEY", "")
        self.timeout = timeout or float(os.environ.get("TEXTFIX_FLUX_TIMEOUT", "120"))
        self.safety = safety if safety is not None else int(os.environ.get("TEXTFIX_FLUX_SAFETY", "2"))
        self.dev_dit = device   # tên giữ như flux.Flux: engine.draw / LaMa đặt model gỡ chữ theo trường này
        if not self.url:
            raise FluxError("chưa đặt TEXTFIX_FLUX_URL")

    def load(self) -> "RemoteFlux":
        return self

    def info(self) -> dict:
        return {"variant": "remote", "model": self.url.rstrip("/").rsplit("/", 1)[-1], "url": self.url,
                "safety_tolerance": self.safety, "devices": {"hisam_lama": self.dev_dit}}

    def generate(self, prompt: str, w: int, h: int, seed: int, refs: list[np.ndarray] | None = None) -> np.ndarray:
        """Vẽ một ảnh w x h (bội số 16) -> RGB uint8 (h, w, 3)."""
        if w % 16 or h % 16:
            raise ValueError(f"cỡ ảnh phải là bội số 16: {w}x{h}")
        body = {"prompt": prompt, "width": w, "height": h, "seed": int(seed), "output_format": "png",
                "safety_tolerance": self.safety}
        for k, im in enumerate((refs or [])[:4]):   # klein nhận tối đa 4 ảnh: input_image, input_image_2..4
            body["input_image" if k == 0 else f"input_image_{k + 1}"] = _b64(im)
        t0 = time.time()
        task = self._submit(body, t0)
        sample = self._wait(task, t0)
        return _fit(_decode(self._get(sample, raw=True)), w, h)

    # ------------------------------------------------------------------------------------------------ HTTP
    def _submit(self, body: dict, t0: float) -> dict:
        while True:
            try:
                return self._post(self.url, body)
            except urllib.error.HTTPError as e:
                if e.code == 429 and time.time() - t0 < self.timeout:
                    time.sleep(BUSY_WAIT)
                    continue
                raise FluxError(f"FLUX từ chối yêu cầu (HTTP {e.code}): {_reason(e)}") from e
            except urllib.error.URLError as e:
                raise FluxError(f"không gọi được FLUX {self.url}: {e.reason}") from e

    def _wait(self, task: dict, t0: float) -> str:
        poll = task.get("polling_url") or _result_url(self.url, task["id"])
        while time.time() - t0 < self.timeout:
            r = self._get(poll)
            st = r.get("status")
            if st == "Ready":
                return r["result"]["sample"]
            if st in ("Request Moderated", "Content Moderated"):
                raise FluxModerated(f"máy FLUX lọc nội dung ({st}): {r.get('details') or ''}".strip())
            if st in ("Error", "Task not found", "No Card Generated"):
                raise FluxError(f"FLUX lỗi ({st}): {r.get('details') or ''}".strip())
            time.sleep(POLL)   # Pending / Reasoning / Generating
        raise FluxError(f"FLUX quá {self.timeout:.0f} s chưa vẽ xong (việc {task.get('id')})")

    def _headers(self) -> dict:
        return {"accept": "application/json", "x-key": self.key, "Content-Type": "application/json"}

    def _post(self, url: str, body: dict) -> dict:
        req = urllib.request.Request(url, json.dumps(body).encode(), self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def _get(self, url: str, raw: bool = False):
        req = urllib.request.Request(url, headers={} if raw else self._headers())   # URL ký của ảnh: không gửi khoá
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            raise FluxError(f"FLUX lỗi khi {'tải ảnh' if raw else 'hỏi kết quả'} (HTTP {e.code}): {_reason(e)}") from e
        except urllib.error.URLError as e:
            raise FluxError(f"mất kết nối FLUX: {e.reason}") from e
        return data if raw else json.loads(data)


def _result_url(url: str, task_id: str) -> str:
    """Máy không trả polling_url: hỏi /v1/get_result?id= trên cùng máy."""
    base = url.split("/v1/", 1)[0] if "/v1/" in url else url.rsplit("/", 1)[0]
    return f"{base}/v1/get_result?id={task_id}"


def _reason(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode("utf-8", "replace")[:300]
    except Exception:
        return str(e.reason)


def _b64(im: np.ndarray) -> str:
    from PIL import Image
    img = Image.fromarray(np.asarray(im)).convert("RGB")
    if img.width * img.height > REF_PIXELS:
        k = (REF_PIXELS / (img.width * img.height)) ** 0.5
        img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode(data: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Máy FLUX trả lệch cỡ (làm tròn khác): đưa về đúng w x h, vì cả bước sau đo theo cỡ đã hẹn."""
    if img.shape[:2] == (h, w):
        return img
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((w, h), Image.LANCZOS))
