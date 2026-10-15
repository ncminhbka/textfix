"""FLUX ở máy khác (textfix/flux_api.py) với máy FLUX giả theo khuôn api.bfl.ai: gửi -> hỏi -> tải ảnh."""

from __future__ import annotations

import io
import json
import urllib.error

import numpy as np
import pytest
from PIL import Image

from textfix.draw import remote


def png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


class FakeBFL:
    """Thay urllib.request.urlopen: POST trả {id, polling_url}, GET hỏi trả lần lượt các trạng thái, GET ảnh trả PNG."""

    def __init__(self, statuses, size=(64, 48), busy=0):
        self.statuses, self.size, self.busy = list(statuses), size, busy
        self.posts, self.headers = [], []

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.headers.append((url, dict(req.header_items())))
        if req.get_method() == "POST":
            if self.busy:
                self.busy -= 1
                raise urllib.error.HTTPError(url, 429, "busy", {}, io.BytesIO(b"too many"))
            self.posts.append(json.loads(req.data))
            return io.BytesIO(json.dumps({"id": "t1", "polling_url": "https://api.test/v1/get_result?id=t1"}).encode())
        if "get_result" in url:
            st = self.statuses.pop(0)
            return io.BytesIO(json.dumps({"id": "t1", "status": st,
                                          "result": {"sample": "https://cdn.test/s.png"} if st == "Ready" else None}).encode())
        return io.BytesIO(png(*self.size))


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(remote, "POLL", 0)
    monkeypatch.setattr(remote, "BUSY_WAIT", 0)

    def use(*a, **kw):
        f = FakeBFL(*a, **kw)
        monkeypatch.setattr(remote.urllib.request, "urlopen", f)
        return f
    return use


def client():
    return remote.RemoteFlux(url="https://api.test/v1/flux-2-klein-9b", key="k", device="cpu", timeout=5)


def test_draws_after_polling_and_sends_what_the_draft_needs(fake):
    f = fake(["Pending", "Generating", "Ready"])
    img = client().generate("a poster", 64, 48, 7, refs=[np.zeros((10, 10, 3), np.uint8)])
    assert img.shape == (48, 64, 3) and img[0, 0].tolist() == [200, 30, 30]
    b = f.posts[0]
    assert (b["width"], b["height"], b["seed"], b["output_format"]) == (64, 48, 7, "png") and "input_image" in b
    assert f.headers[0][1].get("X-key") == "k" and "X-key" not in f.headers[-1][1]   # khoá không đi theo URL ký của ảnh


def test_wrong_size_from_host_is_brought_back_to_the_agreed_size(fake):
    fake(["Ready"], size=(70, 50))
    assert client().generate("x", 64, 48, 0).shape == (48, 64, 3)


def test_moderation_is_its_own_error(fake):
    fake(["Pending", "Content Moderated"])
    with pytest.raises(remote.FluxModerated):
        client().generate("x", 64, 48, 0)


def test_host_error_and_busy_account(fake):
    fake(["Error"])
    with pytest.raises(remote.FluxError):
        client().generate("x", 64, 48, 0)
    f = fake(["Ready"], busy=2)   # 429 hai lần (đủ việc đang chạy) -> chờ rồi gửi lại
    assert client().generate("x", 64, 48, 0).shape == (48, 64, 3) and len(f.posts) == 1


def test_result_url_when_host_gives_no_polling_url():
    assert remote._result_url("https://api.eu.bfl.ai/v1/flux-2-klein-9b", "abc") == "https://api.eu.bfl.ai/v1/get_result?id=abc"


def test_gateway_speaks_the_same_api_as_the_client(monkeypatch):
    # scripts/flux_gateway.py (máy FLUX giả trên notebook) <-> textfix.draw.remote: gửi, hỏi, tải ảnh qua HTTP thật
    import importlib.util
    import socket
    import threading
    import time

    import uvicorn
    spec = importlib.util.spec_from_file_location("flux_gateway", "scripts/flux_gateway.py")
    gw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gw)

    class Fake:
        def generate(self, prompt, w, h, seed, refs=None):
            return np.full((h, w, 3), 10 + seed % 200 + (len(refs) if refs else 0), np.uint8)
    threading.Thread(target=gw.worker, args=(Fake(),), daemon=True).start()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(gw.app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    monkeypatch.setattr(remote, "POLL", 0.02)
    F = remote.RemoteFlux(url=f"http://127.0.0.1:{port}/v1/flux-2-klein-4b", key="", device="cpu", timeout=20)
    img = F.generate("x", 64, 48, 5, refs=[np.zeros((8, 8, 3), np.uint8)])
    srv.should_exit = True
    assert img.shape == (48, 64, 3) and img[0, 0, 0] == 16
