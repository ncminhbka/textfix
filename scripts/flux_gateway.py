#!/usr/bin/env python3
"""MÁY FLUX GIẢ theo khuôn API của BFL (api.bfl.ai), vẽ bằng FLUX.2-klein 4B tại chỗ -- để chạy thử bản docker (FLUX ở máy
khác, textfix/draw/remote.py) ngay trên notebook khi chưa nối được tới FLUX 9B thật.

  python scripts/flux_gateway.py --port 8010 --device cuda:0
  # container: TEXTFIX_FLUX_URL=http://localhost:8010/v1/flux-2-klein-4b  (docker run --network container:<notebook>)

  POST /v1/<tên model>        {prompt, width, height, seed, input_image[, input_image_2..4]} -> {id, polling_url}
  GET  /v1/get_result?id=...  {id, status: Pending | Ready | Error, result: {sample}}
  GET  /v1/sample/<id>.png    ảnh
Một bản FLUX, vẽ lần lượt (hàng đợi); kết quả giữ trong bộ nhớ 30 phút.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import queue
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from PIL import Image  # noqa: E402

TASKS: dict[str, dict] = {}
Q: "queue.Queue[str]" = queue.Queue()
KEEP = 30 * 60
app = FastAPI(title="FLUX gateway (khuôn BFL)")


def decode(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64.split(",", 1)[-1]))).convert("RGB"))


def worker(F) -> None:
    while True:
        tid = Q.get()
        t = TASKS.get(tid)
        if t is None:
            continue
        t["status"] = "Generating"
        try:
            b = t["body"]
            refs = [decode(b[k]) for k in ("input_image", "input_image_2", "input_image_3", "input_image_4") if b.get(k)]
            t0 = time.time()
            img = F.generate(b["prompt"], int(b["width"]), int(b["height"]), int(b.get("seed") or 0), refs=refs or None)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, "PNG")
            t.update(status="Ready", png=buf.getvalue())
            print(f"{tid} xong {time.time() - t0:.1f}s {b['width']}x{b['height']} seed {b.get('seed')}", flush=True)
        except Exception as e:
            t.update(status="Error", details=f"{type(e).__name__}: {e}")
            print(f"{tid} lỗi {e}", flush=True)
        for k in [k for k, v in TASKS.items() if time.time() - v["t"] > KEEP]:
            TASKS.pop(k, None)


@app.post("/v1/{model}")
async def submit(model: str, req: Request):
    body = await req.json()
    if not body.get("prompt") or not body.get("width") or not body.get("height"):
        raise HTTPException(422, "cần prompt, width, height")
    tid = uuid.uuid4().hex[:12]
    TASKS[tid] = {"status": "Pending", "body": body, "t": time.time()}
    Q.put(tid)
    base = str(req.base_url).rstrip("/")
    return {"id": tid, "polling_url": f"{base}/v1/get_result?id={tid}"}


@app.get("/v1/get_result")
async def result(id: str, req: Request):
    t = TASKS.get(id)
    if t is None:
        return {"id": id, "status": "Task not found"}
    out = {"id": id, "status": t["status"], "result": None, "details": t.get("details")}
    if t["status"] == "Ready":
        out["result"] = {"sample": f"{str(req.base_url).rstrip('/')}/v1/sample/{id}.png"}
    return out


@app.get("/v1/sample/{name}")
async def sample(name: str):
    t = TASKS.get(name.rsplit(".", 1)[0])
    if t is None or "png" not in t:
        raise HTTPException(404)
    return Response(t["png"], media_type="image/png")


def main() -> None:
    # image notebook đặt libcuda lệch driver (.../compat) lên đầu LD_LIBRARY_PATH (lỗi 803): bỏ đi rồi tự chạy lại
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if "/compat" in ld:
        os.environ["LD_LIBRARY_PATH"] = ":".join(p for p in ld.split(":") if "/compat" not in p)
        os.execv(sys.executable, [sys.executable, *sys.argv])
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", choices=["distill", "base"], default="distill")
    ap.add_argument("--weights-dir", default=os.environ.get("TEXTFIX_WEIGHTS"))
    a = ap.parse_args()
    from textfix.draw.flux import Flux
    print(f"nạp FLUX {a.model} lên {a.device} ...", flush=True)
    F = Flux(a.model, weights_dir=a.weights_dir, device=a.device).load()
    threading.Thread(target=worker, args=(F,), daemon=True).start()
    print(f"sẵn sàng: http://localhost:{a.port}/v1/flux-2-klein-4b", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
