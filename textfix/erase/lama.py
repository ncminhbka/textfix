"""LaMa (TorchScript big-lama.pt) vẽ lại nền dưới mặt nạ: một bản mỗi thiết bị, nạp lần đầu dùng.
"""

from __future__ import annotations

import os
import threading

import numpy as np

LAMA_PATH = os.path.expanduser(os.environ.get("TEXTFIX_LAMA", "~/persistent-data/lama/big-lama.pt"))


_LAMA: dict = {}


_LAMA_LOCK = threading.Lock()


def lama_ready() -> bool:
    return os.path.exists(LAMA_PATH)


def _lama(device: str | None):
    """Một bản LaMa (TorchScript) mỗi thiết bị, nạp lần đầu dùng (có khoá: nhiều luồng thiết kế gọi cùng lúc) -> (model, khoá)."""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with _LAMA_LOCK:
        if dev not in _LAMA:
            _LAMA[dev] = (torch.jit.load(LAMA_PATH, map_location=dev).eval(), threading.Lock())
        return dev, *_LAMA[dev]


def inpaint(img: np.ndarray, mask: np.ndarray, device: str | None = None) -> np.ndarray:
    """LaMa vẽ lại phần trong mặt nạ theo nền xung quanh (một lượt, không khuếch tán). Ảnh đệm phản chiếu tới bội số 8."""
    import torch
    dev, net, lock = _lama(device)
    H, W = img.shape[:2]
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    im = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    mk = np.pad(mask.astype(np.float32), ((0, ph), (0, pw)))
    with lock, torch.no_grad():
        o = net(torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255).to(dev), torch.from_numpy(mk)[None, None].to(dev))
    return (o[0].permute(1, 2, 0).float().cpu().numpy()[:H, :W] * 255).round().clip(0, 255).astype(np.uint8)


def _blend(inside: np.ndarray, outside: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ghép theo mặt nạ, mép mềm 1 px (không lộ đường cắt)."""
    import cv2
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)[..., None]
    return (inside * a + outside * (1 - a)).round().clip(0, 255).astype(np.uint8)
