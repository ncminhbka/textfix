"""NÉT CHỮ của nháp bằng Hi-SAM (github ymy-k/Hi-SAM: SAM ViT-L + phần tự gợi ý "cắt chữ", bản huấn luyện trên TextSeg).

  ink = strokes(draft)            # bool cỡ ảnh: điểm ảnh nào là NÉT CHỮ (kể cả chữ giun, chữ cong, chữ OCR không đọc ra)

Từ 14/10 đây là nguồn DUY NHẤT cho biết chữ nằm ở đâu ở mức điểm ảnh -- thay bản xoá FLUX (đường cũ: nhánh git
`legacy-flux-erase`). Nó chỉ thấy THÂN NÉT: viền dày, bóng đổ, quầng sáng quanh chữ do bước gỡ chữ (erase.erase) tự nới.
Đo trên 2 x A30 (14/10, 30 nháp accept2): fp16 0,17 s / ảnh, mặt nạ trùng fp32 >= 0,997; model 1,3 GB, đỉnh VRAM 4,4 GB.

Tệp (máy chủ không ra internet: docs/SERVER.md mục Hi-SAM), gốc ~/persistent-data/hisam, đổi bằng TEXTFIX_HISAM:
  Hi-SAM/                                           mã nguồn
  Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth nền SAM ViT-L
  sam_tss_l_textseg.pth                             phần nét chữ
Thiếu tệp: báo lỗi rõ và DỪNG (người dùng chốt 14/10: không lùi về cách nào khác).
Mỗi thiết bị (cuda:0, cuda:1, cpu) một bản, nạp lần đầu dùng; một khoá cho mỗi bản (bộ dự đoán giữ trạng thái của ảnh đang xét).
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np

HOME = Path(os.path.expanduser(os.environ.get("TEXTFIX_HISAM", "~/persistent-data/hisam")))
REPO = HOME / "Hi-SAM"
SAM = REPO / "pretrained_checkpoint" / "sam_vit_l_0b3195.pth"
CKPT = HOME / "sam_tss_l_textseg.pth"
HALF = os.environ.get("TEXTFIX_HISAM_HALF", "on") != "off"   # fp16 trên GPU

_NETS: dict = {}
_LOAD = threading.Lock()


def missing() -> list[str]:
    """Các tệp còn thiếu (rỗng = đủ)."""
    return [str(p) for p in (REPO / "hi_sam" / "modeling" / "build.py", SAM, CKPT) if not p.is_file()]


def _device(device: str | None) -> str:
    if device:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(device: str | None = None):
    """Nạp (một lần cho mỗi thiết bị) -> (bộ dự đoán, khoá). Không đổi thư mục làm việc: tự ghép trọng số nền SAM + phần nét chữ
    (hàm dựng của tác giả đọc nền SAM theo đường dẫn tương đối)."""
    import torch
    device = _device(device)
    with _LOAD:
        if device in _NETS:
            return _NETS[device]
        lost = missing()
        if lost:
            raise RuntimeError("thiếu tệp Hi-SAM (docs/SERVER.md, mục Hi-SAM): " + "; ".join(lost))
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import argparse

        from hi_sam.modeling.build import model_registry
        from hi_sam.modeling.predictor import SamPredictor
        ns = argparse.Namespace(model_type="vit_l", checkpoint=None, device=device, hier_det=False, input_size=[1024, 1024],
                                patch_mode=False, attn_layers=1, prompt_len=12)
        model = model_registry["vit_l"](ns)
        state = torch.load(str(CKPT), map_location="cpu")
        if any(k in state for k in ("optimizer", "lr_scheduler", "epoch")):
            state = state["model"]
        for k, v in torch.load(str(SAM), map_location="cpu").items():
            state.setdefault(k, v)
        info = model.load_state_dict(state, strict=False)
        if info.missing_keys:
            raise RuntimeError(f"trọng số Hi-SAM không khớp model (thiếu {len(info.missing_keys)} khoá, vd. {info.missing_keys[:3]})")
        _NETS[device] = (SamPredictor(model.eval().to(device)), threading.Lock())
        return _NETS[device]


def strokes(img: np.ndarray, device: str | None = None) -> np.ndarray:
    """Mặt nạ nét chữ (bool, cỡ ảnh) của một ảnh RGB uint8."""
    import torch
    device = _device(device)
    P, lock = load(device)
    gpu = device.startswith("cuda")
    with lock, torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=gpu and HALF):
        P.set_image(np.ascontiguousarray(img))
        _m, hr, _s, _hs = P.predict(multimask_output=False)
    return np.asarray(hr[0]).astype(bool)


def to_png(ink: np.ndarray, path) -> None:
    """Lưu mặt nạ 1 bit (vài chục KB)."""
    from PIL import Image
    Image.fromarray((np.asarray(ink) * 255).astype(np.uint8)).convert("1").save(path)


def from_png(path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Đọc mặt nạ đã lưu; size = (rộng, cao) của nháp nếu cần ép cỡ."""
    from PIL import Image
    im = Image.open(path).convert("L")
    if size and im.size != tuple(size):
        im = im.resize(tuple(size), Image.NEAREST)
    return np.asarray(im) > 127
