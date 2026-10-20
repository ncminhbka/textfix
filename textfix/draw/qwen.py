"""Qwen-Image (bản diffusers, mặc định Qwen-Image-2.1-Turbo) vẽ nháp TẠI CHỖ thay FLUX.2-klein: cùng giao diện với flux.Flux.

  Q = QwenDraw(device="cuda:0").load()        # start_server.sh --draw qwen: mỗi card một bản, cùng card với Hi-SAM / LaMa
  draft = Q.generate(prompt_en, W, H, seed, refs=[product])

  TEXTFIX_QWEN_DIR    thư mục diffusers (model_index.json), mặc định ~/persistent-data/Qwen-Image-2.1-Turbo
  TEXTFIX_QWEN_MEM    offload (mặc định: từng phần lên card khi cần -- 12/10 scripts/qwen_test.py: ~45 s / nháp, VRAM đỉnh
                      16,6 GB, chừa chỗ cho Hi-SAM / LaMa trên A30 24 GB) | cuda (cả pipeline trên card, nhanh hơn nếu vừa)
  TEXTFIX_QWEN_STEPS, TEXTFIX_QWEN_CFG   không đặt: mặc định của pipeline (bản Turbo đã chỉnh sẵn)
Ảnh tham chiếu (ảnh sản phẩm, ảnh kho): chỉ gửi khi pipeline có tham số nhận ảnh (IMAGE_ARGS); không có thì vẽ không tham chiếu
và ghi vào log một lần -- prompt vẫn nhắc "image N" như với FLUX.
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("textfix.qwen")

DEFAULT_DIR = Path.home() / "persistent-data" / "Qwen-Image-2.1-Turbo"
IMAGE_ARGS = ("image", "images", "input_image", "input_images", "reference_image", "reference_images", "image_reference")
MEMS = ("offload", "cuda")


def model_dir() -> Path:
    return Path(os.path.expanduser(os.environ.get("TEXTFIX_QWEN_DIR") or str(DEFAULT_DIR)))


class QwenDraw:
    def __init__(self, device: str = "cuda:0", model: str | Path | None = None, mem: str | None = None,
                 steps: int | None = None, cfg: float | None = None):
        self.dir = Path(model) if model else model_dir()
        self.mem = mem or os.environ.get("TEXTFIX_QWEN_MEM", "offload")
        if self.mem not in MEMS:
            raise ValueError(f"TEXTFIX_QWEN_MEM phải là một trong {', '.join(MEMS)}, không phải {self.mem!r}")
        self.steps = steps or (int(os.environ["TEXTFIX_QWEN_STEPS"]) if os.environ.get("TEXTFIX_QWEN_STEPS") else None)
        self.cfg = cfg if cfg is not None else (float(os.environ["TEXTFIX_QWEN_CFG"]) if os.environ.get("TEXTFIX_QWEN_CFG") else None)
        self.dev_dit = device   # tên giữ như flux.Flux: engine.draw đặt Hi-SAM / LaMa theo trường này
        self.pipe, self.params, self.image_arg, self._warned = None, set(), "", False

    def load(self) -> "QwenDraw":
        if not (self.dir / "model_index.json").is_file():
            raise FileNotFoundError(f"không thấy {self.dir}/model_index.json -- đặt TEXTFIX_QWEN_DIR (thư mục diffusers của Qwen-Image)")
        import torch
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(str(self.dir), torch_dtype=torch.bfloat16)
        if self.mem == "cuda":
            pipe.to(self.dev_dit)
        else:
            pipe.enable_model_cpu_offload(device=self.dev_dit)
        vae = getattr(pipe, "vae", None)
        for fn in ("enable_tiling", "enable_slicing"):
            if vae is not None and hasattr(vae, fn):
                getattr(vae, fn)()
        self.pipe = pipe
        self.params = set(inspect.signature(pipe.__call__).parameters)
        self.image_arg = next((x for x in IMAGE_ARGS if x in self.params), "")
        return self

    def info(self) -> dict:
        return {"variant": "qwen", "model": self.dir.name, "pipeline": type(self.pipe).__name__ if self.pipe else None,
                "mem": self.mem, "steps": self.steps or "mặc định", "cfg": self.cfg if self.cfg is not None else "mặc định",
                "image_arg": self.image_arg or None, "devices": {"dit": self.dev_dit}}

    def call_args(self, prompt: str, w: int, h: int, seed: int, refs: list[np.ndarray] | None) -> dict:
        """Tham số gọi pipeline; chỉ giữ tham số pipeline có (bản Qwen khác nhau tên CFG / tên ảnh vào)."""
        import torch
        kw = {"prompt": prompt, "width": w, "height": h, "generator": torch.Generator("cpu").manual_seed(int(seed))}
        if self.steps:
            kw["num_inference_steps"] = self.steps
        if self.cfg is not None:
            kw["true_cfg_scale" if "true_cfg_scale" in self.params else "guidance_scale"] = self.cfg
            if self.cfg > 1:
                kw["negative_prompt"] = " "
        if refs:
            if self.image_arg:
                from PIL import Image
                ims = [Image.fromarray(np.asarray(r)).convert("RGB") for r in refs]
                kw[self.image_arg] = ims if self.image_arg.endswith("s") or len(ims) > 1 else ims[0]
            elif not self._warned:
                log.warning("%s không có tham số nhận ảnh (%s): vẽ không ảnh tham chiếu", type(self.pipe).__name__,
                            "/".join(IMAGE_ARGS))
                self._warned = True
        return {k: v for k, v in kw.items() if k in self.params}

    def generate(self, prompt: str, w: int, h: int, seed: int, refs: list[np.ndarray] | None = None) -> np.ndarray:
        """Vẽ một ảnh w x h (bội số 16) -> RGB uint8 (h, w, 3)."""
        import torch
        if w % 16 or h % 16:
            raise ValueError(f"cỡ ảnh phải là bội số 16: {w}x{h}")
        with torch.no_grad():
            img = self.pipe(**self.call_args(prompt, w, h, seed, refs)).images[0].convert("RGB")
        if img.size != (w, h):
            from PIL import Image
            img = img.resize((w, h), Image.LANCZOS)
        torch.cuda.empty_cache()
        return np.asarray(img)
