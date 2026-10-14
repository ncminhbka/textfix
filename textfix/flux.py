"""FLUX.2-klein trên máy chủ: vẽ nháp (prompt -> ảnh).

  F = Flux("distill").load()                       # DiT + VAE + bộ mã hoá chữ (Qwen3-4B) nạp một lần, giữ trong VRAM
  draft = F.generate(prompt_en, 1024, 1024, seed)  # refs=[ảnh sản phẩm] nếu người dùng tải ảnh lên

Ảnh tham chiếu theo đúng cách FLUX.2 nhận (flux2.sampling.encode_image_refs): mã hoá VAE, ghép token ảnh tham chiếu sau token
ảnh đang khử nhiễu với trục thời gian RoPE t = 10, 20...; chỉ lấy dự đoán của phần ảnh đang khử nhiễu. Tham số mặc định đúng
BFL (flux2/util.py): distill 4 bước, base 50 bước CFG 4.0.

14/10: bước XOÁ LỚP PHỦ bằng FLUX (lời dặn "Remove all text...") đã rời nhánh main -- chữ nằm ở đâu do Hi-SAM trả lời
(textfix/hisam.py), nền dưới chữ do LaMa vẽ lại (textfix/lonly.py). Đường cũ: nhánh git `legacy-flux-erase`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


# cỡ sinh ~1 MP, bội số 16 (VAE nén 16x)
SIZES = {"1:1": (1024, 1024), "4:5": (896, 1120), "9:16": (768, 1344), "16:9": (1344, 768), "2:3": (832, 1248),
         "4:3": (1152, 864)}
MODELS = {"distill": ("flux.2-klein-4b", "flux-2-klein-4b.safetensors", "KLEIN_4B_MODEL_PATH", 4, None),
          "base": ("flux.2-klein-base-4b", "flux-2-klein-base-4b.safetensors", "KLEIN_4B_BASE_MODEL_PATH", 50, 4.0)}
HOME = Path(os.path.expanduser("~"))
SEARCH = [r / "persistent-data" / d for r in (HOME, Path("/home/jovyan"), Path("/"))
          for d in ("FLUX.2-klein-4B", "FLUX.2-klein-base-4B", "")]


def find_weights(extra: list[Path] | None = None) -> dict:
    """Tìm DiT distill / base, VAE, bộ mã hoá chữ; đặt biến môi trường flux2.util đọc. Trả {distill, base, vae, text_encoder}."""
    roots = [p for p in list(extra or []) + SEARCH if p.is_dir()]
    found = {}
    for key, (_, fname, env, _, _) in MODELS.items():
        if os.environ.get(env) and Path(os.environ[env]).is_file():
            found[key] = os.environ[env]
            continue
        hit = next((r / fname for r in roots if (r / fname).is_file()), None) or \
            next((r / "transformer" / fname for r in roots if (r / "transformer" / fname).is_file()), None)
        if hit:
            os.environ[env] = str(hit)
            found[key] = str(hit)
    vae = os.environ.get("AE_MODEL_PATH") or next(
        (str(r / "vae" / "diffusion_pytorch_model.safetensors") for r in roots if (r / "vae" / "diffusion_pytorch_model.safetensors").is_file()), None)
    if vae:
        os.environ["AE_MODEL_PATH"] = vae
        found["vae"] = vae
    te = os.environ.get("TEXT_ENCODER_PATH") or next(
        (str(r / "text_encoder") for r in roots if (r / "text_encoder" / "config.json").is_file()), None)
    if te:
        os.environ["TEXT_ENCODER_PATH"] = te
        found["text_encoder"] = te
    return found


class Flux:
    """DiT + VAE + bộ mã hoá chữ nạp sẵn. Thiết bị: TEXTFIX_DEVICE_DIT / _TE / _AE (mặc định cuda:0 cả ba -- chừa cuda:1
    cho VLM vLLM). Không dùng đồng thời từ nhiều luồng (máy chủ chạy mọi việc GPU trên MỘT luồng)."""

    def __init__(self, variant: str = "distill", steps: int | None = None, weights_dir: str | None = None,
                 device: str | None = None):
        """device: đặt CẢ BA phần lên một card (máy chủ chạy mỗi card một bản -- server/app.py), bỏ qua TEXTFIX_DEVICE_*."""
        self.variant = variant
        self.name, _, _, self.steps, self.cfg = MODELS[variant]
        self.steps = steps or self.steps
        self.dev_dit = device or os.environ.get("TEXTFIX_DEVICE_DIT", "cuda:0")
        self.dev_te = device or os.environ.get("TEXTFIX_DEVICE_TE", self.dev_dit)
        self.dev_ae = device or os.environ.get("TEXTFIX_DEVICE_AE", self.dev_dit)
        self.weights = find_weights([Path(weights_dir)] if weights_dir else None)
        self.dit = self.ae = self.te = None

    def load(self) -> "Flux":
        import torch
        from flux2 import util
        if self.variant not in self.weights:
            raise FileNotFoundError(f"không thấy {MODELS[self.variant][1]} -- đặt {MODELS[self.variant][2]} hoặc --weights-dir")
        torch.backends.cuda.matmul.allow_tf32 = True
        self.dit = util.load_flow_model(self.name, device=self.dev_dit).eval()
        self.ae = util.load_ae(self.name, device=self.dev_ae).eval()
        self.ae_dtype = next(self.ae.parameters()).dtype
        self.te = util.load_text_encoder(self.name, device=self.dev_te)
        self._unc = self._encode("") if self.cfg else None
        return self

    def check_dtypes(self) -> list[str]:
        """Mọi tham số DiT / bộ mã hoá chữ phải là bf16 (nạp đồng thời nhiều bản từng làm lẫn float32 -> "float != BFloat16"):
        lệch thì ép về bf16 và báo."""
        import torch
        fixed = []
        for name, mod in (("dit", self.dit), ("te", self.te)):
            if mod is not None and any(p.dtype == torch.float32 for p in mod.parameters()):
                mod.to(torch.bfloat16)
                fixed.append(name)
        if fixed:
            print(f"[flux {self.dev_dit}] tham số float32 trong {fixed} -> ép bf16")
        return fixed

    def info(self) -> dict:
        return {"variant": self.variant, "model": self.name, "steps": self.steps, "cfg": self.cfg, "weights": self.weights,
                "devices": {"dit": self.dev_dit, "te": self.dev_te, "ae": self.dev_ae}}

    # ------------------------------------------------------------------------------------------------ mã hoá
    def _encode(self, prompt: str):
        import torch
        from flux2.sampling import prc_txt
        with torch.no_grad():
            ctx = self.te([prompt]).to(torch.bfloat16)[0].to(self.dev_dit)
        ctx, ids = prc_txt(ctx)
        return ctx.unsqueeze(0), ids.unsqueeze(0).to(self.dev_dit)

    def _refs(self, imgs: list[np.ndarray]):
        """Ảnh tham chiếu -> (token, id) như flux2.sampling.encode_image_refs, nhưng VAE ở thiết bị riêng."""
        import torch
        from PIL import Image
        from flux2.sampling import default_prep, prc_img
        lim = 2024 ** 2 if len(imgs) == 1 else 1024 ** 2
        toks, ids = [], []
        with torch.no_grad():
            for k, im in enumerate(imgs):
                t = default_prep(img=Image.fromarray(im), limit_pixels=lim)
                z = self.ae.encode(t[None].to(self.dev_ae, self.ae_dtype))[0]
                x, xi = prc_img(z, t_coord=torch.tensor([10 + 10 * k]))
                toks.append(x)
                ids.append(xi)
        return (torch.cat(toks)[None].to(self.dev_dit, torch.bfloat16), torch.cat(ids)[None].to(self.dev_dit))

    # ------------------------------------------------------------------------------------------------ khử nhiễu
    def _sample(self, ctx, w: int, h: int, seed: int, refs=None, neg: str | None = None) -> np.ndarray:
        """Euler theo lịch flux2.sampling.get_schedule; CFG (bản base): batch [có prompt, prompt rỗng]. neg: prompt âm thay cho
        prompt rỗng (chỉ bản có CFG; distill không có nhánh này nên bỏ qua)."""
        import torch
        from flux2.sampling import get_schedule, prc_img
        dev, cfg = self.dev_dit, self.cfg
        g = torch.Generator(device=dev).manual_seed(int(seed))
        z = torch.randn(1, 128, h // 16, w // 16, generator=g, device=dev, dtype=torch.bfloat16)
        x, x_ids = prc_img(z[0])
        x, x_ids = x.unsqueeze(0), x_ids.unsqueeze(0).to(dev)
        n = x.shape[1]
        ts = get_schedule(num_steps=self.steps, image_seq_len=n)
        nb = 2 if cfg else 1
        c, c_ids = ctx
        if cfg:
            unc = self._encode(neg) if neg else self._unc
            c, c_ids = torch.cat([c, unc[0]]), torch.cat([c_ids, unc[1]])
        with torch.no_grad():
            for t_curr, t_prev in zip(ts[:-1], ts[1:]):
                xi, ii = x, x_ids
                if refs is not None:
                    xi, ii = torch.cat([x, refs[0]], 1), torch.cat([x_ids, refs[1]], 1)
                pred = self.dit(x=xi.repeat(nb, 1, 1), x_ids=ii.repeat(nb, 1, 1),
                                timesteps=torch.full((nb,), t_curr, dtype=x.dtype, device=dev), ctx=c, ctx_ids=c_ids,
                                guidance=torch.full((nb,), cfg or 1.0, dtype=x.dtype, device=dev))[:, :n]
                if cfg:
                    pred = pred[1:2] + cfg * (pred[0:1] - pred[1:2])
                x = (x + (t_prev - t_curr) * pred).to(torch.bfloat16)
            zd = x[0].transpose(0, 1).reshape(1, 128, h // 16, w // 16)
            img = self.ae.decode(zd.to(self.dev_ae, self.ae_dtype)).float()
        out = ((img[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        torch.cuda.empty_cache()
        return out

    def generate(self, prompt: str, w: int, h: int, seed: int, refs: list[np.ndarray] | None = None, neg: str | None = None,
                 **_) -> np.ndarray:
        """B1: nháp. refs: ảnh sản phẩm người dùng tải lên (FLUX.2 nhận nhiều ảnh tham chiếu)."""
        return self._sample(self._encode(prompt), w, h, seed, self._refs(refs) if refs else None, neg)

    def edit(self, img: np.ndarray, prompt: str, seed: int, neg: str | None = None) -> np.ndarray:
        """B2: sửa ảnh theo lời dặn, ảnh vào làm tham chiếu duy nhất; ra cùng cỡ ảnh vào (đã là bội số 16)."""
        H, W = img.shape[:2]
        w, h = W // 16 * 16, H // 16 * 16
        out = self._sample(self._encode(prompt), w, h, seed, self._refs([img]), neg)
        if (w, h) != (W, H):
            from PIL import Image
            out = np.asarray(Image.fromarray(out).resize((W, H), Image.LANCZOS))
        return out
