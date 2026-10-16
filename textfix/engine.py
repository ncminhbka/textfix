"""ENGINE end to end: form / prompt -> poster. TextFix CHỈ SỬA CHỮ: poster = nháp với chữ viết lại cho rõ.

  with Engine(flux) as E:                                    # flux = flux.Flux("distill").load() (None: chỉ fix() nháp có sẵn)
      B = E.brief(prompt=..., form=..., design=...)          # LLM: prompt FLUX + câu khách
      r = E.make(B, W, H, seed)                              # FLUX vẽ nháp -> fix()
      r = E.fix(draft, B, seed, ink=None)                    # nét chữ (Hi-SAM) -> dòng -> gỡ chữ (LaMa) -> VLM điền -> dựng -> kiểm
  r: {draft, ink (mặt nạ nét chữ), clean (nền gỡ mọi dòng), base (nền dựng), mask (mặt nạ gỡ chữ), poster, poster_first,
      steps (ảnh 5 cột), plan (JSON: dòng, lệnh, lỗi, nhật ký, câu thiếu), timing}
  B["mode"] == "image" (không chữ khách): make() chỉ FLUX vẽ -> r = {draft, poster (= nháp), plan {mode}, timing}

Hai trạm dùng riêng được (máy chủ chạy dây chuyền nhiều phiên bản song song, server/app.py):
  d = draw(flux, B, W, H, seed, product)          # GPU: nháp -> nét chữ
  r = design(d, B, vlm, page, product=...)        # dòng -> gỡ chữ -> VLM -> rà lệnh -> dựng -> kiểm

14/10: không còn bản xoá FLUX và bước dọn chữ sót (đường đó: nhánh git `legacy-flux-erase`). Dây chuyền: docs/ROADMAP.md.
"""

from __future__ import annotations

import time

import numpy as np

from . import config


class Engine:
    def __init__(self, flux=None, progress=None, llm=None, vlm=None):
        """llm / vlm: hàm gọi tự truyền (thử nghiệm, VLM giả); None: lấy từ biến môi trường (config.models)."""
        self.flux = flux
        self.progress = progress or (lambda msg: None)
        self.llm, self.vlm = llm, vlm

    def __enter__(self) -> "Engine":
        from .brief import SCHEMA
        from .llm import json_call
        from .render.browser import Browser
        config.load_env()
        self.models = config.models()
        m = self.models
        if self.llm is None:
            self.llm = json_call(m["llm"]["url"], m["llm"]["model"], m["llm"]["key_env"], tag="brief", schema=SCHEMA)
        if self.vlm is None:
            self.vlm = json_call(m["vlm"]["url"], m["vlm"]["model"], m["vlm"]["key_env"], tag="designer")
        self.browser = Browser().__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self.browser.__exit__(*exc)

    def brief(self, prompt: str | None = None, form: dict | None = None, design: dict | None = None,
              product_image: bool = False, n_variants: int = 0, aspect: str | None = None) -> dict:
        from .brief import make_brief
        return make_brief(self.llm, prompt, form, design, product_image, n_variants, aspect=aspect)

    def make(self, B: dict, W: int, H: int, seed: int, product: np.ndarray | None = None) -> dict:
        d = draw(self.flux, B, W, H, seed, product, progress=self.progress)
        if B.get("mode") == "image":
            return {"draft": d["draft"], "poster": d["draft"], "plan": {"mode": "image"},
                    "timing": {k: round(v, 1) for k, v in d["timing"].items()}}
        return design(d, B, self.vlm, self.browser.page, product=product is not None, progress=self.progress)

    def fix(self, draft: np.ndarray, B: dict, seed: int = 0, ink: np.ndarray | None = None, product: bool = False) -> dict:
        """product: người dùng tải ảnh sản phẩm (chữ / logo in trên sản phẩm trong nháp -> designer "keep")."""
        d = draw(self.flux, B, draft.shape[1], draft.shape[0], seed, draft=draft, ink=ink, progress=self.progress)
        return design(d, B, self.vlm, self.browser.page, product=product, progress=self.progress)


# ---------------------------------------------------------------------------------------------------- hai trạm
# Máy chủ chạy song song nhiều phiên bản theo DÂY CHUYỀN (server/app.py): trạm VẼ trên luồng của từng GPU (mỗi card một FLUX +
# một Hi-SAM), trạm THIẾT KẾ trên nhóm luồng riêng (OCR, đo dòng, LaMa, VLM ở máy khác, Chromium mỗi luồng một trình duyệt).
def _noop(msg: str) -> None:
    pass


def draw(flux, B: dict, W: int, H: int, seed: int, product: np.ndarray | None = None, draft: np.ndarray | None = None,
         ink: np.ndarray | None = None, progress=_noop) -> dict:
    """TRẠM VẼ (GPU): FLUX vẽ nháp -> Hi-SAM tìm nét chữ. draft / ink có sẵn thì bỏ bước tương ứng.
    -> {draft, ink (bool cỡ nháp), timing}."""
    from .draw import hisam
    T = {}
    if draft is None:
        progress("FLUX vẽ nháp")
        t0 = time.time()
        draft = flux.generate(B["prompt_en"], W, H, seed, refs=[product] if product is not None else None)
        T["ve_nhap"] = time.time() - t0
    if B.get("mode") == "image":   # ảnh thường (không chữ khách): FLUX vẽ là xong
        return {"draft": draft, "timing": T}
    if ink is None:
        progress("Hi-SAM tìm nét chữ")
        t0 = time.time()
        ink = hisam.strokes(draft, getattr(flux, "dev_dit", None))   # cùng card với bản FLUX của luồng vẽ này
        T["net_chu"] = time.time() - t0
    out = {"draft": draft, "ink": np.asarray(ink, bool), "timing": T,
           "device": getattr(flux, "dev_dit", None)}   # card của luồng vẽ: LaMa ở trạm thiết kế chạy cùng card (chia tải hai card)
    if product is not None:   # chữ in trên sản phẩm khách tải: để bước thiết kế nhận ra các dòng đó trong nháp và GIỮ nguyên
        from .lines.ocr import read_lines
        try:
            out["product_ocr"] = [l["text"] for l in read_lines(product) if sum(ch.isalnum() for ch in l["text"]) >= 2]
        except Exception:   # OCR ảnh sản phẩm hỏng: vẫn còn lời dặn + lựa chọn keep của VLM
            out["product_ocr"] = []
    return out


def design(d: dict, B: dict, vlm, page, product: bool = False, progress=_noop) -> dict:
    """TRẠM THIẾT KẾ (chặng B, 14/10): dòng (OCR + nét chữ) -> gỡ mọi dòng (LaMa, MỘT lần) -> VLM GHÉP câu khách <-> dòng + kiểu
    theo vai -> code chép nguyên văn, cỡ / lề / tập theo nháp (compose.layout) -> trả điểm ảnh nháp cho dòng giữ -> dựng -> kiểm -> sửa
    bằng code. d = draw(...)."""
    from . import erase, lines, render
    from .compose import fate, rows
    from .compose import vlm as cvlm
    from .compose.assign import resolve
    from .compose.layout import layout, public
    from .lines.view import _typo_notes
    draft, ink, T = d["draft"], d["ink"], dict(d["timing"])
    texts = B["texts"]
    t0 = time.time()
    progress("đo dòng (OCR + nét chữ)")
    M0 = lines.build(draft, ink, product=product)
    printed = lines.mark_printed(M0, d.get("product_ocr")) if product else []
    if printed:
        _typo_notes(M0["L"], M0.get("geo_log") or [])   # ghi lại phần "chữ khó" cho dòng in trên sản phẩm
    geo_log = list(M0.get("geo_log") or [])
    M = cvlm.view(M0)
    T["dong"] = time.time() - t0
    t0 = time.time()
    progress("gỡ chữ (LaMa)")
    elog = []
    clean, mask_all = erase.erase(draft, ink, M, device=d.get("device"), log=elog)
    T["go_chu"] = time.time() - t0

    def base_of(U_, log_=None):
        return erase.settle(draft, clean, ink, M, {"ops": U_}, log_)

    t0 = time.time()
    progress("VLM ghép câu")
    # MỘT lời gọi, khuôn JSON bắt buộc mỗi dòng một mục (compose.vlm.schema): không hỏi lại dòng bỏ sót / dòng không còn chữ /
    # ghép rỗng (người dùng chốt 16/10). Ghép rỗng -> lỗi nặng "empty_plan".
    A = cvlm.ask(draft, M, B["prompt_en"], texts, vlm, product=product, base=clean)
    wrote = any(u["kind"] == "text" and (u["clients"] or u["text_vlm"]) for u in A.get("assign") or [])
    empty = bool(M["L"] and texts) and not wrote
    llog = []
    if A.get("dropped"):
        llog.append(f"bỏ mục ghép sai khuôn: {A['dropped']}")
    U, rlog0, miss_i = resolve(A, M, texts)
    U, llay = layout(U, M, ink, A["styles"], A["accent"], texts, base=clean)
    log = geo_log + llog + rlog0 + llay
    if product:
        log.insert(0, f"ảnh sản phẩm khách tải: chữ đọc được {d.get('product_ocr') or []} -> dòng nháp là chữ in trên sản phẩm {printed}")
    T["vlm"] = time.time() - t0
    t0 = time.time()
    progress("dựng + kiểm")
    base, mask = base_of(U, log)
    base, slog = rows.respace(base, U, M)   # danh sách có hàng thừa: dàn đều các hàng còn lại (dời dải nền + khung chữ)
    log += slog
    poster, res, rlog = render.render_units(page, base, U, M)
    errs = render.check_units(U, res)
    if empty:
        errs.append({"type": "empty_plan", "ops": []})
    log += rlog + [f"LỖI: {e}" for e in errs]
    poster_first, errs_first = poster, errs   # bản lượt đầu (debug: so trước / sau khi mã sửa lỗi)
    sev = lambda E: sum(e["type"] in render.SEVERE for e in E)   # noqa: E731
    for k in range(render.FIX_ROUNDS if sev(errs) else 0):   # SỬA BẰNG CODE: lỗi nặng đo được -> sửa, dựng lại, giữ khi bớt lỗi
        U2, flog2 = render.autofix_units(U, errs, res)
        if not flog2:
            break
        poster2, res2, rlog2 = render.render_units(page, base, U2, M)
        errs2 = render.check_units(U2, res2)
        ok = (sev(errs2), len(errs2)) < (sev(errs), len(errs))
        log += [f"-- sửa bằng code {k + 1} --"] + flog2 + [f"LỖI CÒN: {e}" for e in errs2] + \
            ["-> giữ bản sửa" if ok else "-> bản sửa không bớt lỗi, giữ bản trước"]
        if not ok:
            break
        poster, U, errs, res = poster2, U2, errs2, res2
        if not sev(errs):
            break
    T["dung"] = time.time() - t0
    vlm_miss = [k for k in A.get("missing") or [] if 0 <= k < len(texts)]
    if sorted(vlm_miss) != sorted(miss_i):
        log.append(f"câu khách không có trên poster: VLM khai {['T%d' % k for k in vlm_miss]}, đo bằng mã {['T%d' % k for k in miss_i]}")
    missing = [texts[i]["text"] for i in miss_i]
    kept = erase.kept_of({"ops": U})
    # DỰNG THẾ NÀO, từng phần tử (debug "chữ khó"): cỡ, co, nén ngang, xoay, theo đường cong
    rinfo = [{"id": r["id"], "unit": r.get("unit"), "fs": round(float(r.get("fs") or r.get("size") or 0), 1),
              "shrink": round(float(r.get("shrink") or 1), 2), "sx": r.get("sx", 1), "angle": round(float(r.get("angle") or 0), 1),
              "path": bool(r.get("path"))} for r in res or []]
    plan = {"style": A["styles"], "assign": A["assign"], "accent": A["accent"], "ops": public(U), "fates": fate.table(U, M),
            "render": rinfo,
            "missing": missing, "notes": A.get("notes"), "errors": errs, "log": log, "slots": lines.public(M), "slots_mode": "L",
            "errors_first": errs_first,
            "erase": {"mask_pct": round(100 * float(mask.mean()), 2), "ink_pct": round(100 * float(ink.mean()), 2),
                      "kept": sorted(kept), "orphans": [l["id"] for l in M["L"] if l.get("orphan")], "log": elog}}
    ink_img = erase.ink_image(draft, ink, mask, M, kept)
    steps = _row([draft, lines.marked_image(draft, M), ink_img, base, poster])
    return {"draft": draft, "ink": ink, "ink_image": ink_img, "hisam_image": erase.hisam_image(draft, ink), "clean": clean, "base": base, "mask": mask, "poster": poster,
            "poster_first": poster_first, "steps": steps, "plan": plan, "timing": {k: round(v, 1) for k, v in T.items()}}


def _row(ims, h: int = 640) -> np.ndarray:
    """nháp | dòng | nét chữ + mặt nạ | nền đã gỡ | poster."""
    from PIL import Image
    ims = [Image.fromarray(np.asarray(x)).convert("RGB") for x in ims]
    ims = [im.resize((int(im.width * h / im.height), h)) for im in ims]
    row = Image.new("RGB", (sum(im.width + 10 for im in ims), h), "white")
    x = 0
    for im in ims:
        row.paste(im, (x, 0))
        x += im.width + 10
    return np.asarray(row)
