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

import json
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
        from .browser import Browser
        from .llm import json_call
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
    from . import hisam
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
    out = {"draft": draft, "ink": np.asarray(ink, bool), "timing": T}
    if product is not None:   # chữ in trên sản phẩm khách tải: để bước thiết kế nhận ra các dòng đó trong nháp và GIỮ nguyên
        from .ocr import read_lines
        try:
            out["product_ocr"] = [l["text"] for l in read_lines(product) if sum(ch.isalnum() for ch in l["text"]) >= 2]
        except Exception:   # OCR ảnh sản phẩm hỏng: vẫn còn lời dặn + lựa chọn keep của VLM
            out["product_ocr"] = []
    return out


def design(d: dict, B: dict, vlm, page, product: bool = False, progress=_noop) -> dict:
    """TRẠM THIẾT KẾ: dòng (OCR + nét chữ) -> gỡ mọi dòng (LaMa, MỘT lần) -> VLM điền chữ -> rà lệnh -> trả điểm ảnh nháp cho dòng
    giữ -> dựng -> kiểm -> sửa bằng code -> vòng duyệt (slots.REVIEW_MODE, mặc định tắt). d = draw(...)."""
    from . import lonly, render, slots
    draft, ink, T = d["draft"], d["ink"], dict(d["timing"])
    t0 = time.time()
    progress("đo dòng (OCR + nét chữ)")
    M0 = slots.build(draft, ink, product=product)
    printed = slots.mark_printed(M0, d.get("product_ocr")) if product else []
    geo_log = list(M0.get("geo_log") or [])
    M = lonly.view(M0)
    T["dong"] = time.time() - t0
    t0 = time.time()
    progress("gỡ chữ (LaMa)")
    elog = []
    clean, mask_all = lonly.erase(draft, ink, M, log=elog)
    T["go_chu"] = time.time() - t0

    def base_of(p_, log_=None):
        return lonly.settle(draft, clean, ink, M, p_, log_)

    def ask():
        return lonly.plan(draft, M, B["prompt_en"], B["texts"], vlm, product=product, base=clean)
    t0 = time.time()
    progress("VLM thiết kế")
    raw = ask()
    # KẾ HOẠCH RỖNG (13/10 accept1 f03: VLM trả về không một lệnh nào -> mọi dòng bị bỏ, poster ra không chữ, 0 lỗi): có dòng chữ, có
    # câu khách mà không lệnh nào viết chữ -> gọi lại VLM một lần; vẫn rỗng thì ghi lỗi nặng "empty_plan"
    wrote = lambda p_: any(o.get("kind") == "text" and o.get("html") for o in p_.get("ops") or [])   # noqa: E731
    empty = bool(M["L"] and B["texts"]) and not wrote(raw)
    if empty:
        progress("VLM trả kế hoạch rỗng, gọi lại")
        raw = ask()
    llog = []
    # CÂU TRẢ LỜI CỤT (mix1 p02: 1 lệnh cho 10 dòng -> 9 dòng trống): gọi lại một lần, lấy bản phủ nhiều dòng hơn
    n0 = len(lonly.covered(raw, M))
    if empty:   # đã gọi lại một lần ở trên; ghi câu trả lời thô để truy vì sao rỗng
        llog.append("VLM trả rỗng, câu trả lời thô: " + json.dumps(raw, ensure_ascii=False)[:300])
    elif M["L"] and n0 < lonly.COVER_MIN * len(M["L"]):
        progress("VLM bỏ sót nhiều dòng, gọi lại")
        again = ask()
        n1 = len(lonly.covered(again, M))
        llog.append(f"VLM chỉ viết {n0} / {len(M['L'])} dòng -> gọi lại: {n1} dòng")
        if n1 > n0:
            raw = again
    raw, flog = lonly.fix_ops(raw, M)
    raw, slog = lonly.split_far(raw, M)
    P, log = slots.validate(raw, M, B["texts"], product=product)
    P, glog = lonly.fill_gaps(P, M, B["texts"], product=product)
    P, mlog = lonly.marks_rule(P, M)
    log = llog + flog + slog + log + glog + mlog
    if empty:
        log.insert(0, "VLM trả kế hoạch không có lệnh chữ nào -> gọi lại: " + ("được" if wrote(raw) else "VẪN RỖNG"))
    log[:0] = geo_log   # vùng chữ của bộ dò, ô chữ cong, dòng OCR sót, cụm nét chưa dòng nào nhận, chấm đầu dòng
    if product:
        log.insert(0, f"ảnh sản phẩm khách tải: chữ đọc được {d.get('product_ocr') or []} -> dòng nháp là chữ in trên sản phẩm {printed}")
    T["vlm"] = time.time() - t0
    t0 = time.time()
    progress("dựng + kiểm")
    base, mask = base_of(P, log)
    poster, res, rlog = render.render(page, base, P, M)
    errs = render.check(P, M, res)
    if empty and not wrote(raw):
        errs.append({"type": "empty_plan", "ops": []})
    log += rlog + [f"LỖI: {e}" for e in errs]
    poster_first, errs_first = poster, errs   # bản lượt đầu (debug: so trước / sau vòng sửa)
    sev = lambda E: sum(e["type"] in slots.SEVERE for e in E)   # noqa: E731
    for k in range(render.FIX_ROUNDS if sev(errs) else 0):   # SỬA BẰNG CODE: lỗi nặng đo được -> sửa, dựng lại, giữ khi bớt lỗi
        P2, flog2 = render.autofix(P, M, errs, res)
        if not flog2:
            break
        poster2, res2, rlog2 = render.render(page, base_of(P2)[0], P2, M)
        errs2 = render.check(P2, M, res2)
        ok = (sev(errs2), len(errs2)) < (sev(errs), len(errs))
        log += [f"-- sửa bằng code {k + 1} --"] + flog2 + [f"LỖI CÒN: {e}" for e in errs2] + \
            ["-> giữ bản sửa" if ok else "-> bản sửa không bớt lỗi, giữ bản trước"]
        if not ok:
            break
        poster, P, errs, res = poster2, P2, errs2, res2
        if not sev(errs):
            break
    T["dung"] = time.time() - t0
    t0 = time.time()   # VÒNG DUYỆT (slots.REVIEW_MODE): "errors" -- 1 lượt, chỉ khi có lỗi nặng; "full" -- thẩm mỹ cả tấm
    changes = log[:]   # code đã đổi gì trong quyết định của VLM (chặn chữ, căn cột, cân cỡ...) -- vòng duyệt phải biết
    for k in range(slots.review_rounds() if slots.review_needed(errs) else 0):
        progress(f"VLM duyệt {k + 1}")
        try:
            P2, vlog2 = slots.validate(slots.review(draft, poster, M, P, errs, res, B["prompt_en"], B["texts"], vlm,
                                                    product=product, changes=changes), M, B["texts"], product=product)
        except Exception as e:   # vòng duyệt hỏng (VLM trả sai, quá giờ): giữ bản đang có
            log.append(f"vòng duyệt {k + 1} hỏng: {type(e).__name__}: {str(e)[:200]}")
            break
        if not P2.get("review_patches"):
            log.append(f"vòng duyệt {k + 1}: không sửa gì ({P2.get('review_why')})")
            break
        poster2, res2, rlog2 = render.render(page, base_of(P2)[0], P2, M)
        errs2 = render.check(P2, M, res2)
        log += [f"-- vòng duyệt {k + 1} --", f"VLM: {P2.get('review_why')}"] + vlog2 + rlog2 + [f"LỖI CÒN: {e}" for e in errs2]
        if not slots.accept(P, errs, P2, errs2):   # mất câu khách / thêm lỗi nặng: hoàn lại đúng các lệnh gây hỏng
            P3 = slots.salvage(P, errs, P2, errs2)
            if P3 is not None:
                poster3, res3, rlog3 = render.render(page, base_of(P3)[0], P3, M)
                errs3 = render.check(P3, M, res3)
                log.append(f"-> hoàn lại {P3['salvaged']}, giữ các vá khác")
                if slots.accept(P, errs, P3, errs3):
                    P2, poster2, res2, errs2, rlog2 = P3, poster3, res3, errs3, rlog3
            if P2 is not P3:
                log.append("-> bản duyệt mất câu khách / thêm lỗi nặng, giữ bản trước")
                break
        if slots.REVIEW_MODE == "full":
            better, why = slots.judge(draft, poster, poster2, vlm)
            log.append(f"giám khảo: {'bản duyệt' if better else 'bản trước'} đẹp hơn ({why})")
        else:   # chế độ errors: không gọi giám khảo, rào đo lường (accept) đã qua là giữ
            better = True
        if not better:
            break
        poster, P, errs, res, changes = poster2, P2, errs2, res2, vlog2 + rlog2
        log.append("-> giữ bản đã duyệt")
    T["duyet"] = time.time() - t0
    tid = lambda m: int(m[1:]) if isinstance(m, str) and m[1:].isdigit() else m   # noqa: E731  "T5" -> 5
    miss_i = [i for i in map(tid, P.get("missing") or []) if isinstance(i, int) and 0 <= i < len(B["texts"])]
    code_miss = [i for i in slots.uncovered(P, B["texts"]) if i not in miss_i]   # VLM bỏ câu mà không khai
    code_miss = [i for i in code_miss if B["texts"][i].get("source") != "ai"]    # chữ phụ trợ (không phải lời khách) được phép bỏ
    if code_miss:
        log.append(f"câu khách không có trên poster (đo bằng mã): {[B['texts'][i]['text'] for i in code_miss]}")
    missing = [B["texts"][i]["text"] for i in sorted(set(miss_i) | set(code_miss))]
    base, mask = base_of(P)   # nền của kế hoạch cuối (vòng sửa / duyệt có thể đã đổi dòng giữ)
    kept = lonly.kept_of(P)
    plan = {"style": P.get("style"), "ops": P["ops"], "missing": missing, "notes": P.get("notes"), "errors": errs, "log": log,
            "slots": slots.public(M), "slots_mode": "L", "errors_first": errs_first,
            "erase": {"mask_pct": round(100 * float(mask.mean()), 2), "ink_pct": round(100 * float(ink.mean()), 2),
                      "kept": sorted(kept), "orphans": [l["id"] for l in M["L"] if l.get("orphan")], "log": elog}}
    ink_img = lonly.ink_image(draft, ink, mask, M, kept)
    steps = _row([draft, slots.marked_image(draft, M), ink_img, base, poster])
    return {"draft": draft, "ink": ink, "ink_image": ink_img, "hisam_image": lonly.hisam_image(draft, ink), "clean": clean, "base": base, "mask": mask, "poster": poster,
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
