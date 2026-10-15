"""TextFix: poster tiếng Việt từ FLUX.2-klein -- CHỈ SỬA CHỮ: nháp giữ nguyên, chữ giun được gỡ và viết lại đúng câu của khách.

Luồng (textfix.engine.Engine; docs/ROADMAP.md):
  form / prompt --LLM--> prompt FLUX + câu khách        (brief)
  FLUX.2-klein vẽ nháp                                   (flux.Flux.generate)
  Hi-SAM tìm nét chữ                                     (hisam.strokes)
  đo dòng chữ: OCR + chữ cong + cụm nét OCR không đọc     (slots.build)
  gỡ chữ: mặt nạ nét + quầng, LaMa lấp một lần            (erase.erase / erase.settle)
  VLM ghép câu khách <-> dòng + kiểu theo vai             (lonly.plan / fill.parse)
  code chép nguyên văn, cỡ / lề / tập theo nháp -> dựng   (fill.resolve / fill.layout / render.render_units / check_units)
"""
