"""TextFix: poster tiếng Việt từ FLUX.2-klein -- CHỈ SỬA CHỮ: nháp giữ nguyên, chữ giun được gỡ và viết lại đúng câu của khách.

Luồng (textfix.engine.Engine; docs/ROADMAP.md):
  form / prompt --LLM--> prompt FLUX + câu khách        (brief)
  FLUX.2-klein vẽ nháp                                   (flux.Flux.generate)
  Hi-SAM tìm nét chữ                                     (hisam.strokes)
  đo dòng chữ: OCR + chữ cong + cụm nét OCR không đọc     (slots.build)
  gỡ chữ: mặt nạ nét + quầng, LaMa lấp một lần            (lonly.erase / lonly.settle)
  VLM điền chữ theo dòng -> rà lệnh -> dựng -> kiểm       (lonly.plan / slots.validate / render.render / render.check)
"""
