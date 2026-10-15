"""TextFix: poster tiếng Việt từ FLUX.2-klein -- CHỈ SỬA CHỮ: nháp giữ nguyên, chữ giun được gỡ và viết lại đúng câu của khách.

Luồng (textfix.engine.Engine; docs/ROADMAP.md), mỗi bước một gói:
  form / prompt --LLM--> prompt FLUX + câu khách          brief/    (make_brief)
  FLUX.2-klein vẽ nháp, Hi-SAM tìm nét chữ                 draw/     (flux.Flux.generate / remote.RemoteFlux, hisam.strokes)
  đo dòng chữ: OCR + chữ cong + cụm nét OCR không đọc       lines/    (lines.build)
  gỡ chữ: mặt nạ nét + quầng, LaMa lấp một lần              erase/    (erase.erase / erase.settle)
  VLM ghép câu khách <-> dòng + kiểu theo vai               compose/  (vlm.ask, parse, fate, assign.resolve)
  code chép nguyên văn, cỡ / lề / tập theo nháp             compose/  (layout.layout)
  dựng, kiểm, sửa bằng code                                 render/   (render_units / check_units / autofix_units)
Công cụ vận hành (không thuộc dây chuyền): tools/.
"""
