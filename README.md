# TextFix

Poster tiếng Việt từ FLUX.2-klein. Model vẽ tốt bố cục, hình ảnh, vỏ và icon, nhưng chữ thành "giun". TextFix **chỉ sửa chữ**:
giữ nguyên mọi điểm ảnh khác của nháp, gỡ chữ giun và viết lại đúng câu của khách vào đúng chỗ model đã vẽ.

## Luồng

```
form / prompt
  -> LLM: prompt FLUX.2 (theo hướng dẫn nâng prompt của BFL) + câu của khách          textfix/brief.py
  -> FLUX.2-klein vẽ nháp                                                            textfix/flux.py
  -> Hi-SAM tìm nét chữ của nháp (mặt nạ từng điểm ảnh)                             textfix/hisam.py
  -> đo DÒNG chữ: OCR + chữ cong + cụm nét OCR không đọc                              textfix/slots.py, geo.py
  -> gỡ chữ: mặt nạ nét + quầng, LaMa lấp một lần                                     textfix/lonly.py
  -> VLM designer điền chữ theo dòng: viết / giữ (chữ in trên sản phẩm) / xoá         textfix/lonly.py, slots.py
  -> rà lệnh, dựng chữ đúng khung dòng trên nền (Chromium), kiểm lỗi, sửa bằng code   textfix/render.py
  -> poster
```

Lộ trình và tình trạng hiện tại: [docs/ROADMAP.md](docs/ROADMAP.md). [docs/DESIGN.md](docs/DESIGN.md) tả thiết kế trước 14/10 (ô vỏ / ô chi tiết, bản xoá FLUX): chỉ còn giá trị tham khảo.

## Chạy

Máy chủ GPU (2x A30, JupyterLab): [docs/SERVER.md](docs/SERVER.md).

```bash
pip install -r requirements.txt
export TEXTFIX_LLM_URL=http://<host>:<port>/v1 TEXTFIX_MODEL=<model nhìn được ảnh> TEXTFIX_LLM_API_KEY=<khoá>
python scripts/check_server.py --fix        # kiểm môi trường, phải in "SẴN SÀNG."
bash scripts/start_server.sh                # giao diện ở cổng 8088
python scripts/probe.py                     # thử engine trên bench/prompts_C.json, nhìn output/probe/*.jpg
```

Chạy trên máy chủ bằng ô notebook (`%run scripts/...`): kết quả tự gói thành zip < 24 MB và tự tải về ([docs/SERVER.md](docs/SERVER.md)).

## Cấu trúc

| Đường dẫn | Việc |
|---|---|
| `textfix/engine.py` | Nối toàn bộ luồng: `Engine.brief`, `make` (vẽ + sửa), `fix` (sửa nháp có sẵn) |
| `textfix/brief.py` | LLM đầu luồng: prompt FLUX + câu khách, code giữ cam kết không bịa chữ |
| `textfix/flux.py` | FLUX.2-klein: vẽ nháp |
| `textfix/hisam.py` | Hi-SAM: mặt nạ nét chữ của nháp |
| `textfix/lonly.py` | Đường chỉ dòng chữ: gỡ chữ (mặt nạ + LaMa), lời dặn VLM, rà lệnh |
| `textfix/slots.py` | Đo dòng chữ, gọi VLM, chặn chữ sai, vòng duyệt |
| `textfix/render.py` | Dựng HTML đúng khung ô (thẻ tiện ích icon / sao / font), kiểm lỗi |
| `textfix/ocr.py`, `glyph.py`, `geo.py` | OCR dòng, đo nét (cỡ, màu), chữ cong |
| `textfix/fonts.py` | Bộ font OFL đủ dấu tiếng Việt |
| `textfix/ship.py` | Gói kết quả máy chủ thành zip < 24 MB, tự tải về trình duyệt (JupyterLab) |
| `server/` | Máy chủ FastAPI + giao diện form |
| `scripts/dev_batch.py`, `batch_report.py` | Chạy một tập qua máy chủ, báo cáo + một zip |
| `scripts/lonly_replay.py`, `erase_bench.py` | Máy cá nhân: dựng lại poster / đo bước gỡ chữ trên nháp đã lưu |
| `tools/annotate.html` | Công cụ gán nhãn ô S / I / L / P (Chrome / Edge, ghi thẳng vào repo) |
| `flux2/` | Mã suy luận FLUX.2 của Black Forest Labs (Apache-2.0), nguyên trạng |
| `assets/` | Font (OFL) |

## Giấy phép tài nguyên

- `flux2/`: Apache-2.0 (Black Forest Labs), xem `flux2/LICENSE.md`.
- Font trong `assets/fonts/`: SIL Open Font License.
- Icon trong `assets/icons/lucide/`: ISC (Lucide).
