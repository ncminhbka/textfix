# TextFix

Poster tiếng Việt từ FLUX.2-klein. Model vẽ tốt bố cục, hình ảnh, vỏ và icon, nhưng chữ thành "giun". TextFix **chỉ sửa chữ**:
giữ nguyên mọi điểm ảnh khác của nháp, gỡ chữ giun và viết lại đúng câu của khách vào đúng chỗ model đã vẽ.

## Luồng

```
form / prompt
  -> LLM: prompt FLUX.2 (theo hướng dẫn nâng prompt của BFL) + câu của khách          textfix/brief/
  -> FLUX.2-klein vẽ nháp                                                            textfix/flux.py
  -> Hi-SAM tìm nét chữ của nháp (mặt nạ từng điểm ảnh)                             textfix/hisam.py
  -> đo DÒNG chữ: OCR chia vùng dòng; hướng, cao chữ, màu, nét đo trên nét Hi-SAM      textfix/slots.py, lines/, geo.py
  -> gỡ chữ: mảnh nét của từng dòng + quầng, LaMa lấp một lần                          textfix/erase.py
  -> VLM đọc nháp: dòng nào là câu nào; số phận mỗi dòng viết / giữ / xoá             textfix/compose/
  -> bố cục: cỡ theo font thật, nới vào nền trống, tập cùng vai, màu                   textfix/fill.py
  -> dựng chữ trên nền (Chromium), kiểm lỗi, sửa bằng code                            textfix/render.py
  -> poster
```

Lộ trình và tình trạng hiện tại: [docs/ROADMAP.md](docs/ROADMAP.md). Nhật ký tune, số đo, hướng đã thử và bỏ: [docs/TUNING.md](docs/TUNING.md). Thiết kế cũ (ô vỏ / ô chi tiết, bản xoá FLUX) nằm ở các nhánh `legacy-slots-si`, `legacy-flux-erase`.

## Chạy

Máy chủ GPU (2x A30, JupyterLab): [docs/SERVER.md](docs/SERVER.md).

```bash
pip install -r requirements.txt
export TEXTFIX_LLM_URL=http://<host>:<port>/v1 TEXTFIX_MODEL=<model nhìn được ảnh> TEXTFIX_LLM_API_KEY=<khoá>
python scripts/check_server.py --fix        # kiểm môi trường, phải in "SẴN SÀNG."
bash scripts/start_server.sh                # giao diện ở cổng 8088
python scripts/dev_batch.py bench/accept2.json   # chạy một tập qua máy chủ đang bật, báo cáo + một zip
```

Chạy trên máy chủ bằng ô notebook (`%run scripts/...`): kết quả tự gói thành zip < 24 MB và tự tải về ([docs/SERVER.md](docs/SERVER.md)).

## Cấu trúc

| Đường dẫn | Việc |
|---|---|
| `textfix/engine.py` | Nối toàn bộ luồng: `Engine.brief`, `make` (vẽ + sửa), `fix` (sửa nháp có sẵn) |
| `textfix/brief/` | LLM đầu luồng: danh sách chữ + prompt FLUX; code kiểm theo bằng chứng (không số bịa, chữ khách luôn có, prompt vẽ đúng danh sách). Test: `pytest tests/test_brief.py` |
| `textfix/flux.py` | FLUX.2-klein: vẽ nháp |
| `textfix/hisam.py` | Hi-SAM: mặt nạ nét chữ của nháp |
| `textfix/erase.py` | Gỡ chữ giun: mặt nạ từ nét Hi-SAM (nét, dự phòng, quầng, dấu câu) + LaMa một lượt. Test: `pytest tests/test_erase.py` |
| `textfix/slots.py` | Dòng chữ: OCR chia vùng dòng, chữ cong (`geo.py`), chữ dọc, cụm nét lạ |
| `textfix/lines/` | Hình học + số đo dòng trên nét Hi-SAM: hướng (nét + OCR), cao chữ, đường chân, màu, trang trí đầu dòng, hàng nhiều cỡ. Test: `pytest tests/test_lines.py` |
| `textfix/compose/` | VLM đọc nháp (`vlm.py`, hỏi lại dòng bỏ sót) + một bảng số phận dòng (`fate.py`). Test: `pytest tests/test_fate.py` |
| `textfix/fill.py` | Ghép câu vào dòng + bố cục (cỡ theo font, nới khung, tập, màu). Test: `pytest tests/test_layout.py` |
| `textfix/render.py` | Dựng HTML đúng khung dòng (thẻ font), kiểm lỗi |
| `textfix/ocr.py`, `glyph.py`, `geo.py` | OCR dòng, đo nét (cỡ, màu), chữ cong |
| `textfix/fonts.py` | Bộ font OFL đủ dấu tiếng Việt |
| `textfix/ship.py` | Gói kết quả máy chủ thành zip < 24 MB, tự tải về trình duyệt (JupyterLab) |
| `server/` | Máy chủ FastAPI + giao diện form |
| `scripts/dev_batch.py`, `batch_report.py` | Chạy một tập qua máy chủ, báo cáo + một zip |
| `scripts/replay.py`, `erase_bench.py`, `lines_bench.py` | Máy cá nhân: dựng lại poster / đo bước gỡ chữ / đo bước đo dòng (`bench/lines_gt`) trên nháp đã lưu |
| `flux2/` | Mã suy luận FLUX.2 của Black Forest Labs (Apache-2.0), nguyên trạng |
| `assets/` | Font (OFL) |

## Giấy phép tài nguyên

- `flux2/`: Apache-2.0 (Black Forest Labs), xem `flux2/LICENSE.md`.
- Font trong `assets/fonts/`: SIL Open Font License.
