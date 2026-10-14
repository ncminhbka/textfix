# Bàn giao, cập nhật 14/10 (khuya)

Chi tiết từng việc nằm trong `docs/ROADMAP.md` và `docs/TUNING.md`. Tệp này chỉ là mục lục để phiên sau bắt đầu nhanh.

## Đã chốt

- **Chặng A** (gỡ chữ bằng Hi-SAM + LaMa): người dùng ưng 100%. Không động vào.
- **Chặng B**: VLM ghép câu khách với dòng nháp; code chép nguyên văn câu khách và dựng theo nháp.
  - Nguyên tắc "trần": mọi lỗi còn lại phải là lỗi của VLM, FLUX hoặc LLM.
  - Không vá luật theo từng ca. Nếu thấy hướng không ổn thì dừng và báo người dùng.

## Hôm nay (đã push lên main)

Các commit, từ cũ đến mới:

| Commit | Nội dung |
|---|---|
| `7c62a44` | Ngắt dòng theo bề rộng nét và font thật; không ép câu thiếu vào dòng thừa; giữ màu nháp |
| `0f2bedd` | Sau dev2: brief không làm mất câu khách; VLM thắng luật chữ sản phẩm; viết lại số trước các hàng; gửi lại chữ OCR cho VLM làm neo |
| `339106f` | Thêm bộ thi `bench/final2.json` (32 ca mới) |
| `266fd31`, `223033c`, `0aab432` | Bước LLM: mọi kiểu ngoặc thành ngoặc kép, mỗi chữ chỉ một lần, lời dặn theo đúng chuẩn của BFL |
| `a6236df` | ROADMAP ghi nhận lỗi của bước LLM và hướng đề xuất |

## Đang chờ người dùng

1. **Hướng làm lại bước LLM** (xem `ROADMAP.md`, mục "Bước LLM đầu luồng").
   - Đề xuất: LLM chỉ tả cảnh; code tự nối danh sách chữ vào cuối prompt, đúng một lần.
   - Câu hỏi mở: có ghi vai trước mỗi câu không.
2. **Lỗi chữ trên sản phẩm bị VLM viết đè** (dev2 d13). Hướng định vị sản phẩm bằng hình học đã ghi trong ROADMAP, chưa làm.
3. **Audit tiếp**: lời dặn VLM (`lonly.SYSTEM_L`), rồi bước dựng và kiểm (`render.py`).
4. **Chưa thi.** Khi người dùng cho thi:
   - chạy 5 ca kiểm tra: `dev_batch.py bench/dev2.json --only d03,d05,d20,d31,d39 --fresh`;
   - rồi chạy `bench/final2.json`, so cùng seed với nhánh `legacy-ops-html`.

## Ghi nhận, sau deadline

- Thay cụm luật vá OCR trong `slots.py` bằng dòng Hi-SAM phân cấp.
- Ngưỡng `OUTLINE_*` có hại (phá hiệu ứng neon ở dev2 d06).
