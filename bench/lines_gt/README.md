# Bộ đáp án chia dòng (`bench/lines_gt/`)

60 nháp FLUX (dev2, mix1, accept2, lượt 14/10), 540 dòng. Dùng để đo bước đo dòng (`scripts/lines_bench.py`), không dùng để
chỉnh ngưỡng theo từng ca. Nháp + mặt nạ Hi-SAM nằm trong các zip báo cáo cùng lượt (`<tập>_report_1014_*.zip`), khoá `src`.

Cách gán (Claude, 15/10): điền sẵn từ dòng của mã lúc đó, xem từng ảnh (nháp + nét Hi-SAM + khung) và sửa: tách / gộp, kiểu
đường, loại, thêm dòng bị sót, bỏ thứ không phải chữ. Chọn: mọi nháp có chữ khó (35) + 25 nháp thường ngẫu nhiên.

Mỗi tệp: `{key, src, size, status, lines: [...]}`; mỗi dòng:

| Trường | Nghĩa |
|---|---|
| `box` | khung dòng (x0, y0, x1, y1), ôm chữ, KHÔNG ôm icon / tick / dấu trang trí đứng trước |
| `shape` | `straight` (ngang) / `tilt` (nghiêng thẳng) / `curve` (cong, cung) / `vertical` (chữ dọc) |
| `kind` | `poster` (chữ poster: phải viết lại) / `object` (chữ trên vật: nhãn, lon, màn hình, con dấu sản phẩm) / `junk` (cụm nét rác) |
| `rows` | số hàng chữ trong dòng (luôn 1: mỗi hàng là một dòng) |
| `eval` | `false` = không tính điểm (chữ chạy vòng kín quanh con dấu / huy chương: giới hạn đã biết) |
| `ocr` | chữ OCR đọc lúc điền sẵn (tham khảo, không phải đáp án chữ) |

Quy ước: hai mục cách nhau một khe rõ trên cùng hàng (tên món | giá; số điện thoại | địa chỉ | website) là hai dòng; một từ /
một câu OCR cắt thành mẩu là một dòng; tiêu đề xếp chồng mỗi hàng một dòng; chữ cùng một câu xuống hàng là các dòng riêng.

**Gán lại mù kiểu đường (15/10).** Nhãn `shape` điền sẵn từ mã cũ nên thiên về mã cũ (chữ trên đồng xu nghiêng 8° vẫn ghi
`straight`). 50 dòng mà đáp án / mã cũ / phép đo nét không cùng một kiểu được gán lại từ ảnh cắt có khung, xáo thứ tự, KHÔNG xem
phương án nào nói gì: 13 dòng đổi kiểu, 2 mẩu nét không đọc được chuyển `eval: false`.
