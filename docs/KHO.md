# Kho ảnh tham chiếu (culture)

FLUX không biết nhiều món / trang phục / vật Tết của Việt Nam: "bún thang" thành phở bò. Kho chỉ nhận khái niệm model gần như
chắc chắn không biết (thịt chó, mắm tôm, bánh đúc...), không nhận món phổ biến model đã vẽ đúng (phở, bánh mì). Khi lời khách (prompt + các ô form)
có tên một khái niệm trong kho, ảnh mẫu của khái niệm đó được đưa vào FLUX như ảnh tham chiếu, và LLM viết prompt gọi đúng
món là "the bun thang from image 2".

- Code tìm **ứng viên** bằng **đúng từ khoá** (nguyên từ, không phân biệt hoa thường), không embedding. Khách gõ có dấu thì
  phải trùng cả dấu ("cơm" không phải "cốm"); gõ không dấu thì so với dạng bỏ dấu. Không đặt cụm chung chung ("phở", "bò",
  "cơm", "đèn lồng"): "phở gà" không được kéo ảnh phở bò.
- **Code + LLM cùng quyết** (20/10, `kho.match_detail`, `brief._pick_refs`):
  - ứng viên **chắc** (cụm >= 2 từ, khách gõ có dấu, trùng nguyên văn: "bún thang", "cốm làng Vòng"): luôn dùng; prompt không
    nhắc "image N" (LLM quên) thì code chèn câu nhắc sau câu đầu (`brief/prompt.py`, `_cite_refs`);
  - ứng viên **không chắc** (alias một từ: "cốm" trong "trà sữa cốm"; gõ không dấu: "com tam", "banh tet"): LLM được dặn tự
    quyết, chỉ dùng khi prompt chính hoặc một hướng thiết kế nhắc "image N". LLM không thêm được ảnh ngoài ứng viên.
  - bỏ ảnh thì code đánh số lại "image N" trong mọi prompt cho khớp thứ tự ảnh gửi đi (`prompt.renumber`).
- **Một ảnh mỗi khái niệm.** Ảnh sản phẩm khách tải luôn là image 1; ảnh kho đi sau, theo **thứ tự xuất hiện** trong lời khách.
  FLUX.2 klein (4B và 9B) nhận tối đa 4 ảnh tham chiếu; khái niệm thừa bị bỏ.
- Id kho đã dùng: `brief.json` / `plan.json` / kết quả API (`brief.kho`), `log.kho` ghi "tên = image N"; `log.kho_sure`,
  `log.kho_llm_kept`, `log.kho_llm_dropped` ghi ứng viên chắc / LLM giữ / LLM bỏ (kèm chữ khách gõ ở chỗ khớp).
- Bước thiết kế không coi chữ trong ảnh kho là chữ sản phẩm (chỉ ảnh khách tải mới được "keep"): FLUX lỡ chép chữ từ ảnh kho
  thì chữ đó là chữ giun như mọi chữ thừa khác.

## Ảnh ở đâu

Ảnh **không vào git**. Kho nằm trên kho đối tượng S3 (MinIO nội bộ), máy chủ đồng bộ về thư mục đệm lúc khởi động
(`textfix/kho_sync.py`), `textfix/kho.py` đọc thư mục đệm.

| Biến môi trường | Ý nghĩa |
|---|---|
| `TEXTFIX_KHO_ENDPOINT` | địa chỉ **API S3** (`http://host:cổng`; web console thường là cổng khác). Trống: tắt đồng bộ |
| `TEXTFIX_KHO_BUCKET` | bucket |
| `TEXTFIX_KHO_PREFIX` | thư mục trong bucket, mặc định `textfix-kho/` |
| `TEXTFIX_KHO_ACCESS_KEY` / `TEXTFIX_KHO_SECRET_KEY` | khoá truy cập |
| `TEXTFIX_KHO_DIR` | thư mục đệm, mặc định `output/kho` |

Đồng bộ chỉ tải ảnh mới hoặc đổi (so `sha1` ghi trong mục lục), ghi `index.json` sau cùng. Kho đối tượng lỗi, mục lục hỏng,
thiếu gói `minio`: giữ đệm cũ (hoặc kho rỗng), máy chủ vẫn chạy như khi chưa có kho. `/api/health` báo `kho`.

## Dựng và đẩy kho

Laptop (có internet, không tới được kho đối tượng):

```bash
PIXABAY_KEY=<khoá> python scripts/kho_fetch.py ../kho_staging   # ứng viên từ Pixabay (khoá chỉ ở biến môi trường)
python scripts/kho_sheet.py ../kho_staging                 # contact sheet: ../kho_staging/_sheets/<id>.jpg
# chọn MỘT ô mỗi khái niệm -> picks.json {"bun-thang": 3, ...}
# không có ảnh đạt: tự tìm / chụp -> {"thit-cho": {"file": "duong/dan.jpg", "source": "giấy phép · tác giả · link"}}
python scripts/kho_index.py ../kho_staging picks.json ../kho_out   # ảnh 1024px + index.json, chạy kho.problems()
# nén ../kho_out thành zip (< 24 MB), tải lên notebook
```

Notebook máy chủ (tới được kho đối tượng; đặt các biến môi trường ở trên trước):

```bash
python scripts/kho_push.py --probe <host>                  # dò cổng API S3 nếu chưa biết
python scripts/kho_push.py kho_out                         # đẩy ảnh rồi index.json, đọc lại như máy chủ
```

## Mục lục

```json
{"id": "bun-thang", "file": "img/bun-thang.jpg",
 "tags": ["bun thang"], "aliases": ["bún thang"],
 "source": "Pixabay Content License · <tác giả> · https://pixabay.com/photos/...",
 "sha1": "..."}
```

- `aliases`: cụm tiếng Việt; chỉ cần bản có dấu (so khớp đã bỏ dấu nên "bun thang" tự khớp, ở mức không chắc). Mỗi mục nên
  có ít nhất một cụm >= 2 từ ("cốm làng vòng") để khách viết đủ tên thì khớp chắc; `kho.problems()` cảnh báo mục chỉ có cụm
  một từ.
- `tags`: tên tiếng Anh; tag đầu là tên LLM dùng trong prompt ("the bun thang from image 2").
- `source`: **bắt buộc** (giấy phép + tác giả + link). Chỉ ảnh có giấy phép cho phép dùng lại.
- Mỗi cụm thuộc **một** mục. `kho.problems()` báo trùng id, thiếu file, thiếu source, cụm trỏ hai mục.
- Thêm khái niệm: thêm vào `CONCEPTS` của `scripts/kho_fetch.py`, chạy lại các bước trên cho id đó.
