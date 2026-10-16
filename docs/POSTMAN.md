# Test TextFix bằng Postman

TextFix nhận nội dung một form (khuyến mại, tuyển dụng...) và trả về poster tiếng Việt. Một poster mất **khoảng 1-3 phút**,
nên API chia làm hai bước: **gửi yêu cầu** (trả ngay mã job), rồi **hỏi kết quả** cho tới khi xong.

```
 Postman                                   Máy chủ TextFix
 ───────                                   ───────────────
 POST /api/jobs  {form}          ──────►   nhận, xếp hàng
                 ◄──────  {job_id}         (trả ngay, chưa có ảnh)

 GET /api/jobs/{job_id}          ──────►   đang vẽ...
                 ◄──────  {status: "running", progress: "..."}
   ... bấm lại sau vài giây ...
 GET /api/jobs/{job_id}          ──────►   xong
                 ◄──────  {status: "done", poster: "/api/jobs/{job_id}/poster", result: {...}}

 GET /api/jobs/{job_id}/poster   ──────►
                 ◄──────  ảnh PNG
```

## 1. Nhập bộ lệnh mẫu

1. Postman → **Import** → chọn tệp `docs/textfix.postman_collection.json`.
2. Bấm vào collection **TextFix API** → tab **Variables** → sửa `baseUrl` thành địa chỉ máy chủ (ví dụ
   `http://10.x.x.x:8004`, không có `/` ở cuối) → **Save**.
3. Chạy **0. Kiểm máy chủ**: phải thấy `"ready": true`. Nếu `false` thì máy chủ đang khởi động (đợi 1-2 phút) hoặc nạp
   lỗi (xem trường `error`).

## 2. Một vòng test

| Bước | Lệnh trong collection | Làm gì / thấy gì |
|---|---|---|
| 1 | **1. Gửi yêu cầu** → chọn một mẫu (ví dụ *Khuyến mại*) → Send | trả `{"job_id": "...", "run_id": "...", "queue": 0}`. Postman tự nhớ `job_id`. |
| 2 | **2. Hỏi kết quả** → Send, **bấm lại sau mỗi 5-10 giây** | `status` đi từ `queued` → `running` → `done` (hoặc `error`). `progress` cho biết đang ở bước nào. |
| 3 | khi `status = "done"`: **3. Lấy poster** → Send | Postman hiện ảnh poster (tab Body). Poster thứ 2, 3...: thêm `?index=1`, `?index=2`. **3b** trả cùng ảnh dạng base64 trong JSON. |
| 4 | (khi cần báo lỗi) **4. Tải zip debug** → Send and Download | zip chứa mọi ảnh trung gian của lượt chạy: gửi kèm khi báo lỗi. |

Lệnh **Gọi một lần, chờ tới khi xong** (`/api/generate`) gộp cả ba bước: Send một lần, chờ 1-3 phút, nhận luôn kết quả.
Tiện để thử nhanh; test tải nhiều thì dùng `/api/jobs`.

## 3. Viết yêu cầu riêng

Thân yêu cầu là JSON, tab **Body → raw → JSON**. Bắt buộc có `category`; các trường khác tuỳ loại form:

| `category` | Form | Trường riêng (`*` = bắt buộc nếu không có `prompt`) |
|---|---|---|
| `promo` | Khuyến mại | `title`, `discount`*, `applied_product`, `date_start`, `date_end` |
| `product_intro` | Giới thiệu sản phẩm | `title`, `product_name`*, `product_desc`*, `highlights`, `price` |
| `opening` | Khai trương | `title`, `opening_date`*, `opening_promo`, `booking_contact` |
| `feedback` | Feedback khách hàng | `title`, `feedback_target`*, `feedback_quote`*, `feedback_highlights`, `special_offer` |
| `recruitment` | Tuyển dụng | `title`, `job_position`*, `job_desc`, `apply_deadline`*, `apply_method`* |
| `guide` | Hướng dẫn | `title`, `guide_steps` (danh sách: `["Bước 1", "Bước 2"]`) |

Dùng được cho mọi loại:

| Trường | Ý nghĩa |
|---|---|
| `prompt` | mô tả tự do. Có `prompt` thì không cần trường bắt buộc nào; mâu thuẫn với form thì theo `prompt` |
| `image_base64` | ảnh sản phẩm (chuỗi base64 hoặc `data:image/png;base64,...`). Chữ in trên sản phẩm được giữ nguyên |
| `store_name`, `phone`, `address`, `website_link` | thông tin cửa hàng |
| `style_pref` | `auto` (mặc định), `hien_dai`, `sang_trong`, `nang_dong`, `am_cung`, `le_hoi` |
| `primary_color` | màu chính, dạng `#ff6600` |
| `aspect_ratio` | `1:1` (mặc định), `4:5`, `9:16`, `16:9` |
| `num_images` | 1-4 poster cho một yêu cầu (mỗi cái một hướng thiết kế; nhiều ảnh thì lâu hơn) |
| `seed` | số nguyên: cùng yêu cầu + cùng seed -> chạy lại ra như cũ |

**Ảnh sản phẩm sang base64**: Linux / Mac `base64 -w0 anh.png` (Mac: `base64 -i anh.png`); Windows PowerShell
`[Convert]::ToBase64String([IO.File]::ReadAllBytes("anh.png"))`. Dán chuỗi vào `image_base64`. Mẫu *Giới thiệu sản phẩm*
có sẵn dòng này: dán ảnh vào hoặc xoá dòng đó (để nguyên chữ mẫu sẽ bị báo `invalid_image`).

Trường lạ (gõ sai tên, trường của form khác) **không gây lỗi**: bị bỏ qua và liệt kê trong `result.ignored_fields`. Gõ sai
tên trường thì xem ở đây.

## 4. Đọc kết quả

Khi `status = "done"`, các trường chính trong `result`:

| Trường | Ý nghĩa |
|---|---|
| (ngoài `result`) `poster` | `/api/jobs/{job_id}/poster`: lấy ảnh poster (lệnh 3); poster thứ i: `?index=i` |
| `posters[i].missing` | câu của khách **không có** trên poster (đáng ghi nhận khi test) |
| `posters[i].errors` | **số** lỗi trình bày còn lại (0 là sạch). Loại lỗi nằm trong `plan.json` (`plan_url`): `too_small` (chữ bị co nhỏ), `overflow` (tràn khung), `content_overlap` (chữ chồng nhau), `low_contrast` (chữ chìm nền) |
| `posters[i].draft_url` | nháp AI vẽ trước khi sửa chữ (so sánh với poster) |
| `brief.texts` | các câu sẽ đặt lên poster (`source`: khách nhập / AI viết thêm) |
| `elapsed_seconds` | thời gian chạy |
| `debug_zip_url` | zip mọi ảnh trung gian |

Ảnh lưu trên máy chủ có thời hạn (khoảng 300 lượt chạy gần nhất); cần giữ thì tải về.

## 5. Lỗi

| Khi nào | Thấy gì | Nghĩa là |
|---|---|---|
| ngay lúc gửi | **HTTP 422**, `detail.code` | đầu vào sai: `invalid_category` (sai loại form), `missing_required_fields` (`detail.fields` = trường thiếu), `invalid_aspect_ratio`, `invalid_num_images`, `invalid_image` |
| ngay lúc gửi | **HTTP 503** `model_not_ready` | máy chủ chưa sẵn sàng: chạy lệnh 0 |
| khi hỏi kết quả | `status: "error"`, `error.code` | job hỏng ở một bước (bảng dưới) |
| khi hỏi kết quả | **HTTP 404** | không có job này (máy chủ đã khởi động lại: job cũ mất, gửi lại) |

| `error.code` | Hỏng ở đâu |
|---|---|
| `llm_failed` | AI viết nội dung (máy LLM) |
| `draw_failed` | AI vẽ nháp (máy FLUX) |
| `draw_moderated` | máy FLUX từ chối nội dung (bộ lọc của máy FLUX) |
| `vlm_failed`, `vlm_empty_plan` | AI xếp chữ vào poster (máy VLM) |
| `lines_failed`, `erase_failed`, `render_failed`, `design_failed` | các bước xử lý ảnh của TextFix |
| `internal_error` | lỗi khác |

Với `num_images` > 1: job chỉ `error` khi **mọi** poster hỏng; một poster hỏng còn các poster khác thì job `done`, poster
hỏng có `posters[i].error` riêng.

**Báo lỗi**: gửi kèm `run_id` + zip debug (lệnh 4), hoặc ít nhất thân yêu cầu đã gửi và `error`.

## Lưu ý khi test

- Máy chủ làm lần lượt theo hàng đợi: gửi nhiều yêu cầu cùng lúc thì yêu cầu sau chờ lâu hơn (`position` khi hỏi kết quả
  = số job đứng trước).
- Bản demo **không có đăng nhập / khoá API**.
- Chi tiết kỹ thuật đầy đủ (ảnh debug từng bước, `plan.json`): `docs/API.md`.
