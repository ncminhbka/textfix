# TextFix API: ví dụ curl

Đặt địa chỉ máy chủ một lần (đổi thành địa chỉ public / gateway; không có `/` cuối):

```bash
BASE=http://<máy chủ>:8004
```

Một poster mất **1-3 phút**, nên có ba bước, chỉ cần `job_id`:

```
1. POST /api/jobs                    -> {"job_id": "..."}             gửi yêu cầu (trả ngay)
2. GET  /api/jobs/{job_id}           -> {"status": "running" ...}     hỏi mỗi 2-5 giây tới khi "status": "done"
3. GET  /api/jobs/{job_id}/poster    -> ảnh PNG                       lấy poster
```

Chi tiết trường: `docs/API.md`.

## 0. Máy chủ sẵn sàng chưa

```bash
curl -s $BASE/api/health
# {"ready": true, "error": null, "queue": 0, ...}   -- ready = false: đang khởi động / nạp lỗi (xem "error")
```

## 1. Gửi yêu cầu

```bash
curl -s -X POST $BASE/api/jobs -H 'Content-Type: application/json' -d '{
  "category": "promo",
  "title": "Siêu sale cuối tuần",
  "discount": "Giảm 30%",
  "applied_product": "Toàn bộ trà sữa",
  "date_start": "20/10/2026",
  "date_end": "22/10/2026",
  "store_name": "Mây Coffee",
  "phone": "0901 234 567",
  "aspect_ratio": "4:5",
  "num_images": 1
}'
# {"job_id": "2468c0a1a852", "run_id": "run_20261016_083553_2468", "queue": 0}
```

Các loại form khác (thay phần thân):

```bash
# Giới thiệu sản phẩm
'{"category":"product_intro","product_name":"Nồi chiên không dầu AirPro 5L","product_desc":"Chiên giòn không cần dầu","highlights":"Dung tích 5 lít; 8 chế độ nấu","price":"1.290.000đ"}'
# Khai trương
'{"category":"opening","title":"Khai trương chi nhánh mới","opening_date":"01/11/2026","opening_promo":"Mua 1 tặng 1","booking_contact":"Hotline 1900 1234","store_name":"Bếp Nhà Mây","aspect_ratio":"9:16"}'
# Feedback khách hàng
'{"category":"feedback","feedback_target":"Dịch vụ spa thư giãn","feedback_quote":"Nhân viên nhẹ nhàng, không gian thư thái","special_offer":"Giảm 15% cho khách quay lại","store_name":"Lotus Spa"}'
# Tuyển dụng
'{"category":"recruitment","title":"Tuyển nhân viên pha chế","job_position":"Nhân viên pha chế","apply_deadline":"30/11/2026","apply_method":"Gửi CV về hr@maycoffee.vn","store_name":"Mây Coffee"}'
# Hướng dẫn (guide_steps là mảng)
'{"category":"guide","title":"Cách đặt hàng online","guide_steps":["Chọn món trên app","Nhập địa chỉ","Thanh toán","Nhận hàng"]}'
# Chỉ prompt tự do (không cần trường bắt buộc)
'{"category":"promo","prompt":"Poster Black Friday cho shop giày, giảm đến 50%, tông đen vàng"}'
```

### Kèm ảnh sản phẩm

Ảnh là chuỗi base64 trong trường `image_base64`. Thân lớn: ghi ra tệp rồi gửi bằng `-d @tệp`:

```bash
B64=$(base64 -w0 san_pham.jpg)          # Mac: base64 -i san_pham.jpg
printf '{"category":"product_intro","product_name":"Nồi chiên AirPro 5L","product_desc":"Chiên giòn không cần dầu","image_base64":"%s"}' "$B64" > req.json
curl -s -X POST $BASE/api/jobs -H 'Content-Type: application/json' -d @req.json
```

## 2. Hỏi kết quả (mỗi 2-5 giây)

```bash
JOB=2468c0a1a852
curl -s $BASE/api/jobs/$JOB
# đang chạy: {"status": "running", "progress": "0/1 xong · ảnh 1: gỡ chữ (LaMa)", "position": null, "elapsed": 41.2, ...}
# xong:      {"status": "done", "poster": "/api/jobs/2468c0a1a852/poster", "result": {...}}   -> sang bước 3
# lỗi:       {"status": "error", "error": {"code": "vlm_failed", "stage": "...", "message": "..."}}
```

`status`: `queued` → `running` → `done` | `error`.

Vòng hỏi tự động (cần `python3` để đọc JSON):

```bash
JOB=$(curl -s -X POST $BASE/api/jobs -H 'Content-Type: application/json' -d @req.json | python3 -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')
while :; do
  R=$(curl -s $BASE/api/jobs/$JOB)
  S=$(echo "$R" | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["status"], r.get("progress") or "")')
  echo "$S"
  case "$S" in done*|error*) break;; esac
  sleep 3
done
curl -s -o poster.png $BASE/api/jobs/$JOB/poster
```

## 3. Lấy poster

Khi `status` là `done`:

```bash
curl -s -o poster.png $BASE/api/jobs/$JOB/poster                    # ảnh PNG
curl -s -o poster2.png "$BASE/api/jobs/$JOB/poster?index=1"         # poster thứ 2 (khi num_images > 1)
curl -s "$BASE/api/jobs/$JOB/poster?format=base64"                  # JSON: {"job_id", "index", "width", "height", "mime", "image_base64"}
```

| Mã HTTP | `detail.code` | Nghĩa |
|---|---|---|
| 200 | | ảnh (hoặc JSON base64) |
| 409 | `not_ready` | chưa xong, hỏi tiếp bước 2 |
| 409 | `job_failed` / `poster_failed` | job / poster đó lỗi (`detail.error` có mã lỗi) |
| 404 | `job_not_found` / `poster_not_found` | sai `job_id` (hoặc máy chủ đã khởi động lại) / sai `index` |
| 410 | `poster_expired` | ảnh đã bị dọn (máy chủ giữ ~300 lượt gần nhất): bên gọi tự lưu ảnh về |

## 4. Gọi một lần, chờ tới khi xong (thử nhanh)

```bash
curl -s --max-time 600 -X POST $BASE/api/generate -H 'Content-Type: application/json' \
  -d '{"category":"promo","title":"Siêu sale cuối tuần","discount":"Giảm 30%","store_name":"Mây Coffee"}'
# trả thẳng "result" (như bước 2 khi xong), trong đó có "job_id" -> lấy ảnh: GET $BASE/api/jobs/<job_id>/poster
# lỗi: HTTP 500 {"status":"error","error":{...},"run_id":...}
```

Giữ kết nối 1-3 phút: gateway / proxy phía trước phải cho phép timeout đủ dài. Tích hợp thật nên dùng `/api/jobs`.

## 5. Zip debug của một lượt (khi báo lỗi)

```bash
curl -s -o run.zip $BASE/api/runs/run_20261016_083553_2468/zip
```

## Lỗi đầu vào (trả ngay, HTTP 422)

```bash
curl -s -X POST $BASE/api/jobs -H 'Content-Type: application/json' -d '{"category":"recruitment","title":"Tuyển dụng"}'
# 422 {"detail": {"code": "missing_required_fields", "fields": ["job_position", "apply_deadline", "apply_method"], ...}}
```

Mã: `invalid_category`, `missing_required_fields`, `invalid_aspect_ratio`, `invalid_num_images`, `invalid_image`.
`503 model_not_ready`: máy chủ chưa sẵn sàng. Mã lỗi job (`llm_failed`, `draw_failed`, `vlm_failed`, ...): `docs/API.md`.

## Windows

PowerShell 5.1: `curl` là lệnh khác, dùng **`curl.exe`**. Để tránh lỗi dấu nháy, ghi thân JSON ra tệp (UTF-8) rồi gửi:

```powershell
curl.exe -s -X POST "$BASE/api/jobs" -H "Content-Type: application/json" -d "@req.json"
```

## Đường dẫn cần mở ở gateway

`/api/*` là đủ (gửi, hỏi, lấy poster). `/outputs/*` chỉ cần nếu muốn xem ảnh debug từng bước (các trường `*_url` trong
`result`, đường dẫn tương đối với gốc máy chủ). `/` là giao diện demo / debug, không cần public.
