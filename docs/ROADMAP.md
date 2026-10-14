# Lộ trình: TextFix chỉ sửa chữ, Hi-SAM + LaMa thay bản xoá FLUX

Bắt đầu 14/10/2026. Tệp này là nguồn duy nhất cho câu hỏi "đang ở đâu, còn gì". Mỗi việc xong thì tick, kèm một dòng số đo
nếu có. Đường cũ (bản xoá FLUX) nằm ở nhánh git `legacy-flux-erase`, không so sánh, không phát triển.

## Đã chốt (người dùng, 14/10)

- TextFix **chỉ sửa chữ**. Vỏ, nút, icon to, **icon và dấu đầu dòng**, hoạ tiết là điểm ảnh của nháp. Nháp vẽ xấu thì chịu.
- Không bỏ dòng nào của nháp. Nháp vẽ thừa hay lặp thì VLM viết cho đủ.
- Bỏ hẳn bản xoá FLUX trên `main`. Hi-SAM cho biết chữ ở đâu, LaMa vẽ lại nền dưới chữ.
- Cụm nét Hi-SAM thấy mà OCR không trả dòng: thành dòng, **VLM quyết** giữ / viết lại / xoá. VLM không nhắc tới thì giữ.
- Chữ neon quầng rộng và chữ rất nhỏ: cho chạy đã, xấu thì sửa sau.
- Thiếu tệp Hi-SAM hoặc LaMa: báo lỗi rõ và dừng.
- LaMa chạy **một lần** mỗi poster (gỡ mọi dòng). Dòng VLM chọn giữ được trả lại điểm ảnh nháp bằng mặt nạ, không gỡ lại.

## Dây chuyền đích

| # | Bước | Trạm | Ghi chú |
|---|---|---|---|
| 1 | LLM viết đề | điều phối | không đổi |
| 2 | FLUX vẽ nháp | vẽ (GPU) | không đổi |
| 3 | Hi-SAM tìm nét chữ | vẽ (GPU) | thay "FLUX xoá + dọn chữ sót" (6,5 s -> ~0,2 s) |
| 4 | Đo dòng: OCR + dòng cong + cụm nét chưa ai nhận | thiết kế | không còn ô vỏ, ô chi tiết, dấu đầu dòng |
| 5 | Gỡ mọi dòng: mặt nạ nét + quầng, LaMa lấp | thiết kế | một lần LaMa |
| 6 | VLM điền chữ theo dòng: viết / giữ / xoá | thiết kế | |
| 7 | Mã rà lệnh: chặn chữ, lấp dòng bị bỏ, tách lệnh gộp sai | thiết kế | |
| 8 | Trả điểm ảnh nháp cho dòng được giữ | thiết kế | chỉ tính lại mặt nạ |
| 9 | Dựng chữ, kiểm, tự sửa lỗi nặng | thiết kế | |

## Chặng A -- Hi-SAM + LaMa thay bản xoá (đang làm)

Mã
- [x] Nhánh `legacy-flux-erase` giữ đường cũ (đã push).
- [x] `textfix/hisam.py`: nạp một lần mỗi thiết bị, fp16, đọc / ghi mặt nạ. *(chạy CPU tại máy; chưa chạy trong máy chủ)*
- [x] `lonly.erase_mask` + `lonly.erase`: nét Hi-SAM theo dòng, quầng (dải sát nét, lan đuôi nhạt trên nền phẳng), LaMa. Bỏ đường luật dự phòng. *(tại máy)*
- [x] `slots.build(draft, ink)`: không cần bản xoá; cụm nét chưa ai nhận thành dòng `orphan`. *(tại máy)*
- [x] Gỡ dấu đầu dòng khỏi `main`. Icon trước dòng để nguyên, mã gỡ icon khỏi lệnh VLM (`lonly.marks_rule`). Riêng chấm tròn "•" là ký tự: Hi-SAM gỡ nó cùng chữ nên dòng mới viết lại "• " (`slots._lead_dots`).
- [x] `lonly.erase` (LaMa một lần, gỡ mọi dòng) + `lonly.settle` (trả điểm ảnh nháp cho dòng giữ, chỉ tính lại mặt nạ).
- [x] `lonly.fill_gaps`: dòng `orphan` VLM không nhắc -> giữ; VLM ghi xoá -> xoá.
- [x] Rào tách lệnh VLM gộp các dòng không xếp chồng (`lonly.split_far`; thử trên ca bốn thẻ: tách đúng 4 lệnh).
- [x] Lời dặn VLM (`SYSTEM_L`): icon không phải việc của VLM, chấm tròn, dòng `orphan` (keep / text / skip). *(chưa chạy VLM thật)*
- [x] `engine.draw` / `engine.design`: bỏ xoá FLUX và dọn chữ sót, truyền mặt nạ nét. *(chạy trọn trạm thiết kế tại máy với VLM giả)*
- [x] `server/app.py`: nạp Hi-SAM theo card, kiểm tệp lúc khởi động, lưu `ink.png` / `ink.jpg` / `base.jpg`, bỏ `plate*`. **(CHƯA CHẠY LẦN NÀO)**
- [x] `server/ui.html`: bỏ hai ô bản xoá; bốn ô xem ① Hi-SAM nét chữ thô (`hisam.jpg`), ② mặt nạ gỡ (`ink.jpg`), ③ LaMa nền sạch VLM nhìn (`clean.jpg`), ④ nền dựng (`base.jpg`); bảng "Gỡ chữ" (thời gian Hi-SAM / LaMa, % nét, % mặt nạ, cụm nét lạ VLM quyết gì); bỏ lớp S / I; trang không bị trình duyệt giữ bản cũ. *(dựng thử tại máy bằng Chromium với một lượt giả)*
- [x] UI debug "Chữ khó (typography)": mỗi dòng cong / dọc / nghiêng / dựng thẳng / OCR sót / cụm nét lạ / chấm đầu dòng / khung dời khỏi icon / chữ in trên sản phẩm -- mã nhận ra bằng bằng chứng gì (`l["typo"]`), lệnh VLM / mã, dựng thế nào (`plan["render"]`: xoay, đường cong, cỡ, co, nén); khung xem dòng vẽ dòng nghiêng theo đa giác thật.
- [x] `steps.jpg` năm cột (nháp, dòng, nét chữ, nền, poster); zip báo cáo mang `ink.png` thay bản xoá. **(batch_report chưa chạy)**
- [ ] Tờ ảnh ghép và trang HTML của `batch_report`: thêm cột nét chữ / nền.
- [x] `audit.flags`: cờ cho dòng `orphan`, dòng gỡ theo lõi khung, lệnh gộp bị tách.
- [x] `scripts/erase_bench.py` theo đường mới. *(tuỳ chọn `--recheck` cho số vô lý > 100%: chưa dùng được)*
- [x] `scripts/lonly_replay.py` theo đường mới (dựng lại tại máy với lệnh VLM đã lưu).
- [x] Xoá khỏi `main`: `cleanup.py`, `overlay.py`, `textmask.py`, `flux.erase`, 12 script chỉ đường cũ dùng.
- [x] Dọn mã trước chặng B (14/10): bỏ khỏi `slots.py` lời dặn cũ `SYSTEM` / `SYSTEM_V1` / `REVIEW` / `JUDGE` (có danh mục icon), `describe`, vòng VLM duyệt (`review`, `judge`, `salvage`, `accept`; nhánh cũ giữ), các luật ô vỏ / ô chi tiết / icon / hàng sao của `validate`; bộ dựng chỉ còn lệnh chữ; M chỉ còn dòng L; bỏ `flux.edit`, `geo.band_arc`, `ship.pack_dir`, `scripts/probe.py`. `slots.py` 2.021 -> ~1.150 dòng. Lời dặn VLM: 0 chữ "icon". Kiểm: 12 poster dựng lại trước / sau giống nhau (11 giống từng điểm ảnh, 1 lệch một dòng chấm, nhìn như nhau); chạy trọn trạm thiết kế với VLM giả; dựng UI.
- [x] Bỏ bộ icon: `textfix/icons.py`, `assets/icons/` (1.544 SVG), `scripts/fetch_icons.py`, phần `<i-icon>` / `<i-stars>` của `render.expand`, mục đếm icon trong `check_server.py`. Bộ dựng gỡ thẻ icon / sao / svg khỏi chữ và ghi log thay vì vẽ.
- [x] `docs/SERVER.md`, `README.md`, mục 14/10 trong `docs/TUNING.md`, `scripts/check_server.py` kiểm tệp Hi-SAM / LaMa.
- [x] Xoá `docs/DESIGN.md`, `PLAN_BLOCKS.md`, `SLOTS_GT.md`, `tools/annotate.html` (tả / phục vụ đường cũ; còn trong git và các nhánh legacy). Dữ liệu `bench/slots_gt/` giữ nguyên.

Đo tại máy (trước khi xin chạy máy chủ)
- [x] Bước 4 trên 30 nháp accept2: 241 / 242 dòng cũ còn, 3 dòng cong còn đủ, thêm 4 cụm nét (chữ vòng trên huy hiệu trà).
- [x] Bước 4 trên cả 65 nháp (accept2 + mix1): 531 / 532 dòng cũ còn, 7 / 7 dòng cong, 11 cụm nét ở 4 poster, 4 dòng có chấm.
- [ ] Bước 5: mới xem mắt 10 nháp khó với luật quầng cuối (OCR đọc lại nền: 0 ký tự sót ở cả 10); chưa chạy đủ hai tập.
- [ ] Dựng lại với lệnh VLM đã lưu: 30 poster accept2 với luật quầng đầu (xem 8, không vỡ); chưa dựng đủ 65 với luật cuối -- người dùng chọn chạy thẳng máy chủ.

Máy chủ
- [x] Lượt `accept2` trên main 3ac2285 (zip 1014_040431, giải nén ở `output/accept2h/`): đủ 30, không poster nào lỗi. Trung vị
  18,5 s / poster không tính LLM (nét chữ 0,2 s, gỡ chữ 1,7 s; trước ~29 s). 248 dòng (trước 242), 3 / 3 dòng cong. VLM không viết
  icon nào. Không hết VRAM.
- [x] Nghiệm thu, xem 30 poster: gỡ chữ sạch, vỏ / nút / con dấu / huy hiệu còn nguyên. BA LỖI DO CHẶNG A, đã vá (tại máy):
  (1) q03 MẤT LOGO Coca-Cola trên lon: OCR đọc logo thành dòng "eo.mg", VLM không nhắc, mã xoá trống -> poster có ảnh sản phẩm thì
  dòng không ai viết được GIỮ; (2) q02 ba chiếc lá trước dòng bị gặm: quầng nay phải dính vào nét, cỡ dòng không vượt 2,2 lần
  cao nét, dấu câu chỉ gỡ ở giữa chữ, khung dòng dời khỏi icon đứng trước; (3) h04 "NGÀY • HỘI": chấm đầu dòng cần cả OCR
  đọc ra ký tự không phải chữ. Thêm: ngưỡng nền phẳng 4 -> 8 (bóng chữ 3D e04 lúc hết lúc không).
- [x] ~~Một lượt accept2 nữa với bản vá~~ -- người dùng bỏ (14/10); bản vá kiểm tại máy trên đúng các ca lỗi.
- [ ] VRAM đỉnh khi Hi-SAM + LaMa + FLUX chung card (lượt này không lỗi, chưa có số).

## Chặng B -- VLM ghép câu, code chép chữ và dựng theo nháp (đang làm)

Chốt 14/10 (thay hướng "khối, bậc cỡ, chỗ trống thật"): nhóm lấy từ VAI câu khách trong brief, không đoán từ ảnh, không để VLM
khai; không cắt câu khách, chữ dài thì co; nháp thiếu câu thì chịu; các bước luôn theo thứ tự khách. Cơ sở: probe 30 poster
(`output/probe_role/`): chữ đúng câu khách 30/30, 19 tập cùng vai dựng đều. Đường cũ (VLM viết html, group): nhánh `legacy-ops-html`.
- [x] VLM chỉ trả `assign` (câu <-> dòng, `split`, dòng lấp, keep / skip, `bullet`), `styles` theo vai, `accent`, `missing` (`lonly.SYSTEM_L`, `fill.parse`).
- [x] `textfix/fill.py`: chép nguyên văn, một câu nhiều chỗ, dòng chưa ai ghép, chặn số bịa ở dòng lấp, thứ tự các bước (cờ `ordered` của brief).
- [x] Hình theo nháp: khung = nét Hi-SAM, cỡ = trung vị cỡ đo, màu theo dòng, lề theo mép nét, ngắt dòng theo tỉ lệ ký tự OCR; dòng khác cỡ / mẩu cùng hàng.
- [x] Tập cùng vai (cỡ nháp đôi một lệch <= 1,25): một cỡ, một màu, một lề, một mức nén theo mục chật nhất (`render.render_units`).
- [x] Bỏ: html VLM, `_guard`, `validate`, `marks_rule`, `split_far`, `fix_ops`, nhóm / cân cỡ / căn lề của render, `_pack_lists`, `_list_order`.
- [x] UI debug: đơn vị chữ (màu = tập), bảng câu ↔ dòng, cỡ / co từng phần tử; `scripts/lonly_replay.py` dựng lại plan cũ qua `fill.from_ops`.
- [x] Dựng lại tại máy 30 accept2 + 35 mix1c (lệnh VLM cũ qua `fill.from_ops`): 0 tập lệch cỡ, chữ đúng nguyên văn; câu thiếu đều đã thiếu trong lệnh cũ.
- [x] VLM không nhận chữ OCR (người dùng 14/10); số bước đặt lại theo thứ tự khách; `engine.design` chạy trơn với VLM giả.
- [ ] Hi-SAM phân cấp (từ / dòng) chia dòng thay OCR: tick sát chữ, hai cột gộp, một khung hai hàng.
- [ ] Máy chủ: một lượt accept2 với lời dặn mới, xem 100% ảnh, so lượt 1014_040431.

## Chặng C -- Thi

- [ ] `bench/final1.json`, một lượt, chấm. Ngưỡng: hỏng <= 10%, không mất chữ sản phẩm, không số / giá / liên hệ bịa.

## Giới hạn đã biết, chưa xử lý

- Chữ neon quầng rộng: còn bóng mờ sau khi gỡ.
- Chữ rất nhỏ: Hi-SAM thấy ít nét (chân trang accept2 e02); chế độ cắt mảnh chưa đo trên GPU.
- Chữ chạy vòng trên con dấu / huy hiệu: OCR đọc từng mẩu, phần còn lại thành các cụm nét rời.
- Logo kiểu thư pháp (Coca-Cola, Dove) Hi-SAM phần lớn không đánh dấu: không dùng Hi-SAM để nhận chữ sản phẩm.
