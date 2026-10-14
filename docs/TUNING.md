# Theo dõi tune từng bước

## Quy tắc tune khi nhận một tập dev mới

1. **Nhận đủ dữ liệu.** Giải nén vào `output/<tập>/`; đếm `plan.json` so với manifest. Zip cụt / thiếu -> cứu phần còn
   nguyên, báo ngay ca nào thiếu, xin tải lại; không kết luận trên tập thiếu mà không ghi rõ.
2. **Ghi mốc trước khi sửa gì.** `scripts/runs_report.py --runs output/<tập> --all --again --no-ship`: lỗi lượt đầu / cuối
   theo loại, sửa bằng code chạy / giữ, câu thiếu (VLM khai + đo bằng mã), thời gian -> mục "Mốc" của tập.
3. **Chấm lượt tune trước.** Tập mới là phép thử của các sửa "chờ server" / "thử": từng dòng đó đổi thành **đạt** /
   **không đạt** kèm số. Sửa không đạt thì xem lại trước khi thêm sửa mới.
4. **Xem bằng mắt 100% ảnh** (nháp | nét chữ | nền | poster, kèm plan khi cần). Ghi mỗi lỗi vào **bước gây ra nó**, không phải bước
   nó lộ ra (chữ co nhỏ ở B4 có thể do B2.5 đo sai cỡ; chữ rác ở poster có thể do B2 xoá sót) -- truy tới số đo / lệnh cụ thể.
   Bảng "Lỗi theo bước" ghi số ảnh dính.
5. **Xếp ưu tiên** theo số ảnh dính × độ nặng: sai / mất / bịa chữ khách > chữ rác, chữ chồng > vỡ bố cục (vỏ mất, khung sai) >
   thẩm mỹ (cỡ, căn, hiệu ứng). Lỗi là **chính sách** (prompt hay form thắng, có cho tự viết nhãn không, chấp nhận hạn chế
   model) -> hỏi người dùng, không tự quyết.
6. **Sửa tổng quát, có số đo.** Ưu tiên **[code]** đo được hơn **[prompt]**; prompt chỉ cho điều code không biết. Không viết
   luật theo một ca. Mỗi sửa có số trước -> sau: replay plan cũ / dựng lại / bộ đáp án ô / bộ tổng hợp; không đo được local
   (LLM, VLM, FLUX) thì ghi **chờ server**.
7. **Chống khớp quá sát.** Mỗi sửa chạy trên **toàn tập** (không chỉ ca lỗi) + `bench/slots_gt` (`slots_gt.py eval --vs base`)
   + `geo_synth` khi đụng ô / hình học: không bắt nhầm, không tụt số. Ngưỡng chọn ở **khoảng tách** giữa ca đúng và ca sai trên
   phân bố, không vừa khít một ca. Lỗi một ảnh không lặp lại và không nặng -> ghi "theo dõi", chưa sửa.
8. **Tính năng dở dang để sau cờ tắt** (vd. `TEXTFIX_GEO`) cho tới khi cả đường đi (đo -> VLM -> dựng) xong và đo được.
9. **Ghi lại.** Mỗi sửa một dòng ở nhật ký đúng bước, gắn [code] / [prompt LLM] / [prompt VLM] + trạng thái; cập nhật bảng
   "Đã siết prompt"; docs/DESIGN.md cho cơ chế mới. Commit theo cụm (kiểm không có tên dịch vụ nội bộ / bí mật).
10. **Báo người dùng:** lỗi theo bước, đã sửa gì (số đo), cái gì chờ server, cái gì cần quyết.
11. **Soạn tập dev kế tiếp:** thêm ca cho lỗi mới thấy và nhóm chưa phủ, seed cố định; giữ tập cũ để chạy lại so trước / sau.

Mỗi lượt tune ghi một dòng: ngày, tập, lỗi thấy, thay đổi, số đo trước -> sau, trạng thái.
Trạng thái: **đo** (có số đo trên cả tập) · **thử** (chỉ thử vài ca / dựng lại / LLM giả) · **chờ server** (cần LLM / VLM / FLUX
thật để biết) · **đề xuất** (chưa làm) · **chấp nhận** (quyết không sửa).
Loại sửa: **[code]** luật / số đo trong mã · **[prompt LLM]** system prompt B0 · **[prompt VLM]** system prompt designer B3.
Tập: `dev2` = bench/dev2.json (chạy server 09/10, đủ 62 ảnh / 32 ca) · `gt` = bench/slots_gt (23 ảnh, đáp án ô) · `synth` =
scripts/geo_synth.py (chữ cong / nghiêng / nhiều cỡ tổng hợp, đáp án tuyệt đối).

## Các bước

| Bước | Việc | Mã | Có AI | Núm chỉnh chính | Đo bằng |
|---|---|---|---|---|---|
| B0 Brief | intent (poster / ảnh thường), câu khách nguyên văn, prompt FLUX, các hướng thiết kế; code giữ cam kết | `textfix/brief.py` | LLM | `SYSTEM`, `FIELD_ROLE`, `ORIENT`, luật nguyên văn / ngoặc kép / trường form / `overridden` | brief.json: câu khách đủ / thừa / sai, `log` |
| B1 Nháp | FLUX.2-klein vẽ cả poster có chữ | `textfix/flux.py` | FLUX | prompt (B0), seed, số bước | mắt: bố cục, số hàng danh sách |
| B2 Nét chữ + gỡ chữ | Hi-SAM tìm nét chữ của nháp (trạm vẽ); mặt nạ nét + quầng theo dòng, LaMa lấp MỘT lần, dòng "keep" được trả điểm ảnh nháp | `textfix/hisam.py`, `lonly.py` (erase_mask, erase, settle) | Hi-SAM + LaMa | `MASK_GROW`, `WEAK`, `HALO_*` | `scripts/erase_bench.py` (ký tự OCR còn đọc được trên nền, % đổi), xem mắt `ink.jpg` / `base.jpg` |
| B2.5 Dòng | dòng OCR + chữ cong + cụm nét OCR không đọc (`orphan`) + chấm tròn đầu dòng; không còn ô vỏ / chi tiết / dấu đầu dòng | `slots.py` (lines, build, _orphans, _lead_dots), `glyph.py`, `geo.py` | OCR | `ORPHAN_*`, `DOT_*`, ngưỡng `geo.py` | số dòng, dòng cong, cụm nét trên các nháp đã lưu |
| B3 Designer | VLM điền chữ theo dòng: text / keep / skip (chỉ cụm nét lạ); code tách lệnh gộp sai, chặn chữ, lấp dòng bị bỏ, luật icon / chấm đầu dòng | `lonly.py` (SYSTEM_L, fix_ops, split_far, fill_gaps, marks_rule), `slots.py` (plan, validate, _guard) | VLM | `SYSTEM_L`, `STACK_GAP`, luật `_guard` | plan.json: `missing`, log chặn chữ, cờ `audit.flags` |
| B4 Dựng + kiểm | Chromium dựng, đo lỗi, sửa bằng code (vòng duyệt VLM tắt) | `textfix/render.py`, `engine.py` | — | `CONDENSE`, `FIX_ROUNDS`, `LH_ACCENT`, `PARA_WORDS`, `CONTRAST_*` | `scripts/runs_report.py`: lỗi lượt đầu / cuối |

## Mốc dev2 (09/10, mã 3fe4022 trên server, đủ 62 ảnh)

| | Số |
|---|---|
| Lỗi lượt đầu | too_small 50 · low_contrast 12 · overflow 8 · uneven_size 8 · boxes_overlap 6 · unassigned 3 |
| Sửa bằng code | chạy 20 lần, giữ 14 (vòng duyệt VLM tắt: 0) |
| Lỗi cuối | too_small 50 · overflow 7 · uneven_size 8 · boxes_overlap 1 |
| Câu khách thiếu | VLM tự khai ở 9 ảnh; đo bằng mã thêm 5 ảnh VLM bỏ mà không khai |
| Thời gian trung vị (s) | LLM 6.8 · nháp 2.7 · xoá 4.8 · dọn 1.0 · ô 0.6 · VLM 9.6 · dựng 1.8 |

Xem bằng mắt cả 62 ảnh (nháp | bản xoá | poster). Lỗi theo bước, số ảnh dính (một ảnh có thể dính nhiều):

| Bước | Lỗi | Ảnh |
|---|---|---|
| B0 | khung 16:9 viết "vertical poster" (d16, d32) · chữ phong cách thành câu khách (d06) · trường form bị viết lại rồi bị bỏ (d32) · form / prompt mâu thuẫn (d23) | 6 |
| B1 | hướng dẫn vẽ thiếu hàng (d19, d20) | 3 |
| B2 | bản xoá còn sót chữ, chữ mới dựng chồng lên (d18 v1 "Gửi CV…", d27 v1 "-36") | 2 |
| B2.5 | pill / nút nhạt hoặc chỉ có viền không thành ô -> pill vẽ lại bé tí (d02, d10, d13, d17) · khung dòng chữ nhỏ co về một mẩu + cỡ nháp đo phồng -> chữ co nhỏ (too_small, 28-35 ảnh) · chữ cong bị cắt mẩu (d10, d12 v2) · khung tối nhiều dòng không thành ô (d26, d12) | ~35 |
| B3 | chép chữ méo của nháp (d24 "TỬA") · viết chữ lên màn hình sản phẩm (d27 "00:36", "GIÁ", "VND") · viết tắt địa chỉ (d12 "Q.3") · bỏ chữ in trên sản phẩm (d25 con dấu) · bỏ câu không khai (5 ảnh) · bỏ trống hàng giữa danh sách (d05) · thẻ trong thẻ (d18) · đổi căn giữa thành căn trái (d30) · mất hiệu ứng vàng kim (d02 v1) | ~15 |
| B4 | dấu chồng dòng trên "DRQP" (d06) · cột tên / giá dính (d05) · emoji ô vuông (d10) · chữ chìm nền không bắt (d11) · trích dẫn mồ côi chữ (d17) | 5 |

## Đã siết prompt LLM / VLM chưa

| Prompt | Đã thêm (09/10) | Bằng chứng dev2 | Kiểm |
|---|---|---|---|
| **LLM B0** | `intent` poster / image; prompt sơ sài cạnh form đã điền vẫn là poster, mọi trường là chữ; prompt mâu thuẫn form -> prompt thắng, ghi `overridden`; khung ảnh nói bằng lời (gợi ý trong tin nhắn) | d23, d16, d32, ca d33-d40 | LLM giả · **chờ server** |
| **VLM B3** | điền hàng danh sách từ trên xuống, chỉ bỏ hàng thừa cuối; màn hình / nhãn / bao bì sản phẩm luôn keep, không viết chữ / số lên đó; dữ kiện không viết tắt ("Quận 3" không thành "Q.3"); dòng trong vỏ không tự thêm nền / viền (không thẻ trong thẻ); giữ căn lề của nháp từng khối | d05, d27, d12, d18, d30 | **chờ server** (VLM ở máy ngoài, không chạy local) |
| VLM B3, chưa đổi | luật "chữ model tự bịa: viết theo vai trò" vẫn giữ -- cho phép nhãn chung như "Khai Trương" khi khách không ghi tiêu đề (d16, nhìn ổn); dữ kiện vẫn cấm | d16 | **cần quyết** nếu muốn cấm hẳn |

Phần lớn lỗi VLM được chặn bằng **[code]** (chắc chắn, đo lại được trên plan cũ); prompt chỉ thêm khi code không biết được (vùng
nào là màn hình sản phẩm, căn lề mong muốn).

## Nhật ký

### B0 Brief

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | — | "tạo ảnh con mèo" báo lỗi "không tìm ra chữ" | [prompt LLM] `intent`; [code] không câu khách -> chỉ FLUX vẽ | lỗi -> ảnh thường | thử (LLM giả) · chờ server (d33-d36) |
| 09/10 | — | form đủ + prompt sơ sài có thể làm LLM bỏ form | [prompt LLM] prompt sơ sài cạnh form vẫn là poster, mọi trường là chữ | — | chờ server (d37-d40) |
| 09/10 | dev2 d16, d32 | khung 16:9 mà prompt viết "vertical poster" | [code] khung ảnh nói bằng lời trong tin nhắn (`ORIENT`) | — | thử · chờ server |
| 09/10 | dev2 d06 | phong cách "Y2K", "vintage" trong ngoặc kép thành câu khách -> báo thiếu câu | [code] chuỗi LLM dùng làm từ tả (không ngoặc) trong prompt_en -> không thêm lại | 2 câu thừa -> 0 | thử (LLM giả) |
| 09/10 | dev2 d32 | LLM viết "Apply deadline: 15/12/2026" -> bị bỏ, mất hạn nộp | [code] thêm lại giá trị trường form không có trong câu khách | mất 1 trường -> 0 | thử (LLM giả) |
| 09/10 | dev2 d01 | ngoặc kép cong “…” không được bắt lại khi LLM sót | [code] nhận cả ngoặc cong | — | thử |
| 09/10 | dev2 d23 | form "Giảm 20%", prompt "GIẢM 30%": cả hai thành câu khách, poster không vẽ câu prompt | **prompt thắng** (quyết 09/10): [prompt LLM] ghi trường bị thay vào `overridden`; [code] bỏ giá trị đó, không thêm lại | 2 câu form thừa -> 0 | thử (LLM giả) · chờ server |

### B1 Nháp

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 d19, d20 | hướng dẫn 5 / 8 bước: nháp vẽ 4 / 6-7 hàng -> mất bước, số bước nhảy (1 2 4 5) | hạn chế của FLUX; câu thiếu đã báo trên UI | 3/4 ảnh hướng dẫn | **chấp nhận** (quyết 09/10) |

### B2 Xoá

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 d18 v1 | OCR nháp sót dòng "Gửi CV: tuyendung@…" -> chữ còn trên bản xoá không bị coi là chữ sót -> poster có chữ rác | [code] dòng bản xoá không khớp dòng nháp: OCR lại vùng đó trên nháp (cắt, phóng 2×) thấy cùng chữ -> chữ sót (`CONFIRM`); tắt khi khách tải ảnh sản phẩm | dev2: +3 dòng sót được dọn (d18, d09 số điện thoại, d19) | đo (bản xoá thô, inpaint) · chờ server (FLUX) |
| 09/10 | dev2 d27 v1 | inpaint "-36%" trên huy hiệu không sạch, còn "-36" -> chữ mới dựng chồng | [code] lượt 2: OCR lại bản đã dọn, chữ còn -> FLUX vùng cắt | — | chờ server (lượt 2 cần FLUX) |
| — | dev2 d09, d22 | vết loang chỗ logo / huy hiệu sau dọn (poster thường che được) | — | — | theo dõi |

### B2.5 Ô

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 | pill trắng / xám trên nền gần màu (ΔE 10-18 < 20) không thành S -> VLM vẽ pill ôm sát chữ, bé tí (d10, d13, d17, d32) | [code] lượt "vỏ nhạt" quanh từng dòng (`SOFT_*`) | dev2: +5 vỏ đúng, 0 vỏ mất | đo (dev2) |
| 09/10 | dev2 d02 | nút chỉ có viền ("Shop now") không thành S | [code] `_ring`: viền kín bao dòng, lòng không đổi | dev2: +2 vỏ đúng | đo (dev2) |
| 09/10 | gt m03_s1 | vỏ nhạt bắt nhầm pill mà bản xoá còn giữ (chỉ nhạt màu) | [code] bỏ vỏ khi bản xoá còn mép vỏ (`SOFT_EDGE` 0.6; vỏ thật 0.01-0.39, ca nhầm 1.05) | gt S F1 0.800 -> 0.842 (= trước lượt vỏ nhạt) | đo (gt + dev2) |
| 09/10 | dev2 (62) | too_small: VLM ghi cỡ = cỡ nháp (trung vị 0.99) mà chữ vẫn tràn ngang ngay từ đầu vì **cỡ nháp đo phồng**: dòng chữ thường nhiều chữ số đo ra cao chữ hoa rồi chia cao chữ thường ("Hotline 0866 777 888" 1.37×) | [code] dòng không toàn hoa có >= 15% ký tự cao: cỡ = cao nét cao (p90) / 0.72 | synth chữ biết cỡ (315 dòng, 7 font): p90 cỡ ước / thật 1.37 -> 1.18 | đo |
| 09/10 | dev2 (62) | too_small: khung dòng nắn theo lớp phủ co về một mẩu khi chữ nhỏ / nhạt ngoài vỏ ("25 Nguyễn Huệ, TP. Huế" 281 -> 30 px; 16 dòng) | [code] khung nắn ngoài vỏ < 0.6 bề ngang OCR -> giữ bề ngang OCR (`SNAP_MIN_W`) | gt L F1 0.967 -> 0.976 | đo |
| 09/10 | dev2 (62) | (hai dòng trên gộp lại) | replay: lệnh VLM cũ, cỡ px nhân theo cỡ mới / cũ | too_small 50 -> 39, ảnh có lỗi 35 -> 28 | đo (replay) |
| — | dev2 d26, d12 | khung tối chứa nhiều dòng / khung ngày không thành S | dò vỏ quanh CỤM dòng | — | đề xuất |
| — | dev2 d06 v0 | vỏ dính chữ to bên cạnh (pill "Freeship" + "199K" thành một S) -> VLM vẽ thẻ trắng phủ cả giá | tách vỏ khỏi dòng chữ lớn kề | — | đề xuất |

#### Đợt HÌNH HỌC CHỮ (mở 09/10): chữ cong / lượn sóng / to dần / một dòng nhiều cỡ

Mô hình chung: mỗi dòng = **đường chân** (chuỗi điểm, trái -> phải) + **hồ sơ cỡ**. OCR vẫn lo đọc chữ, khung, góc dòng thẳng /
nghiêng; `textfix/geo.py` dò hình dạng từ NÉT (chuỗi mảnh nét cạnh nhau cùng cỡ -> điểm theo lát -> khớp đường tròn / đa thức /
đường thẳng + cỡ đổi đều) và gộp các mẩu OCR / chi tiết I trên cùng đường chân thành một ô; gộp mẩu cùng đường chân khác cỡ (OCR
tách "GIẢM" | "50%") thành ô có `runs`. **Tắt mặc định** (`TEXTFIX_GEO=on`) tới khi B3 + B4 xong.

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | synth 60 | OCR cắt cung thành mẩu thẳng / bỏ hai đầu cung; tách dòng nhiều cỡ | [code] `geo.arcs` + `geo.runs` | cung nhận 0% -> 59%, đúng 1 ô 30% -> 76%; nhiều cỡ đúng 1 ô 5% -> 73% | đo (synth) |
| 09/10 | synth 90 | thêm lượn sóng, to dần | [code] mô hình đường chân chung | cung: có đường chân 60%, lệch 0.05 cỡ · sóng 42% (lệch 0.08, đa số coi là cung) · to dần 25% (chiều đúng 100%) · chéo 100% đúng 1 ô, sai góc 0.5° · nhiều cỡ 67% | đo (synth) |
| 09/10 | gt, dev2 | không được làm hỏng dữ liệu thật | — | gt L / S / I không đổi; dev2: nhận đúng cung d10 v0; 1 nhận nhầm nhiều cỡ (chữ trang trí 10-16 px -> chặn cỡ >= 24 px) | đo |
| — | — | dựng chữ theo đường chân (SVG textPath, cỡ từng ký tự), mô tả cho VLM | B4 + [prompt VLM] | — | đang làm |
| — | dev3_geo | kiểm trên nháp FLUX thật | `%run scripts/dev_batch.py bench/dev3_geo.json` | — | chờ server |

### B3 Designer

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 d24 | chép chữ méo của nháp: "CUỐI TỬA TUẦN" | [code] lệnh viết câu khách: bỏ chữ không có trong câu khách nào | 1 lệnh sửa, 0 nhầm / 62 ảnh | đo (dev2) |
| 09/10 | dev2 d25, d06 | chữ in trên sản phẩm VLM cho skip (tự ghi "printed logo on product") -> mất con dấu | [code] skip có lý do "printed / on product" -> keep | 2 lệnh đổi, cả 2 đúng | đo (dev2) |
| 09/10 | dev2 d27 v0 | VLM viết "00:36", "GIÁ", "VND" lên màn hình nồi cơm | [code] số trong lệnh không khai câu khách phải trùng NGUYÊN số của câu khách ("00" không còn khớp chuỗi con của "1.590.000đ"); [prompt VLM] màn hình / nhãn / bao bì sản phẩm luôn keep / skip | "00:36" bị chặn; "GIÁ", "VND" chờ prompt | đo (code) · chờ server (prompt) |
| 09/10 | dev2 d12 v1 | viết tắt địa chỉ "Quận 3" -> "Q.3" | [prompt VLM] dữ kiện không viết tắt; [code] đo câu thiếu bắt được (báo "Không vẽ") | — | chờ server |
| 09/10 | dev2 | VLM bỏ câu mà không khai `missing` | [code] đo câu thiếu bằng mã (`slots.uncovered`), cộng vào `missing` | +5 ảnh được báo đúng, 0 báo oan | đo (dev2) |
| 09/10 | dev2 d05 | danh sách bỏ trống hàng giữa (nháp vẽ thừa hàng) | [prompt VLM] điền hàng từ trên xuống, chỉ bỏ hàng thừa cuối | — | chờ server |
| 09/10 | dev2 d18 v0 | thẻ trắng lồng trong thẻ trắng | [prompt VLM] dòng trong vỏ không tự thêm nền / viền | — | chờ server |
| 09/10 | dev2 d30 v0 | khối chữ căn giữa ở nháp thành căn trái | [prompt VLM] giữ căn lề từng khối | — | chờ server |
| — | dev2 d02 v1 | tiêu đề vàng kim thành trắng phẳng | luật đã có ("background-clip:text … gold"), VLM không theo | — | theo dõi |
| — | dev2 d16 | VLM tự viết nhãn "Khai Trương" (khách không ghi tiêu đề) | luật hiện cho phép nhãn chung, cấm dữ kiện | — | **cần quyết** |

### B4 Dựng + kiểm

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 d06 | "DROP" thành "DRQP": dấu sắc dòng dưới đâm vào chữ dòng trên (line-height 1.0) | [code] line-height >= 1.15 khi dòng sau có chữ hoa mang dấu | hết | thử (dựng lại) |
| 09/10 | dev2 d05 | bảng giá "Bún bò tái40.000đ" (cột flex trong thẻ font) | [code] thẻ bọc `display:contents`; không nén ngang khối flex | hai cột đúng, chữ to bằng nháp | thử (dựng lại) |
| 09/10 | dev2 d10 | emoji 🎁 thành ô vuông | [code] bỏ emoji | hết | thử (dựng lại) |
| 09/10 | dev2 d11 | "40%" cam trên nền đỏ gần vô hình, không bị bắt | [code] tương phản đo từng đoạn chữ | không bắt -> bắt + đổi màu | thử (dựng lại) |
| 09/10 | dev2 d17 | trích dẫn dài: mỗi dòng một chữ mồ côi, chữ co nhỏ | [code] đoạn văn bỏ `<br>` theo nháp, tự ngắt | chữ to hơn, hết mồ côi | thử (dựng lại) |
| — | dev2 | overflow: sửa bằng code 0/7 | — | — | theo dõi |

## Tập dev 3 (bench/dev3.json, soạn 09/10)

40 yêu cầu, 62 ảnh -- dữ liệu MỚI để kiểm các sửa trên dev2 có tổng quát không, và mốc cho đợt hình học chữ:

| Nhóm | Ca | Ảnh | Kiểm gì |
|---|---|---|---|
| typography khó | g01-g16 | 32 | chữ cong trên / dưới, vòng con dấu, ruy băng cong, chéo, nhãn nghiêng, lượn sóng, so le, một dòng nhiều cỡ / nhiều font |
| prompt thật dài | p01-p11 (prompt_test.txt) | 15 | feedback dày chữ (tiêu đề, mô tả, điểm nổi bật, ưu đãi), review 5 sao, before / after, ngoặc cong, loại để mặc định promo |
| form đủ + prompt sơ sài | f01-f10 | 12 | "làm poster đẹp", "thiết kế giúp mình", không dấu, "banner", emoji; 6 loại form, ảnh sản phẩm, loại để sai |
| ảnh thường | i01-i03 | 3 | không chữ -> chỉ FLUX; "poster cho tiệm hoa" không cho chữ -> không bịa chữ |

## Kết quả dev3 (09/10, mã có đủ sửa dev2; `TEXTFIX_GEO` tắt)

**Dữ liệu:** 61/62 ảnh (f03 lỗi "y1 must be >= y0" -- đã sửa c8bd4a9). Zip 01 tải lần đầu rỗng, lần hai bị VS Code mở như văn bản
lưu lại (byte nhị phân thành U+FFFD) -- đừng mở zip trong trình soạn thảo.

**Mốc:**

| | dev3 (61 ảnh, 3 ảnh thường) | dev2 (62) |
|---|---|---|
| Lỗi lượt đầu | too_small 49 · boxes_overlap 21 · low_contrast 11 · overflow 2 · uneven_size 2 · mark_twice 1 | too_small 50 · low_contrast 12 · overflow 8 · uneven_size 8 · boxes_overlap 6 |
| Sửa bằng code | 21 lần, giữ 18 | 20, giữ 14 |
| Lỗi cuối | too_small 51 · boxes_overlap 4 · uneven_size 2 · overflow 1 | too_small 50 · overflow 7 · uneven_size 8 · boxes_overlap 1 |
| too_small / ảnh | typography 0.7 · form 0.6 · **prompt dài 1.4** (câu khách TB 342 ký tự) | 0.8 |
| Câu thiếu | 14 ảnh (VLM khai + mã đo; VLM bỏ không khai: 0 -- mã đã cộng vào) | 9 + 5 |
| Thời gian | job trung vị 65 s, tối đa 137 s; ảnh thường 4-8 s | — |

**Chấm các sửa của lượt dev2:**

| Sửa | dev3 | Kết quả |
|---|---|---|
| B0 khung ảnh bằng lời | 4/4 ca 16:9 "horizontal / landscape"; dọc "vertical / portrait" | **đạt** |
| B0 ảnh thường | i01, i02 ảnh thường; i03 "poster tiệm hoa" không cho chữ -> ảnh thường, không bịa chữ | **đạt** |
| B0 [prompt LLM] form + prompt sơ sài là poster | f10 (form feedback + "tạo ảnh"): LLM vẫn trả `intent: image`, bỏ hết chữ | **không đạt** (prompt) |
| B0 [code] thêm lại trường form | f10 cứu thành poster đủ 2 trường; f01 ngày gộp bị bỏ -> thêm lại; f04 | **đạt** |
| B0 prompt thắng form | dev3 không có ca mâu thuẫn | chưa chấm |
| B2 dọn 2 lượt + xác nhận OCR nháp | OCR bản dọn: 0 chữ đọc được còn sót (chỉ hàng sao p01); lượt 2 chạy 1 ảnh (p02) sạch | **đạt** phần chữ đọc được; còn sót chữ nghiêng / bóng (xem dưới) |
| B2.5 cỡ chữ + khung không co | too_small / ảnh 0.8 -> 0.6-0.7 (ca thường) | **đạt một phần** |
| B2.5 `SNAP_MIN_W` | gây lỗi f03 (khung lật khi giữ bề ngang OCR) | **không đạt** -> sửa c8bd4a9 (gt L F1 0.976 -> 0.980) |
| B3 [code] skip "printed" -> keep | g15 v1: ruy băng chữ POSTER thành keep -> chữ nháp sai chính tả "QUẢ TỆNG" lên poster | **không đạt** -> chỉ khi có ảnh sản phẩm |
| B3 [code] bỏ chữ không có trong câu khách / số nguyên | không bắt nhầm ca nào | đạt (không có ca để bắt) |
| B3 [prompt VLM] không thẻ trong thẻ | p02 v1 viên thuốc vẽ đôi lệch | **không đạt** |
| B3 [prompt VLM] giữ căn lề | p10 lề trái các mục so le | **không đạt** |
| B4 dấu chồng / flex / emoji / đoạn văn | không tái phát | đạt |

**Lỗi theo bước (xem 100% ảnh):**

| Bước | Lỗi | Ảnh |
|---|---|---|
| B0 | LLM xếp form + "tạo ảnh" thành ảnh thường (code cứu) | 1 |
| B1 | không chừa chỗ cho câu (f05 "Nhân viên bán hàng (5 người)") | 1 |
| B2 | xoá sót chữ trên nhãn nghiêng (g06 v0 "MỚI") / huy hiệu (f09); vệt bóng tiêu đề (g04 v0); FLUX vẽ thêm hình chữ X mờ (g08 v1) | 4 |
| B2.5 + hình học | **chữ cong**: OCR không đọc, cung thành 15 chi tiết I mỗi ô một chữ (g01 v0), vòng con dấu thành đoạn thẳng nhỏ (g02), cung cắt mảnh (g03, g12), chữ cung dưới huy hiệu mất (g12 v0) · **nhiều cỡ một dòng** mất điểm neo (g08 v0) · nhãn nghiêng bị dựng thẳng (g13 v0) | 9 |
| B3 | chữ khách bị keep pixel nháp (g15 ×2) · tiêu đề chéo nhét vào ô khác, gần như mất (g07 v0) · bỏ chữ trong tiêu đề ("GIAO" g09 v1, "sau" p01 v1) + tự ngắt dòng mồ côi · chia một câu vào hai ô cạnh nhau -> đảo thứ tự đọc (p04, p07) · icon / chấm đầu dòng trơ (p08, p11) · bịa "WEAR CHIC" từ chữ nháp méo (f09) · đổi hoa thành thường (f06) · vẽ đôi viên thuốc (p02) · sao 3/5 khi prompt "5 sao" (p11) | ~14 |
| B4 | tiêu đề chìm nền bị **đổi hẳn màu** xấu (g11 v1, g12 v1) · chữ vàng kim bị cắt mép (g14 v1) · chữ trên ruy băng chéo dựng rất nhỏ (g05 v0) | 4 |

**Sửa lượt này (code, đo bằng dựng lại plan dev2 + dev3):**

| Sửa | Bước | Kết quả |
|---|---|---|
| luật skip "printed" -> keep **chỉ khi khách tải ảnh sản phẩm** | B3 [code] | dev2 d25 vẫn đổi đúng; g15 hết đổi sai |
| keep mà chữ OCR của ô trùng câu khách (giống >= 0.6, hoặc mẩu OCR khớp liền >= 80%) -> viết lại bằng câu khách | B3 [code] | đúng 1 ca (g15 v0) / 123 plan; ca nhầm p09 ("Noinlien Hic" trên thân nồi) bị chặn bằng điều kiện khớp liền |
| icon / chấm đầu dòng đi kèm dòng bị skip -> skip | B3 [code] | đúng p08, 0 nhầm / 123 plan (p11: dòng không skip mà bị gộp -- theo dõi) |
| tiêu đề chìm nền (>= 1.4 cỡ trung vị hoặc >= 6% cạnh ngắn): **giữ màu, thêm viền tương phản**; kiểm low_contrast tin lệnh đã có viền | B4 [code] | g11 v1 vàng + viền tối, g12 v1 trắng + viền tối: đọc rõ, giữ thiết kế |
| chữ dải màu (`background-clip:text`) ở khung -> span ôm chữ | B4 [code] | g14 v1 "MÙA THU" đủ nét |
| khối nhiều dòng tràn ngang -> thử bỏ `<br>` nếu vừa khung ở nguyên cỡ | B4 [code] | p01 / f02 không áp (VLM tự ngắt mồ côi, không tràn) -- giữ cho ca tràn |
| chữ dải màu: span có đệm làm đo tràn oan (dev2 d02, d11 +3 too_small) -> bỏ đệm | B4 [code] | hết |
| **hồi quy toàn tập** (dựng lại 120 plan dev2 + dev3, mã cũ vs mới, cùng máy) | — | dev3: too_small 48 = 48, boxes_overlap 8 = 8 · dev2: too_small 52 = 52, boxes_overlap 4 = 4 -- không tăng lỗi; khác số với server (51 / 4, 50 / 1) là do máy local vs server (Chromium / font), không do sửa |

**Còn mở / cần quyết:**
- **[cần quyết] VLM tự viết chữ không có trong câu khách**: nhãn chung "Khai Trương" (dev2 d16, trông ổn) và bịa "WEAR CHIC" từ chữ nháp
  méo (dev3 f09, sai). Đề xuất: cấm hẳn -- ô không ứng câu khách nào thì skip (luật code chắc chắn hơn prompt).
- **[cần quyết] rút gọn câu khách**: prompt VLM cho rút gọn câu dài cho vừa chỗ -> tiêu đề mất chữ ("sau", "GIAO"). Đề xuất: tiêu đề /
  câu ngắn (<= 12 chữ) không được bỏ chữ; chỉ câu thân dài mới được rút gọn.
- Chữ cong: bật `geo` cần B4 dựng theo đường chân + mô tả cho VLM (đang làm); dev3 có nháp thật để tune dò cung.
- Dày chữ (prompt dài): too_small 1.4 / ảnh -- FLUX chừa chỗ ít hơn nhiều so với câu khách; cân nhắc dặn LLM viết prompt có khối chữ
  thân đủ chỗ.

## 12/10: bỏ nắn khung dòng, nhóm VLM khai là lệnh

Chỉ thị của người dùng (12/10): thẩm mỹ số 1, coi trọng nháp hơn prompt; chữ được co trong khung OCR; bỏ hẳn bước nắn khung;
VLM đánh dấu các khung cùng ngữ nghĩa -- danh sách PHẢI cùng lề + cùng cỡ, một ý xuống dòng PHẢI cùng cỡ / cùng kiểu; một dòng
phải co thì cả nhóm co theo. Lượt này chỉ làm dòng chữ (L); vỏ (S) và chi tiết (I) giữ nguyên, quan sát thêm.

Nguyên nhân đo được (5 lượt chạy 12/10 + 120 poster dev2 / dev3): bước nắn khung theo nét mực (`slots._snap`) đẩy mép trái vào
ở 11-25% dòng, cắt thấp 15-39% dòng (nét mực = nháp - bản xoá đo thiếu trên nền ảnh chụp) -> danh sách thò thụt, chữ co. VLM đã
khai nhóm đúng (`group`) nhưng mã chỉ coi là gợi ý (lề chỉ căn khi hình học khớp, cỡ có ngoại lệ 0.65).

| Bước | Thay đổi | too_small | overflow | boxes_overlap | content_overlap | uneven_size |
|---|---|---|---|---|---|---|
| mốc | (mã 2e061e0, dựng lại 120 plan server) | 100 | 8 | 12 | 0 | 10 |
| A | [code] B2.5 khung dòng = khung OCR nguyên bản (xoá `_snap`, `_unstack`); B4 lệnh chữ MỘT dòng chỉ kiểm tràn NGANG (`hfit`: nền đã xoá sạch, dấu thanh nhô khỏi khung không đè vào đâu) | 32 | 3 | 21 | 2 | 11 |
| B | [code] B4 nhóm VLM khai (`group`) thi hành không điều kiện (`render._group_frames`, `declared_groups`): cỡ = trung vị cỡ nháp rồi cả nhóm lấy cỡ nhỏ nhất vừa khung; nhóm xếp cột: một lề theo `align`, vị trí = trung vị mép các khung OCR, bề ngang = hình bao nhóm (không ra ngoài vỏ, không lấn phần tử cùng hàng); `flow`: thêm font / đậm / màu của lệnh đầu. Nhóm có dòng nghiêng hoặc viết vào ô I: bỏ qua (dev3 g01_v0: 15 mẩu chữ cung) | 29 | 3 | 21 | 2 | 10 |

- A: 51 / 120 poster ít lỗi hơn, 7 nhiều lỗi hơn (xem mắt cả 7: không poster nào xấu đi; `boxes_overlap` tăng vì khung OCR thô
  của các dòng sát nhau chạm nhau, chữ dựng ra không đè nhau). B: 26 / 120 poster có nhóm VLM khai, 4 ít lỗi hơn A, 0 nhiều hơn.
- 5 lượt 12/10 (dựng lại plan server): mép trái danh sách 131 / 92 / 55 / 83 -> 65 / 65 / 65 / 65 (85be), cỡ danh sách 15 ->
  21-27 px (nháp 25-28); lỗi cuối 4 too_small -> 2 too_small + 1 uneven_size.
- [prompt, CHƯA đo với VLM] designer: `group_kind` ("list" | "flow") + `align` trên mỗi lệnh; mục danh sách viết hoa chữ đầu;
  ký hiệu đầu dòng: nền còn thì chỉ viết chữ, nền mất và không có ô I thì tự viết ("•" / `<i-icon>`). LLM brief + mã: mục
  `list_item` viết hoa chữ đầu (cả trong ngoặc kép của prompt FLUX). Groq miễn phí không chạy được lời dặn designer (17k token
  một lượt, giới hạn 7k / phút) -> phải đo trên server.
- [code, phòng thủ] nhóm VLM khai GIỐNG HỆT: ngoài cỡ + lề còn một font / độ đậm / màu / nghiêng / giãn chữ / hoa-thường (giá
  trị nhiều dòng dùng nhất, CSS cưỡng), cỡ bằng nhau tuyệt đối (bỏ dung sai 3%). Lý do: VLM viết từng dòng riêng, cùng `group`
  mà ghi đậm 400 / 300 / 400, cỡ 28 / 29 / 27 (lượt 08aa). 120 plan dựng lại: lỗi không đổi so với bước B (0 poster xấu đi).
  [prompt] designer phải tự viết mọi lệnh của một nhóm y hệt nhau (cùng font, đậm, cỡ, màu) -- mã chỉ là lưới đỡ.
- Còn thấy: chấm đầu dòng bị bản xoá xoá mà không thành ô I (85be, 7229) -- chờ lời dặn mới; đoạn văn trong thẻ (08aa) vẫn nhỏ
  vì câu khách dài hơn chỗ nháp vẽ (người dùng chấp nhận co trong khung).
- Tắt nhóm-là-lệnh: `TEXTFIX_GROUP_FRAME=0`. Công cụ: `scripts/replay_plans.py` (mặc định dựng bằng khung OCR; `RRBOX=snap` =
  khung nắn cũ lưu trong plan trước 12/10).

### 12/10 chiều: 5 lượt e2e đầu với lời dặn mới (run 320d, 7a00, 3fc7, 777c, 5d27)

Chấm lời dặn: cả 5 lượt khai nhóm kèm kiểu + cách căn; 7 nhóm >= 2 lệnh, 6 đúng kiểu (7a00 khai danh sách thành flow), 6 viết y
hệt nhau; chấm đầu dòng có ở cả 4 lượt có danh sách, không đôi. 777c, 5d27 đẹp; 3 poster còn hỏng vì lý do khác:
- 320d: hai cột chữ ở đáy, số ô xen kẽ trái / phải, VLM gộp theo số liền nhau (L6 + L7 + L8) -> hai khối đè nhau.
  [prompt] COLUMNS: các ô của một lệnh phải chồng dọc cùng cột; chỉ mảnh của một dòng OCR cắt đôi mới gộp ngang.
- 7a00: nháp 3 gạch đầu dòng, VLM nhồi 3 ý vào gạch đầu, bỏ gạch cuối -> cả nhóm co 0.70. [prompt] mỗi gạch nháp vẽ một ý.
- 777c: chấm đầu dòng thành icon vòng tròn rỗng to. [prompt] chấm nhỏ -> "• ", icon chỉ khi nháp vẽ icon / tick.
- 3fc7, 320d: câu ưu đãi dài co 0.63 - 0.70 trong vỏ. [prompt] chữ trong vỏ rút còn từ chính, giữ cỡ nháp.
- 5d27: luật chống lặp xoá "tổng vệ sinh" giữa huy hiệu vì trùng tên "An Gia Tổng Vệ Sinh". [mã] cùng cụm chữ nằm trong hai câu
  khách: mỗi lệnh khai một câu riêng chứa cụm đó là hợp lệ.
- brief (3fc7, 777c, 5d27): danh sách vào texts HAI lần (từng mục + cả câu gốc làm body, nối vào cuối prompt FLUX) từ khi mục
  danh sách viết hoa chữ đầu: phép so "các mục đã phủ chuỗi trong ngoặc kép" phân biệt hoa thường. [mã] so casefold.
- Ghi cho lượt S / I: 3fc7 vỏ S1 dính đường viền bị xoá (43..982 thay vì ~240..760) -> thanh navy đè khung.
120 plan dựng lại sau các sửa: không đổi (too_small 29, overflow 3, boxes_overlap 21, content_overlap 2, uneven_size 10).
4 lời dặn mới CHƯA đo với VLM thật.

### 12/10 tối: vỏ (S) và chi tiết (I) -- giữ, sửa chỗ sai

Mốc (bench/si_labels.json, chấm mắt một lượt 130 poster): 65 vỏ -- ~50 vẽ lại đúng; sai dáng 2, bỏ nhầm vỏ có chữ 2, vỏ giả 1,
ô dính khung bị xoá 1 (3fc7), chữ trong vỏ co nhỏ 5 (phần L). 163 chi tiết -- 88 bỏ (gần hết là mảng ẢNH bước xoá vẽ lại khác),
25 giữ, 26 icon, 21 chữ cung (lượt chữ khó).
- [mã] vỏ "keep" = dán điểm ảnh nháp theo mặt nạ vỏ + lấp nét chữ cũ trong vùng dòng (inpaint), không lấp sát mép / viền.
  Thử ép cả 57 vỏ giữ được (scripts/shell_keep_probe.py): 9 lộ lỗi (bóng chữ cũ ở vỏ trong mờ / loang 4, vỏ vỡ mép 3, mặt
  nạ ăn chữ bên cạnh 1, chữ cung nháp còn lại 1) > ngưỡng 3 -> KHÔNG làm mặc định. Thắng rõ: vòng tròn viền (7252_v1), ô vỏ
  to hơn vỏ thật (3fc7: dải khung đáy trở lại nguyên), vỏ chỉ ôm một dòng (9283_v0).
  [prompt] S keep chỉ cho dáng CSS không vẽ được + một màu phẳng + không có chữ ngoài các dòng L, và khi ô S to hơn vỏ thấy
  trên nháp. Thử lọc "lòng phẳng màu" bằng mã (>= 85% lòng cách màu lòng <= 12 ΔE): loại nhầm 22 vỏ tốt, vẫn lọt 6 ca lỗi -> bỏ.
- [mã] vỏ chứa chữ được viết mà bị skip (hoặc keep không lấp được) -> vẽ lại theo số đo, trừ khi màu lòng đo trùng màu chữ
  (2ad2_v1) hoặc lệnh chữ bên trong đã tự có nền / viền (dev3 g04: VLM vẽ pill ngay trên lệnh chữ, vẽ thêm là vỏ lồng vỏ).
  Trên 130 plan chỉ còn nổ ở dev2 d26_v0 (hộp tối sau "Mở cửa 5:00 - 23:00" trở lại, thêm 1 boxes_overlap). [mã] vỏ giả: màu lòng cách màu mực dòng bên trong < 8 ΔE -> không tạo ô S (2 ô <= 0.4, ô kế
  16.5): bỏ 2871_v0 "Pool Party" và 2ad2_v1 (vỏ thật nhưng màu lòng đo sai thành màu chữ -- trước đó VLM cũng bỏ).
- [prompt] I: thứ không phải chữ mà nháp vẽ ổn (viền, kẻ, hoạ tiết, logo, mảng ảnh) -> keep; phân vân skip / keep -> keep.
  [mã] chi tiết keep không kéo theo chữ nháp của dòng không giữ (4 poster cũ đổi: 2 sạch hơn, 1 như cũ, 1 nhoè hơn ở chữ in
  trên vật mà dòng bị skip).
- plan.json thêm "si": đếm S / I theo quyết định (dấu * = mã đổi). 120 plan dựng lại: too_small 29, overflow 3, content_overlap 2,
  uneven_size 10 không đổi; boxes_overlap 21 -> 22 (d26_v0 ở trên).
CHƯA đo với VLM thật: VLM có chọn keep đúng chỗ không, có giữ nhầm chữ méo không.

### 12/10 tối: 4 lượt e2e sau sửa S / I (run 28aa kombucha, e97e gym, 38ef bootcamp, 93f1 thợ điện nước)

Chấm: mỗi gạch đầu dòng một ý + viết hoa đúng ở 2 / 3 poster có danh sách; nhóm viết y hệt đúng cả 4; keep cho S / I và luật
cột chưa có dịp thử (không poster nào có vỏ dáng lạ, mảng ảnh bị đổi, hai cột chữ). Lỗi thấy:
- 38ef: hàng lặp giữa danh sách bị VLM bỏ -> thủng một hàng. [mã] `slots._pack_lists`: hàng bị bỏ nằm giữa các hàng của một
  danh sách VLM khai, cùng cột, cùng số dòng -> dồn các ý lên, hàng trống xuống cuối.
- 38ef: tiêu đề 10 chữ / 3 dòng bị luật "đoạn văn" ngắt lại thành 2 dòng ("kỹ / sư"). [mã] dòng cỡ tiêu đề (>= 1.4 trung vị)
  giữ cách ngắt VLM viết.
- 93f1: ba hàng danh sách -- icon to, hai icon nhỏ chồng nhau, không icon. [mã] `slots._icon_sets`: từ nửa số hàng (>= 2) có
  icon thì mỗi hàng đúng một icon (thừa -> giữ cái trên, ô icon bị VLM bỏ -> vẽ lại bằng icon hay dùng nhất), cả bộ cùng cỡ
  trung vị + thẳng một cột (`render._even_icons`). [prompt] icon của danh sách là một bộ.
- 28aa: huy hiệu bước xoá giữ lại mà VLM bỏ hết chữ bịa bên trong -> huy hiệu trống. [prompt] vỏ còn trên nền không để trống:
  viết một ý ngắn từ câu khách. 28aa: VLM tô thanh nền cho từng dòng danh sách. [prompt] tương phản bằng bóng chữ, không nền riêng.
- CHƯA sửa (chờ người dùng quyết): chữ nhồi vào ô quá nhỏ rồi co (28aa mô tả 110 ký tự vào dòng 20 ký tự: 0.54; e97e ưu đãi
  hai dòng vào ô một dòng: 0.46; 93f1 tiêu đề 0.66) -- đề xuất một vòng hỏi lại VLM riêng cho lệnh co < 0.6; 93f1 các hàng danh
  sách lặp nội dung (một câu khách "a, b, c, d" cho ba hàng) -- đề xuất mã tự chia ý; tiêu đề đúng số dòng nháp thì co ngang
  thay vì ngắt thêm dòng.

### 12/10 tối: bước brief -- chữ phụ trợ do LLM viết, tách "Điểm nổi bật" bằng mã, tập form1

Người dùng thật điền đủ các ô form (sự thật: ưu đãi, giá, ngày, liên hệ) nhưng ô prompt trống hoặc một câu mơ hồ; trước đây
LLM bị cấm thêm chữ -> poster thiếu tiêu đề / nút, FLUX tự vẽ chữ vô nghĩa cho đầy, VLM phải đoán (huy hiệu trống ở 28aa).
Yêu cầu không chữ nào ("tạo poster quảng cáo quán của tôi") ra ảnh trơn.
- [prompt + mã] texts có "source": "user" (đúng từng chữ như cũ) | "ai" (tối đa 4 câu, <= 48 ký tự: tiêu đề khi khách không
  cho, khẩu hiệu, nút kêu gọi, nhãn mục). Mã bỏ câu "ai" có số / % / tiền / liên kết hoặc chữ cam kết - ưu đãi (giảm, tặng,
  miễn phí, bảo hành, cam kết, số một...). Designer thấy câu "ai" gắn nhãn OURS: được rút gọn / đổi chữ / bỏ; không tính là
  "câu khách thiếu". LLM nói "image" và không có chữ khách -> bỏ hết câu "ai" (ảnh con bò vẫn là ảnh).
- [mã] trường form highlights / feedback_highlights: tách theo ", " / ";" / xuống dòng thành từng list_item (không tách "1,5 lít"),
  thay chuỗi gộp trong ngoặc kép của prompt bằng từng ý. Chữ trong ngoặc kép của prompt lấy đúng cách viết hoa của texts.
- Thử bằng LLM giả 6 ca (form khuyến mại, điểm nổi bật gộp / đã tách, prompt mơ hồ, ảnh con bò, bịa không khai): đúng cả 6.
  CHƯA đo với LLM thật.
- bench/form1.json: 22 ca mô phỏng người dùng điền form (6 loại x tối thiểu / điển hình / đủ; không tiêu đề; gõ thường, không
  dấu; prompt mơ hồ không chữ; 2 ca ảnh thường). Chạy: `%run scripts/dev_batch.py bench/form1.json`.

- 38ef: hàng sao kèm chữ "5 sao" (prompt "hiển thị đánh giá 5 sao" -> LLM đưa "5 sao" vào texts). [mã] brief: câu chỉ ghi số
  sao mà khách không đặt trong ngoặc kép / không điền ở form -> bỏ khỏi texts, prompt tả "a row of N gold rating stars";
  designer: lệnh có thẻ sao thì bỏ mẩu chữ chỉ ghi số sao. [prompt] cả hai bước: sao là hình, không viết "5 sao" cạnh hàng sao.

### 12/10 tối: bộ form1 (22 ca người dùng điền form), LLM / VLM thật

22 / 22 chạy xong. Bước brief: 2 ca ảnh thường đúng là ảnh; "Điểm nổi bật" tách đúng ở 4 / 4 ca ("1,5 lít" không bị tách);
LLM viết chữ phụ trợ ở 11 / 20 ca poster, không câu nào mang số / giá / tên bịa; chữ không dấu của khách giữ nguyên.
- f17 "tạo cho tôi ảnh feedback dịch vụ tiếng anh" bị coi là ảnh thường (nháp có bong bóng chữ loạn, không ai sửa).
  [prompt] người Việt gọi poster là "ảnh": "ảnh feedback / quảng cáo / khuyến mãi...", banner, menu = poster.
- f02: LLM ghép hai ô ngày thành "20/10 - 31/10" -> bị bỏ (không nguyên văn), hai ngày rời không lên poster. [mã] câu chỉ
  gồm các giá trị khách nối bằng dấu -> là chữ khách. f12: câu phụ trợ bọc giá trị khách ("Tuyển dụng Nhân viên bán hàng")
  -> giữ, đánh dấu như chữ khách (designer không được rút gọn); phần LLM viết vẫn qua bộ chặn.
- f07 "Grand Opening" (tiếng Anh), f16 "GỌI ĐIỆN THOẠI ĐẶT BÀN" khi khách không cho số. [prompt] chữ phụ trợ luôn tiếng Việt,
  nút kêu gọi không trỏ tới liên hệ khách không cho.
- f09: câu trích 85 ký tự co còn 0.32 (12 px). [mã] LỖI DỰNG CHUNG: html là chữ trần lẫn span ('chữ <span>đậm</span> chữ')
  đặt thẳng vào khung flex -> mỗi mẩu thành một cột xếp ngang. Có chữ trần ở cấp ngoài thì bọc một span. Dựng lại f09: 32 px.
  120 plan cũ dựng lại: tổng lỗi không đổi.
- Chưa sửa: f16 tiêu đề co 0.54 (VLM chèn một dòng trống đầu lệnh); f15 khách cho 6 bước, nháp vẽ 5 hàng đánh số 1 2 3 5
  (giới hạn model vẽ); f11 tên gói dịch vụ không có chỗ trên nháp.

### 12/10 đêm: chữ hình dạng khó (cung, dòng to OCR sót) -- TEXTFIX_GEO=on, mặc định còn TẮT

Mốc (xem mắt 32 nháp g01-g16 của dev3): hỏng 5 / 7 nháp chữ cung, mất hẳn tiêu đề "FLASH SALE" (OCR không ra dòng), 2 / 4 nhãn
nghiêng nhiều dòng; chữ nghiêng một khối và một dòng nhiều cỡ đã ổn.
- NGUỒN ĐO: bản đồ vùng chữ của chính bộ dò trong RapidOCR (DBNet), lấy TRƯỚC khâu ép khung 4 điểm (`ocr.regions`). Bộ dò thấy đủ
  vòm chữ, vòng con dấu, "FLASH SALE", dòng giữa nhãn dán; chúng chỉ mất ở khâu ép khung. Không thêm model.
  1307 vùng / 162 nháp: mọi vùng hiển thị (cao dòng >= 3.5% cạnh ngắn) võng >= 0.3 cao dòng là chữ cong thật (dòng thẳng <= 0.24).
- Đã thử và BỎ: đo từ nét mực (nháp - bản xoá). `geo.arcs` nhánh "lượn / to dần" bắt nhầm 30 / 130 poster thường; dải nét
  (`geo.bands`) bắt nhầm 36 / 130 (chữ V, A, W làm đường tâm gợn), phải ba ngưỡng đặt tay mới về 0. Mã còn trong geo.py, không gọi.
- [mã] B2.5 `slots._geo_lines`: (a) vùng cong -> một ô L có đường chân, gộp mẩu OCR + chi tiết trên nó; (b) vùng không có dòng
  OCR, là lớp phủ bị gỡ (nét mực >= 10% vùng), bộ đọc ra >= 2 chữ / số -> ô L mới (không làm khi khách tải ảnh sản phẩm: con dấu
  bánh pía, chữ trên lon cũng là "vùng không dòng OCR"). 162 nháp: ô cung ở 8 nháp chữ khó + 2 poster thường có tiêu đề cong
  thật; ô dòng sót: "FLASH", "MAI", "SAPE", "I vi" (4 nháp không ảnh sản phẩm). Không ô nào đổi sai.
- [mã] B4: lệnh chữ viết vào một ô cung dựng bằng SVG textPath theo đường chân (kiểu chữ đo từ chính lệnh; nén rồi co như dòng
  thẳng); kiểm chồng theo dải chữ dọc đường (`_multi_ov`), không cân cỡ / căn lề / cắt khung với dòng thẳng. B3: mô tả "CURVED
  LINE ... room for about N characters", "OCR missed this line", ảnh khoanh ô vẽ đường cong; lời dặn hai đoạn.
- Dựng thử 9 nháp (lệnh VLM cũ, câu của ô cung gán bằng mã): cả 9 ra đúng dáng nháp. 120 plan cũ dựng lại: tổng lỗi không đổi.
- CHƯA đo: VLM thật với ô cung / ô dòng sót; nhãn nghiêng nhiều dòng (góc chung, dòng "MỚI" chưa thành ô); "FLASH" + "SALE" xếp
  chồng ra một vùng dính (một ô, cỡ đo 244 px); ca g08_v1 chưa tìm nguyên nhân; bản RapidOCR trên server có lấy được bản đồ
  không (log "bộ dò: N vùng chữ" / "KHÔNG lấy được bản đồ").
  Chạy: `export TEXTFIX_GEO=on` trước khi khởi động máy chủ, rồi `%run scripts/dev_batch.py bench/dev3.json --only g01,...,g16`.

### 13/10: chữ cong đi vào đầu nguồn (sau 2 lượt e2e GEO bật: ba93 con dấu tiệm bánh, 1077 donut)

Chấm lượt trước: server LẤY ĐƯỢC bản đồ vùng chữ ("bộ dò: 3 / 5 vùng chữ") -- **đạt**. VLM thật viết đúng vào ô cong ở 3 / 3 ô
(mỗi ô một lệnh, tự ghi "curved lines redrawn by renderer") -- **đạt**. Hai poster vẫn hỏng, cả hai do mã, cùng một gốc: dòng
cong sinh ra SAU khi vỏ / chi tiết đã đo, nên các luật có sẵn không biết đến nó.
- 1077: ba chữ "KHÓ" đậm của dòng cong là một mảng to, gọn, đặc -> thành "vỏ" nâu; luật vỏ giả (màu lòng = màu chữ) chạy lúc
  chưa có dòng nào trong đó; sau đó luật "vỏ chứa chữ được viết thì vẽ lại" tô mảng nâu đè lên chữ. Ô cong không có màu đo.
- ba93: bộ dò thấy vòm "TIỆM BÁNH NHÀ" và mẩu "GẠO" là hai vùng rời trên cùng một đường tròn; VLM gộp hai ô vào một lệnh; bộ
  dựng gặp "ô cong + ô khác" lùi về khối chữ thẳng trong khung bao, đè lên ổ bánh; bộ kiểm không báo gì.
- [mã] B2.5 `slots._layers` + `build`: phần hình học tạo dòng nào (cong / OCR sót) thì vỏ và chi tiết ĐO LẠI với danh sách dòng
  cuối; vùng chữ của dòng đó (`mask` = vùng bộ dò + mẩu đã gộp + dải dọc đường chân) là chữ với mọi luật của `overlay` như khung
  của dòng thẳng (không thành vỏ, không thành chi tiết, không tìm vỏ nhạt quanh khung thẳng của một vòm). Dòng nằm trong vỏ nào
  tính theo điểm giữa chữ (`_anchor`), không theo tâm khung thẳng. Màu ô cong đo trên lõi nét bị gỡ.
- [mã] `geo.circle_of` + `apply_arcs`: vùng cong khớp đường tròn (lệch <= 0.15 cao chữ) thì mẩu OCR / chi tiết có tâm trên đường
  tròn đó, cùng cỡ, bắt đầu trong vòng 2 cao chữ kể từ đầu cung (tính theo mép gần, không theo tâm) nhập vào ô cung; góc tính
  liền mạch quanh giữa cung (cung vắt qua 3 giờ không gãy).
- [mã] B3 `validate`: lệnh chữ có ô cong lẫn ô L khác -> cả câu viết theo đường cong (dài nhất), các ô L kia để trống, bỏ <br>;
  B4 `render.path_of`: lệnh có ô cong không bao giờ dựng thành khối thẳng. [prompt VLM] ô cong không chung lệnh với ô nào khác.
- Đo: dựng lại 2 lượt bằng lệnh VLM của server -- ba93 chữ chạy theo vòm, 0 lỗi (trước: khối thẳng đè hình); 1077 hết mảng nâu,
  còn 1 too_small (hàng chữ trên cùng, không liên quan). Thử nhánh gộp (lệnh [ô cong, ô thẳng]): tách đúng, chữ theo đường cong.
  Chống khớp sát: GEO tắt, 30 nháp dev2 / dev3 / form1 đo ô lại (cùng dòng OCR) giống hệt mã cũ; GEO bật, 165 nháp
  (`geo_measure.py`): log dò cung / dòng sót chỉ đổi ở ba93 (thêm "GẠO") -- không nháp nào khác bị nối thêm mẩu. Sau đo lại: ecda_v0
  thêm 1 chi tiết (biểu tượng nhỏ cạnh "MINIGAME", trước bị khung dòng OCR nuốt -- đúng là chi tiết).
- Theo dõi: dòng OCR sót mà vùng bộ dò chỉ ôm một phần mảng chữ to thì phần thừa thành chi tiết vụn (chỉ thấy ở 0767 khi ép chạy
  như không có ảnh sản phẩm; trên server ca đó không tạo dòng sót). Lệnh chữ cong không bóng chữ nằm sát bóng đổ của vật: kiểm
  tương phản tính trên khung bao của vòm (low_contrast 0.26 ở bản thử gộp) -- chưa sửa.
- KHÔNG làm: phép kiểm riêng "chữ dựng ra nằm đúng chỗ nháp vẽ chữ". Với dòng thẳng nó trùng phép kiểm tràn khung đã có; với
  dòng cong, lỗi nó định bắt (lùi về khối thẳng) giờ không còn đường xảy ra.
- CHƯA đo (cần server): VLM thật có còn gộp ô cong với ô khác không; nhãn nghiêng nhiều dòng; "FLASH" + "SALE" xếp chồng.

### 13/10: 3 lượt e2e đề bài dài (2638 nồi cơm, 1f85 studio, a9ed resort; mã TRƯỚC bản sửa chữ cong) -> đánh số theo khối, dấu đầu dòng

Xem mắt: 2638 hỏng (VLM gộp L5-L8 của HAI cột vào một lệnh, icon đặt vào ô dòng chữ, câu ưu đãi + 3 ý mất, bộ kiểm 0 lỗi);
1f85 hỏng (tiêu đề mất chữ "nghìn", mô tả co 0.54, ba ý co 0.46, icon lệch hàng); a9ed dùng được (mất 1 ý, danh sách 19 px so
với nháp 26). Lời dặn COLUMNS của 12/10: **không đạt** (2638 gộp đúng theo số liền nhau).
Người dùng chốt (13/10): nháp vẽ ít hàng hơn số ý thì CHỊU (không coi là lỗi, không vẽ lại nháp); câu dài là việc của VLM, chỉ
viết lời dặn cho gọn, KHÔNG thêm vòng hỏi lại.
- [mã] B2.5 `slots._renumber` / `_xy_order`: số ô L đánh theo KHỐI -- cắt đệ quy tại khe trống rộng nhất, cột trái hết rồi mới
  sang cột phải; ngoại lệ BẢNG (một bên là cột giá trị, đa số ô toàn chữ số, mỗi ô cùng hàng với một ô bên kia): theo hàng "tên
  rồi giá". 159 plan đã lưu: thứ tự đổi ở 55 poster; lệnh chữ nhiều ô mà các ô KHÔNG liền số 23 -> 5, cả 5 là lệnh VLM gộp nhầm
  (2638, 320d x2, 472a hai dòng đáy cách nhau cả poster, cb04 nhảy qua một dòng) -- tức gộp đúng nay luôn là số liền nhau, gộp
  nhầm thì không. Bảng giá dev2 2963 (8 cặp tên - giá VLM ghép theo hàng) giữ nguyên thứ tự. [prompt VLM] đoạn COLUMNS thay bằng
  BLOCKS (ngắn hơn một nửa).
- [mã] B2.5 `slots._row_marks` (`TEXTFIX_ROWMARK=0` để tắt): chi tiết nhỏ (cạnh dài <= 1.6 cao dòng, gần vuông) ngay bên trái
  một dòng, cùng hàng, cách <= 1.2 cao dòng -> nhập vào Ô CỦA DÒNG (khung nới sang trái, "mark" trong plan.json), không còn là
  ô I; VLM viết dấu ở đầu html của dòng. 159 plan: 21 chi tiết ở 6 poster, xem mắt cả 21 đều là tick / icon đầu dòng (VLM cũ:
  15 icon, 6 skip vì hàng bị bỏ); loại được nét viền khung quanh dòng (99a1) nhờ điều kiện gần vuông. Icon to, đứng xa, không ở
  bên trái dòng nào vẫn là ô I. Lưu ý: `slots_gt` chấm các dấu này là I -> F1 lớp I sẽ tụt theo thiết kế, chưa cập nhật đáp án.
- [mã] quan hệ chi tiết - dòng: dòng CÙNG HÀNG trước, rồi mới dòng gần nhất (1f85: ba icon bên trái ba dòng đều ghi "below L5").
- [mã] B3 `_guard` 1a: chữ không có trong câu khách nào, đứng đúng chỗ một chữ của câu lệnh khai -> THAY bằng chữ của khách
  ("ngàn" -> "nghìn"), trước đây bị xoá. Thử 6 câu: chệch 1 chữ / chữ hoa được thay; rút gọn hợp lệ, nguyên văn không đổi; chữ
  thừa cuối câu vẫn bị bỏ; sai dấu vẫn sửa như cũ. `validate`: lệnh icon / sao viết vào ô dòng chữ -> skip.
- [mã] B4 `_uneven`: độ đậm chỉ so giữa các lệnh VLM khai cùng nhóm (a9ed: nhãn "Điểm nổi bật" đậm đứng thẳng cột với ba mục).
- [prompt VLM] FIT: một đoạn duy nhất thay ba chỗ dặn rải rác -- chỗ là chỗ nháp vẽ, số ký tự nháp viết là trần, câu dài VIẾT LẠI
  NGẮN bằng chữ của khách, có ví dụ; không đổi chữ khách bằng từ đồng nghĩa. Row marks viết lại theo "row mark in front".
- Đo: dựng lại 139 plan (dev2 62, dev3 58, form1 19) mã cũ / mới: 138 không đổi, d24_v1 hết 1 uneven_size báo nhầm. Ba plan mới:
  2638 hai icon nhầm ô bị bỏ, a9ed hết uneven_size. Đo ô lại ba nháp: 2638 đoạn mô tả L5-L8, ba gạch L9-L11; 1f85 ba icon nhập
  vào L6-L8.
- CHƯA đo (cần VLM thật): VLM có hết gộp hai cột không; có viết dấu đầu dòng trong lệnh chữ không; FIT có làm câu ngắn lại không.

### 13/10: chữ nghiêng / xếp chồng (32 nháp chữ khó của dev3, dựng lại bằng lệnh VLM cũ)

Xem mắt 32 nháp: 8 nháp có vòm đều ra ô cong đúng. Hỏng còn lại: nhãn nghiêng nhiều dòng (g05_v0, g13), sticker (g06), tiêu đề
xếp chồng (g07_v0, g09_v1), chữ vòng quanh huy hiệu (g12), chữ trên ruy băng 3D (g15).
- [mã] B4 XOAY HAI LẦN: VLM ghi `transform: rotate(31deg)` ở thẻ bên trong, bộ dựng xoay thêm 31 độ (g05_v0 nhãn "GIẢM 30%" dựng
  đứng; g13_v1 -20 thành -42). Bộ dựng gỡ mọi rotate / skew của VLM trong lệnh nó tự xoay. [prompt VLM] dòng "drawn at N degrees":
  không tự ghi rotate / skew; các dòng nghiêng xếp chồng trong một nhãn là một lệnh.
- [mã] B4 góc của khối = góc của dòng DÀI nhất (dài / cao), khi đa số dòng nghiêng; trước là trung vị (g13: "ĂN" hai chữ cái đo
  -4 độ, "NGON" -19 -> -11).
- [mã] B3 `validate`: `<br><br>` trong lệnh chữ -> một `<br>` (g09_v1: "CƠM / NGON / (trống) / TẬN / (trống) / NƠI").
- [mã] B2.5 `_stacked`: vùng "OCR sót" có khe không nét ở giữa -> tách hai dòng. g07_v0 KHÔNG tách được ("FLASH" và "SALLE" của
  nháp đó chờm lên nhau, không có khe) -- còn là một ô; để ngỏ.
- Đo: dev3 58 poster dựng lại, danh sách lỗi không đổi ở cả 58; xem mắt g05_v0, g13_v1 đúng góc nháp, g09_v1 hết lỗ. 130 nháp
  (`geo_measure`): log dò cung / dòng sót không đổi.
- Chấp nhận cho bản 1 (đề nghị): chữ nhỏ vòng quanh huy hiệu, chữ trên ruy băng 3D. g06_v0: OCR sót hẳn dòng "MỚI" của sticker,
  bản xoá còn chữ cũ -> chữ mới đè lên chữ sót; chờ xem với GEO bật.

## Bộ accept1 (13/10, 42 ca, mã 95edeef, GEO bật) -- chấm và sửa; từ đây accept1 là tập DEV

42 / 42 chạy xong. Lỗi cuối (bộ kiểm): too_small 13, boxes_overlap 5, low_contrast 1, content_overlap 1. Xem mắt 100%: trên 39
poster (bỏ 1 ảnh thường, 2 giới hạn đã biết g05 / g12): đẹp 15, dùng được 9, HỎNG 15 (38%). Bộ kiểm chỉ báo lỗi ở 6 / 15 poster
hỏng; f03 (không một chữ) báo 0 lỗi.

Chấm lượt trước: số ô theo khối -- **đạt** (0 lệnh gộp hai cột / 41 poster; c01, c02, a01 đúng). Nhóm list cùng kiểu -- **đạt**.
"nghìn" giữ nguyên, uneven_size không báo nhầm -- **đạt**. Chữ cong: g01, g02, g03 **đạt**; g04, g06 hỏng do mã. Nghiêng: g07,
g08, b03 **đạt** (không còn xoay hai lần). FLASH / SALE: **đạt** (g09). Vỏ keep răng cưa (s02), chữ trên lon (p01): **đạt**.
FIT (câu dài viết ngắn): **nửa đạt** -- 8 poster còn lệnh co < 0.7. Dấu đầu dòng: mã nhập ở 7 poster, VLM vẽ đúng 3, vẽ vòng
rỗng 3, không vẽ 2 -- **nửa đạt**. Luật "chữ chệch" 1a của 13/10: **không đạt** (a01 thay bừa ba chữ).

| Bước | Poster hỏng | Sửa |
|---|---|---|
| B0 brief | f08: form + prompt, LLM khai price / highlights bị prompt đè -> giá mất, nút giá trống | người dùng chốt 13/10: PROMPT LUÔN THẮNG FORM -> gỡ luật "ô sự thật luôn giữ" của Claude. [prompt LLM] chỉ xung đột thật (prompt ghi giá trị khác cho CÙNG ô) mới đè, prompt chỉ thêm chữ thì không đè ô nào; [mã] giá trị form có nguyên văn trong prompt thì không tính là bị đè. Chưa đo với LLM thật |
| B1 nháp | a02 chữ đáy quá nhỏ; danh sách ít hàng hơn số ý (a01, a04, b02, b05, f06, c03...) | chấp nhận (giới hạn model, quyết định 13/10) |
| B2.5 ô | c03: hàng chấm nối làm OCR gộp tên + giá | [mã] `_split_gaps`: dòng OCR có >= 3 chấm liền -> cột chỉ có nét thấp (<= 0.3 cao dòng) là khoảng trống. Đo ô lại: tách đúng "Trà đào cam sả" / "35.000đ" |
| | g04: hai vòm lồng nhau, vòm sau nuốt vòm trước | [mã] `apply_arcs` không gộp ô đã là ô cung. Đo ô lại: hai ô cung |
| | g06: hai dòng nghiêng xếp chồng nhận nhầm là vòm (võng 0.30, dài 2.8 cao dòng) | [mã] `geo.ARC_SPAN` = 3.2: 18 vùng võng của accept1 / dev2 / dev3, 17 cung thật dài 3.7 - 12.3, ca nhầm 2.8. Đo ô lại: không còn ô cung; ba mẩu thành chi tiết I (bộ đọc không đọc được vùng xiên 51 độ) |
| | f02, a05: vỏ đo được một nửa (nửa kia khác bản xoá < DIFF) | [mã] `overlay`: nới vỏ ra điểm liền kề cùng màu lòng mà bản xoá có đổi (SOFT_DIFF). 109 vỏ / 187 nháp: 7 vỏ đổi khung, xem mắt cả 7 đều ôm đúng hơn (f02, a05, s02, dev2 07fe, dev3 0319 / 8c94 / 97f6), 0 vỏ xấu đi |
| | b03: mảnh chữ số "2" thành vỏ; s03: 2 / 3 huy hiệu tròn viền không thành vỏ | theo dõi (mỗi loại 1 poster; thử "vỏ nằm trong vùng chữ của bộ dò": bộ dò chỉ phủ 32% mảnh đó, không đủ tách) |
| B3 VLM | f03: kế hoạch rỗng | [mã] `engine.design`: không lệnh chữ nào -> gọi lại một lần; vẫn rỗng -> lỗi nặng `empty_plan`. Chưa đo (cần VLM) |
| | b05: hai lệnh cho một ô (chữ + keep) -> hai lớp chữ đè | [mã] `validate`: một ô một lệnh, viết chữ > vẽ > giữ > bỏ. Dựng lại: hết đè |
| | f06: đổi thứ tự bước cho khớp hình nháp | [mã] `_list_order`: danh sách một cột theo thứ tự câu khách. Dựng lại: đúng thứ tự |
| | g10: khai "GIẢM" 34 px và "50%" 229 px một nhóm | [mã] `render.GROUP_SIZE` 1.4: lệnh có cỡ nháp lệch xa trung vị nhóm không thuộc nhóm. Dựng lại: đúng tỉ lệ nháp |
| | p02: tên sản phẩm viết vào hàng danh sách, luật chống lặp bỏ tiêu đề thật | [mã] nhiều ô cùng giống câu -> giữ ô chữ to nhất. Chưa dựng lại được (plan lưu sau bước chặn) |
| | b06, c01, f02: chấm đầu dòng vẽ bằng icon vòng tròn / "○" | [mã] `validate`: vòng tròn đầu dòng -> "•". Dựng lại: b06, f02 đúng; c01 còn chấm đôi (nền còn chấm, VLM vẫn thêm) |
| | s03: VLM gộp ô vỏ vào lệnh chữ, vẽ nền cho từng dòng | chấp nhận (VLM) |
| B3 mã | a01: luật 1a thay chữ không liên quan | [mã] chỉ thay khi giống >= 0.6 và cùng chữ đầu. Thử: "ngàn" -> "nghìn" vẫn thay; câu a01 ra "Cơm chín đều, mềm dẻo / giữ vị bữa cơm nhà" |
| B4 dựng | a04: danh sách ba dòng có icon trong MỘT lệnh thành ba cột hẹp | [mã] luật đoạn văn bỏ qua danh sách; SVG icon ("flex:none") không tính là cột flex. Dựng lại: ba dòng |

Đo chung: dựng lại accept1 (41) + dev2 (62) + dev3 (58) + form1 (19) mã cũ / mới: danh sách lỗi đổi ở 2 poster (b06, f02: bớt
lỗi), 0 poster thêm lỗi; luật mới chạm 2 poster cũ (dev3 p04 chấm đầu dòng, p06 ô hai lệnh).
Sau cụm sửa, đếm lại 15 poster hỏng: 7 đã thấy đúng khi dựng / đo ô lại tại máy (a04, b05, c03, f06, g04, g10, f02); 5 phải
chạy server mới biết (a01, f03, f08, p02, g06); 3 để nguyên (a02 nháp vẽ chữ quá nhỏ, b03, s03).

### 13/10: lời dặn designer viết lại gọn (41.200 -> 15.400 ký tự), `TEXTFIX_PROMPT=v1` để dùng bản cũ

- Danh mục icon: trước gửi cả 1544 tên Lucide (19.900 ký tự, 48% lời dặn). 207 poster đã lưu: 26 poster dùng icon, 67 thẻ, 26 tên
  khác nhau, 20 tên phủ 91%; tên dùng nhiều nhất là `circle` (15 lần -- cái vòng rỗng đầu dòng). Nay gửi 123 tên theo 10 chủ đề
  (`slots.ICON_HINTS`) + câu "tên Lucide khác vẫn dùng được" (bộ dựng tự tìm tên gần nhất).
- Thân lời dặn 19.000 -> 11.900 ký tự: sáu luật hay bị vi phạm nhất đưa lên đầu (nhìn theo nháp / chữ theo khách, FIT, sự thật,
  một ô một lệnh + không trả kế hoạch rỗng, mỗi câu một lần, danh sách giống nhau đúng thứ tự); gộp các đoạn lặp (FIT nằm ba
  chỗ, "cùng vai cùng kiểu" hai chỗ, vỏ lồng vỏ hai chỗ); bỏ lời giải thích đã có mã lo (xoay chữ, vòng tròn đầu dòng).
  Các từ khoá mà bảng mô tả ô dùng giữ nguyên ("CURVED LINE", "drawn at N degrees", "OCR missed this line", "row mark in front",
  "OURS").
- CHƯA đo (cần VLM): chạy lại accept1 là phép thử. So với bản cũ trên cùng seed: `export TEXTFIX_PROMPT=v1` rồi khởi động lại
  máy chủ, chạy `dev_batch.py bench/accept1.json --fresh` lần nữa -- bảng báo cáo tự so hai lượt.

## 13/10 chiều: CHỈ DÒNG CHỮ -- poster = nháp với chữ viết lại (`TEXTFIX_LONLY=on`, mặc định TẮT)

Người dùng chốt, thay quyết định 12/10 "giữ ô S và I": (1) không bỏ dòng nào nháp đã vẽ, thiếu câu khách thì VLM viết cho đủ;
(2) icon nhỏ trước mỗi dòng vẽ lại; (3) mọi thứ còn lại -- vỏ, icon to, hoạ tiết, sao, avatar -- là điểm ảnh của nháp.
Số đo dẫn tới quyết định (207 nháp): ô chi tiết I đo đúng ~0,7 và 88 / 163 ô là rác; bước xoá giữ avatar 8/8, icon tô đặc trong
thẻ ~9/10, gỡ dấu nhỏ nét mảnh cạnh chữ; vỏ 1-2% ảnh bị gỡ 16/16, 2-4% 69%, 4-8% 48%, > 8% chỉ 2/22.

- [mã] `textfix/lonly.py`. `view`: VLM và bộ dựng chỉ thấy ô L (dấu nhỏ đầu dòng đã nằm trong ô dòng). `draft_base`: nền dựng =
  NHÁP, trừ chỗ là chữ. Mỗi dòng: màu nền của dòng = màu chiếm nhiều nhất trong vùng dòng trên nháp; nền đó ở bản xoá còn
  nguyên (<= 8 ΔE) -> lấy điểm ảnh bản xoá trong vùng dòng; nền đổi (bước xoá gỡ cả vỏ) -> giữ vỏ của nháp, lấp nét chữ cũ
  [ĐÃ THAY 13/10 tối, xem mục "viền gai mờ" bên dưới: nền phẳng thì tô đè nét chữ bằng màu nền của nháp, không dán bản xoá]
  (inpaint); không có nền phẳng (chữ trên ảnh chụp) -> bản xoá. Dòng "keep" (chữ in trên vật): nguyên điểm ảnh nháp. Vùng chữ
  Chữ OCR bỏ sót: mảng bị gỡ chạm lõi một vùng chữ của bộ dò, phần lớn nằm ngoài mọi ô dòng, không trùm tâm dòng nào, bộ đọc
  đọc ra chữ / số -> bản xoá, để chữ méo không hiện lại (b03 chữ số "2"); icon bộ dò nhận nhầm là chữ thì giữ (b04 icon ngân
  hàng). Dòng trên nền phẳng: sau khi ghép, lau mọi nét còn sót trong vùng dòng (b04: bản xoá còn mẩu chữ cũ trong nút).
  `fix_ops`: lệnh chữ không có nền / viền / đệm riêng; VLM lỡ bỏ dòng thì chữ cũ vẫn được xoá, chỗ đó trống.
- [prompt VLM] `SYSTEM_L` (11.100 ký tự): chỉ ô L, kind text | keep; luật 2 "mọi dòng nháp vẽ đều được viết" -- dòng thừa lấp
  bằng (a) câu khách chưa có chỗ, (b) cụm rút từ câu khác của khách, (c) cụm tự viết không mang sự thật / cam kết; các bước,
  hàng tên - giá, ngày, liên hệ không bịa; không vẽ nền, viền, icon to; số bước và sao còn trên nền thì không viết lại.
- [mã] `engine.design`: nhánh L_ONLY; plan.json ghi `"slots_mode": "L"` và dòng log "nền = nháp trừ chữ: ...".
- Đo tại máy (`scripts/lonly_replay.py`, giữ lệnh chữ của VLM CŨ, chỉ đổi nền): xem bảng ở cuối mục khi có. Phép đo này chỉ trả
  lời phần NỀN; VLM có viết tốt hơn khi chỉ còn dòng chữ hay không phải chạy server.
- CHƯA đo (cần VLM): VLM có lấp đủ dòng không, chữ lấp có hợp không, có còn vẽ nền riêng cho chữ không.

### 13/10 tối: "viền gai mờ" quanh chữ ở chế độ chỉ dòng chữ -> nền phẳng thì TÔ ĐÈ nét chữ, không dán bản xoá

Người dùng xem lượt server đầu của chế độ mới: icon to của nháp giữ tốt, nhưng nhiều chỗ có viền gai mờ. Soi 6 nền (accept1 a01,
b01, b04, f02, f04, s03), ba kiểu vết đều do bước ghép nền:
- miếng bản xoá dán vào vùng dòng lệch tông nhẹ so với nháp -> lộ khung chữ nhật mờ (s03 tiêu đề, giá);
- vỏ bị bước xoá gỡ: lấp nét bằng inpaint với mặt nạ >= 25 ΔE, viền khử răng cưa còn sót và loang thành bóng chữ lốm đốm (f02 thẻ);
- chữ có quầng tối trên ảnh chụp bị coi là nền phẳng rồi lấp nét -> mảng vỡ hạt (f02 khối liên hệ); màu nền lấy trên vùng dòng
  đã nới nên nút ôm sát chữ bị coi là nét chữ, lau trắng phần giữa (b01 nút "Đặt lịch ngay").
- [mã] `lonly.draft_base` viết lại. Màu nền của dòng lấy trong LÕI dòng (nới 0,1). Khớp mặt bậc hai qua các điểm nền; nền phẳng =
  >= 90% điểm gần mặt (<= 12 ΔE) thì sát mặt (<= 6) (`FIT_MIN`; đo: chữ trong nút / thẻ / nền trơn 0,92-0,99, chữ trên gối, sàn
  gỗ có quầng 0,83-0,86). Nền phẳng -> nét chữ = mảng khác mặt nền có >= 30% nằm trong lõi dòng (viền nút, icon cạnh dòng nằm
  ngoài lõi nên không bị đụng) hoặc mảng nhỏ trong vùng dòng (dấu thanh, chấm đầu dòng); tô các mảng đó + viền 3 px bằng MẶT MÀU
  NỀN khớp từ nháp. Không phẳng -> bản xoá như trước. Bỏ hẳn inpaint và bước lau nét sót sau khi ghép.
- [mã] hai chốt đoán nhầm nền -> bản xoá: màu nhiều nhất trong lõi là nét chữ (các điểm màu đó đổi hẳn ở bản xoá, phần còn lại
  nguyên: chữ trắng đậm trên vòm cầu vồng, g03) [sửa tiếp bên dưới: chốt này thêm điều kiện màu đó rời thành nhiều mảng; `INK_MAX` đã bỏ].
- [mã] người dùng: "không được lem nhem, một số pill có dấu hiệu lem". Soi mọi nút / thẻ có chữ ở cỡ gốc: (1) nút có vệt sáng
  (a02 nút cam, nền tối có quầng đèn) -- mặt bậc hai lệch nền vài ΔE nên chỗ tô hiện hình chữ mờ -> BÙ CỤC BỘ: cộng trung bình có
  trọng số (Gauss 0,35 cao dòng) của (nháp - mặt) trên các điểm nền quanh nét; (2) nút vàng kim có ánh mà bước xoá gỡ (a03): không
  đạt `FIT_MIN` nên dán bản xoá = khoét một lỗ thấy mặt nước trong nút -> khi nền của dòng đã ĐỔI ở bản xoá, thử lại với ngưỡng
  gấp đôi (nền <= 12, lưng chừng 12-24 phải ít): nút có ánh qua, chữ có quầng mờ trên sàn gỗ (f02) vẫn về bản xoá; (3) mẩu nền
  ngoài nút bị mép vùng dòng cắt ra không còn tính là "mảng nhỏ" (phải nằm gọn trong vùng dòng đã co 2 px).
- [mã] xem mắt 30 nền của dev2 / dev3 / form1: (1) NÚT BỊ LAU TRẮNG GIỮA (d27 "-36%", "Mua ngay"; f13 nút xanh "Gọi 0977...") --
  chốt "màu nhiều nhất là nét chữ" bắt nhầm: chữ trắng trong nút màu trên nền trắng, bước xoá gỡ nút thì lòng nút "đổi hẳn" còn
  chỗ chữ trắng "nguyên" -> thêm điều kiện màu đó phải RỜI thành nhiều mảng (chữ cái); lòng nút liền một mảng (mảng lớn nhất
  >= 60%) thì là nền. Bỏ chốt `INK_MAX` (chữ đậm trong nút nhỏ + viền 3 px vượt 60% vùng dòng là chuyện thường). (2) SÓT CHỮ CÁI
  CŨ (d07 "5" của "50%"): chữ số to lệch khung OCR, lại dính viền bóng của vòng tròn -> mảng chạm lõi và nằm trọn trong vùng nới
  0,6 cũng là nét chữ; mảng chạm lõi nhưng bị từ chối thì tách riêng điểm CÙNG MÀU MỰC (<= 25 ΔE, `INK_SAME`) rồi xét lại.
  Mỗi dòng lấy bản xoá ghi lý do vào log ("bản xoá vì: L3 nền có vân 0.78; ...").
- GIỚI HẠN thêm: tiêu đề có viền màu đè lên mép nút cùng màu (d12 "SIÊU SALE 11.11" trên nút tím) -- nút bị tô loang lên trên;
  chữ cong / chữ trên ruy băng mà plan cũ không có đường cong thì sót chữ cái ngoài khung (d10, dev3 g15).
- Đã thử và gỡ: "bước xoá đổi >= 70% vùng dòng thì ép tô trên nháp" -- không cứu được huy hiệu chữ vòng nhỏ (g05) mà làm khối
  liên hệ có quầng tối trên sàn gỗ (f02) vỡ hạt trở lại.
- Đã thử và gỡ: "lan dần nền chuyển màu" (khớp mặt, nhận thêm điểm sát mặt, khớp lại) cho nút vàng kim nhiều dải sáng trải ngang
  poster (a01: một màu chỉ phủ 10% vùng dòng) -- không cứu được nút đó mà làm khối liên hệ trên sàn gỗ (f02) lem trở lại.
- GIỚI HẠN còn: nút kim loại nhiều dải sáng bị bước xoá gỡ (a01) mất phần dưới dòng chữ, chỉ còn hai đầu nút;
  huy hiệu nhỏ có chữ chạy vòng (g05) -- vùng dòng cong trùm cả huy hiệu, bản xoá đã gỡ huy hiệu nên dán bản xoá
  khoét mất một góc; vỏ có vân / chuyển màu gắt bị bước xoá gỡ thì mất phần dưới dòng chữ.
- [công cụ] `batch_report.py` ghi thêm TỜ ẢNH GHÉP `output/runs/_reports/<tập>/sheets/<tập>_NN.jpg` (mỗi tờ 3 ca, nháp | poster):
  bấm đúp trong cây thư mục JupyterLab là xem, không cần mở cổng máy chủ; zip nhẹ kèm các tờ này + poster cạnh dài 1024.
- [tập thử] `bench/mix1.json` (35 ca mới, seed 8001-8035): m đề dài, v vỏ / huy hiệu / thẻ, i icon / các bước, t bảng / thực đơn,
  c chữ cong / nghiêng / viết tay, p ảnh sản phẩm (3 ảnh chưa dùng ở accept1), f chỉ form, x đề mơ hồ / chữ trên ảnh chụp.
  Chạy MỘT lượt với `TEXTFIX_LONLY=on`, chấm bằng mắt, không so đường cũ (người dùng: không muốn bật tắt máy chủ).

### 13/10 chiều: chữ in trên sản phẩm khách tải PHẢI GIỮ

Người dùng: có ảnh sản phẩm thì LLM và VLM phải biết, chữ trên sản phẩm giữ nguyên, không thay bằng bất cứ gì.
- [prompt LLM] gợi ý khi có ảnh sản phẩm: không tả / không xin chữ trên sản phẩm, chữ poster đặt quanh sản phẩm.
- [mã] `engine.draw` đọc chữ trên ảnh sản phẩm (`product_ocr`); `slots.mark_printed`: dòng nháp có chữ OCR gần trùng một dòng
  trên ảnh sản phẩm -> `printed`; bảng mô tả ghi "PRINTED ON THE CLIENT'S PRODUCT"; `validate`: dòng `printed` luôn `keep` dù
  VLM ghi gì; luật "keep trùng câu khách thì viết lại" TẮT khi có ảnh sản phẩm (tên sản phẩm trên vỏ hộp trùng câu khách).
  [prompt VLM] `PRODUCT_NOTE` nói rõ không viết lại, không sửa, không dịch, không đặt câu khách lên sản phẩm.
- Đo 5 ca có ảnh sản phẩm (accept1 p01, p02; dev2 d13, d24, d31): mã nhận 13 dòng in trên sản phẩm (SAPPORO, PREMIUM BEER, Dove,
  INTENSE, REPAIR...), 0 dòng poster bị nhận nhầm; 4 dòng méo nặng mã không nhận ("DEIN TAON", "moo.m"...) -- VLM cũ đều đã giữ.

## Lượt chạy server tiếp theo

Cập nhật mã (docs/SERVER.md), khởi động lại máy chủ **và kernel**, rồi:
**Xem kết quả không cần tải các zip đầy đủ (13/10):** `dev_batch.py` chạy xong tự gọi `scripts/batch_report.py`: in bảng từng ca
ra terminal (lỗi còn, chữ co, câu mất, cờ nghi hỏng của `textfix/audit.py`, so với lượt trước: tốt lên / xấu đi / như cũ), ghi
trang `output/runs/_reports/<tập>.html` (mở qua `.../proxy/8088/outputs/_reports/<tập>.html`) và MỘT zip nhẹ vài MB ở
`output/runs_zip/<tập>_report_<giờ>.zip` (HTML + ảnh thu nhỏ + plan / brief / request) để gửi Claude chấm. `--full` mới gói đầy đủ.
Cờ nghi hỏng đo trên 39 poster accept1 chấm mắt: bộ kiểm hình học riêng bắt 6 / 15 poster hỏng, báo nhầm 6 / 24; gộp cờ bắt 11 / 15,
báo nhầm 6 / 24 -- dùng để theo dõi xu hướng và chọn poster cần xem, chưa đủ để quyết định tự động.

**Mốc `mix1` lượt đầu (chạy bằng 5fad3b5, TRƯỚC bản sửa nút bị lau giữa ef5904e; chấm thô trên tờ ảnh ghép, 34 poster):**
đẹp 14, dùng được 15, hỏng 5 (accept1 đường cũ: 15 / 39 hỏng; hai bộ đề khác nhau). Năm poster hỏng: c01 con dấu mất vòng
(gỡ chữ), x04 cầu vồng bị mảng kem cắt ngang (gỡ chữ), p02 mất danh sách + vòng giá và nút trống + nhãn trên gói thứ hai bị
xoá (VLM bỏ dòng; chữ sản phẩm không được giữ -- phải truy), t03 nút "Tặng Pepsi" trống, chữ chạy xuống đáy (VLM), t02 mất cột
ngày / buổi học (VLM bỏ dòng). Nhẹ hơn: v05 nút "Trả góp 0%" có khối đen ở giữa, m03 nút có vệt trắng, t01 / f05 mảng mờ đục
quanh chữ trên nền có vân, i05 huy hiệu trống, i04 / f02 chữ co quá nhỏ. Lỗi chia gần đều giữa GỠ CHỮ và VLM BỎ DÒNG (mỗi dòng
bỏ = một nút trống).

**mix1 lượt hai (bản 6476222, cùng seed; chấm thô 34 poster):** hết nút bị lau giữa (m03), mảng mờ quanh "THỰC ĐƠN" (t01), nhãn
tai nghe co nhỏ (i04); p02 VLM trả đủ lệnh nên đủ chữ và giữ nhãn gói. Còn hỏng: c01 con dấu mất vòng, x04 cầu vồng bị cắt, x03
chữ neon nhoè (VLM), t03 VLM tự viết "Gà giòn" thay giá combo; nhẹ: v05 nút có khối đen, f05 / i05 một nút trống (đã có
`fill_gaps`), f02 chữ trong nút đỏ bị đúp viền, t02 ngày dính liền tên lớp.
- [mã] `lonly.draft_base`, ba sửa đo trên zip mix1: (1) mảng "chạm lõi, nằm trọn vùng nới" chỉ là nét chữ khi cạnh dài <= 2 cao
  dòng (`GLYPH_MAX`) -- vòng con dấu không còn bị lau (c01); (2) nét chữ DÍNH LIỀN hình khác (mảng chạm mép vùng nới, >= 10% nằm
  ngoài vùng dòng, `SPILL`) hoặc phần tô trùm > 60% vùng dòng (`COVER_MAX`): bản xoá ĐÁNG TIN (đổi < 60% vùng dòng, `TRUST`) thì
  lấy bản xoá, không thì bỏ mảng dính ra -- x04 cầu vồng nguyên, f05 hết mảng trắng đục trên sương; (3) LẤY BẢN XOÁ THEO HÌNH NÉT
  CHỮ: chỉ điểm bản xoá khác nháp (hạt giống > 20 ΔE, lan tới > 8) trong vùng dòng, nới 2 px; đổi > 85% vùng dòng mới lấy cả
  vùng -- t01 giấy kraft sạch, vỏ của dòng bên cạnh không bị đụng. Kiểm 15 poster ba tập: không ca nào xấu đi.
- CÒN: v05 nút có ánh mà bản xoá gỡ (không nền phẳng + bản xoá không đáng tin) vẫn bị khoét -> việc của LaMa; c01 con dấu do
  chính bản xoá phá một nửa; v06 chữ trắng trên nút kem (chữ gần trùng màu nền) còn bóng mờ; d12 như cũ.
- [thử LaMa, CPU 6,5 s / ảnh 1024, big-lama.pt 196 MB] mặt nạ = `textmask.text_mask` nới 2 px. v05 nút cam bóng: nút nguyên, chữ
  hết, sạch; x03 chữ neon: sạch. HỎNG vì MẶT NẠ chứ không vì LaMa: a01 nút vàng kim (mặt nạ nuốt cả nút), c01 con dấu (nuốt cả
  con dấu, 18% ảnh), g05 huy hiệu (méo). => LaMa dùng được cho nhánh "nền không phẳng + bản xoá không đáng tin" KHI có mặt nạ nét
  chữ chặt; việc kế là mặt nạ, không phải model.

**GỠ CHỮ VIẾT LẠI: một mặt nạ nét chữ + LaMa (13/10 đêm).** Người dùng: zip không phải để vá lỗi lẻ; muốn đưa hướng chỉ dòng chữ
lên trần. Cây luật (nền phẳng? bản xoá tin được? ba cách lấp, hơn mười ngưỡng) thay bằng hai bước cho MỌI dòng:
- [mã] `lonly.ink_mask`: nét chữ cũ = hợp của (1) mảng cùng MÀU MỰC của dòng chạm lõi dòng, >= 70% nằm trong vùng dòng; (2) chỗ
  BẢN XOÁ khác nháp trong vùng dòng, khi bản xoá đổi < 60% vùng dòng (chỉ gỡ chữ); (3) mảng KHÁC MÀU NỀN của dòng chạm lõi, khi
  phần đó <= 60% vùng dòng. Nới 0,12 cao dòng (`TEXTFIX_MASK_GROW`; nới 0,05 thì LaMa vẽ lại hình chữ từ viền còn sót). Không
  bằng chứng nào: lõi dòng. + khung dấu đầu dòng, + chữ OCR bỏ sót.
- [mã] `lonly.draft_base`: có `big-lama.pt` (`TEXTFIX_LAMA`, mặc định `~/persistent-data/lama/big-lama.pt`; `off` để tắt) thì
  LaMa lấp mặt nạ trên NHÁP, ngoài mặt nạ nguyên điểm ảnh nháp, không điểm ảnh nào của bản xoá vào poster. Không có tệp / lỗi
  khi chạy: đường luật cũ (`_base_rules`).
- [đo] `scripts/erase_bench.py` (sót = ký tự OCR còn đọc được trên nền; đổi = % ảnh khác nháp; ngoài = đổi ngoài vùng dòng).
  54 poster (nửa số của mix1, accept1, dev2), bản LaMa ĐẦU (chưa có (3), nới 0,05): luật 6 poster sót / 47 ký tự, LaMa 2 / 22;
  đổi 7,19% và 7,51%. Hai số này KHÔNG bắt được nút bị khoét / bóng chữ mờ -- phải kèm xem mắt. CPU 12 s / poster.
- Xem mắt 20 poster với bản cuối: sạch hơn luật ở nút vàng kim (a01, a03), nút cam bóng (v05), bốn nút pastel kể cả chữ trắng
  trên nút kem (v06), giấy kraft (t01), cầu vồng (x04), homestay (f05). CHƯA được: con dấu (c01, vòng cùng màu mực bị gỡ cả),
  vòm chữ plan cũ (g04, còn nửa tiêu đề), nút giá của nệm bị khuyết một góc (f02), vòm cầu vồng bị mờ (g03).
  Bản cuối CHƯA đo lại bằng erase_bench (máy không GPU) -- xác nhận trên máy chủ.

**mix1 lượt ba (bản 363ea25, LaMa trên máy chủ; chấm thô 35 poster):** đẹp 19, dùng được 11, hỏng 5 (lượt đầu 14 / 15 / 5).
LaMa chạy cả 35 poster, không lỗi; bước dựng + gỡ chữ 6,1 s / poster, VLM 16,2 s. Lỗi đã ĐỔI CHỖ: chỉ 1 poster hỏng vì gỡ chữ (c01
con dấu); 3 hỏng vì VLM -- i04 và v02 VLM trả kế hoạch RỖNG cả ba lần gọi (chưa rõ vì sao; nay ghi câu trả lời thô vào log), i02
VLM bỏ dòng -> `fill_gaps` giữ chữ nháp méo, nhìn như chữ rác; t02 sai ngày giữa hai hàng của bảng. LaMa hơn luật: v05 nút cam
(hết khối đen), t01, f05, x03. LaMa kém luật: f02 nút đỏ nhỏ bị nuốt thành một gạch, x04 cầu vồng nhoè chỗ tiêu đề đè lên.
- [mã] `fill_gaps`: trước khi giữ chữ nháp, ghép một - một dòng bị bỏ với câu khách CHƯA lệnh nào viết, ngưỡng giống 0,4 (`LOOSE`).
  Thử lại trên lệnh thật: v02 11 / 11 dòng viết câu khách (trước 9, hai ô giữ chữ méo), i02 2 / 2, i04 6 viết + 4 giữ, f04 2 + 3.
  Dòng còn "giữ chữ nháp" là dòng THỨ HAI của một câu đã viết ở dòng trên ("Air 3", "30ml") -- chờ người dùng quyết: giữ hay xoá.
- [mã] VLM trả rỗng: không gọi lần thứ ba; log ghi 300 ký tự đầu của câu trả lời thô.

**Mặt nạ nét chữ, vòng hai (14/10, tập luyện, xem mắt 9-15 poster mỗi lần vì máy không GPU):**
- [mã, giữ] `ink_mask`: (a) mảng nét chữ nằm >= 50% trong vùng của DÒNG KHÁC thì dòng này không lấy (nút nhỏ nằm dưới một vòm chữ;
  accept1 f02 nút giá hết khuyết góc); (b) bản xoá đáng tin là trọng tài: mảng mà nó chỉ đổi dưới một nửa thì chỉ lấy phần đổi;
  (c) bằng chứng bản xoá bỏ phần có màu nền của dòng (lòng nút nhỏ bị gỡ) khi vẫn còn phần khác màu nền.
- [ĐÃ THỬ VÀ GỠ] LẤP HAI NGUỒN (chỗ bản xoá đáng tin lấy điểm ảnh bản xoá theo hình nét chữ, còn lại LaMa): x04 cầu vồng dưới tiêu
  đề đẹp hoàn toàn và f02 thẻ sạch, NHƯNG sinh bóng chữ ở nút pastel (v06), hộp trích dẫn (f05), nút cam (v05), huy hiệu vàng
  (c02, x04) -- "đáng tin" theo tỉ lệ đổi không loại được vỏ nhạt bị gỡ; siết thêm "nền của dòng còn nguyên" vẫn còn bóng. Về lại
  LaMa cho cả mặt nạ. Muốn thử lại phải có phép đo bóng chữ tự động trước.
- CÒN: mix1 f02 nút đỏ nhỏ vẫn thành một gạch; x04 cầu vồng nhoè dưới tiêu đề (LaMa); chữ cong.

**BÀI THI accept2 (bản cbf5d89, LaMa trên máy chủ; chấm thô 30 poster):** đẹp 21, dùng được 7, hỏng 2 (7%) -- đạt ngưỡng 10%.
Hỏng: h01 tiêu đề ba dòng chồng nhau trên ảnh món ăn (VLM), l04 đường thời gian chồng chữ + một mốc bịa (VLM). Không poster nào
hỏng vì gỡ chữ; 3 ca ảnh sản phẩm giữ đủ chữ; không kế hoạch rỗng (h01: VLM viết 1 / 6 dòng, gọi lại được 6). ~34 s / poster
(VLM 12, dựng + LaMa 7, xoá + dọn 6,5, LLM 5, vẽ nháp 2,7). Người dùng nêu hai lớp lỗi về HƯỚNG CHỮ:
- CHỮ NGẢ NGHIÊNG dù nháp thẳng: OCR đo góc từ cạnh trên đa giác, dòng ngắn sai vài độ, mã dựng nghiêng từ 2 độ. 1.505 dòng của
  sáu tập: 2-4 độ 57 dòng, 4-10 độ 34 (phần lớn từ ngắn), >= 10 độ 40 (nghiêng thật).
  [mã] `slots._line_angle` ở `lines()`: < 4 độ -> 0; 4-10 độ chỉ giữ khi cạnh dài / cạnh ngắn >= 3; >= 10 giữ. Trên plan đã lưu:
  73 dòng về thẳng, 58 dòng giữ nghiêng (accept1 b03 -6,0 / -5,4 độ được giữ).
- CHỮ VIẾT DỌC bị viết ngang bé tí (h04): OCR trả đa giác cao - hẹp, góc 0, chữ rác. [mã] `slots._vertical`: đa giác cao >= 2,2
  rộng thì đọc lại theo +90 / -90 (`ocr.read_rect`), lấy chiều tin hơn, ghi góc ±90 + chữ đọc được. Dựng lại h04 tại máy: hai
  dòng dọc dựng dọc, đọc được. 6 / 1.505 dòng (4 dòng là chữ in trên sản phẩm).
- [mã] `fill_gaps`: dòng không còn câu nào để viết -> XOÁ TRỐNG (người dùng chốt; trước giữ chữ nháp = chữ giun).
- Dựng lại accept2 tại máy (không LaMa, không autofix): bộ kiểm 12 -> 16 poster có lỗi, phần tăng là low_contrast (nền luật) và 3
  boxes_overlap (e03, k01, l04) -- phép dựng lại bỏ `fix` của autofix nên chưa kết luận; xác nhận trên máy chủ.
- [công cụ] `scripts/ab_lonly.sh <đề>`: chạy hướng cũ rồi chỉ dòng chữ cùng seed; báo cáo xếp nháp | lượt trước | lượt này và gói
  poster + plan lượt trước vào cùng zip. `bench/icons1.json`: 12 đề nhiều icon (seed 9101-9112) để so hai hướng.

**DẤU ĐẦU DÒNG BỊ ĐÚP (14/10, người dùng: "● • Quét mã QR", bản xoá ĐÃ gỡ chấm mà poster vẫn còn):** ở chế độ chỉ dòng chữ nền
là NHÁP, nên dấu của nháp chỉ mất khi nó thuộc vùng gỡ của một dòng. `_row_marks` chỉ gắn dấu lấy từ danh sách chi tiết I của
lớp phủ; đo 9 dòng có dấu (accept2 + mix1): bản xoá gỡ dấu 6 / 9 mà không dòng nào được gắn (dấu rơi ở bộ lọc chi tiết). 507
dòng: mã gắn 8, VLM viết dấu 27.
- [mã] `slots._draft_marks`: dò dấu NGAY TRÊN NHÁP bên trái mỗi dòng thẳng -- mảng gọn, khác nền >= 25 ΔE, cạnh 0,12-1,6 cao
  dòng, cách chữ 0,12-1,2 cao dòng, nền cửa sổ phẳng, không đụng ô dòng khác. Loại: >= 3 mảng trong dải / hàng chấm li ti (chấm
  nối tên - giá), chấm nhỏ đứng lẻ không thẳng cột với chấm dòng khác, vòng tròn đọc ra số (số bước). Sáu tập, 1.519 dòng: thêm
  13 dấu ở 9 poster, xem ảnh 10 đúng / 3 nhầm (hạt trang trí).
- [mã] `lonly.lead_marks` (sau validate): dòng mà VLM VIẾT dấu ở đầu lệnh chữ nhưng chưa có dấu gắn -> dò lại đúng dòng đó với
  luật nới. Thử trên 8 poster có danh sách đã lưu: chỉ thêm 1 dấu -- ở các poster còn lại chấm "•" đã nằm trong khung OCR của
  dòng (được gỡ cùng chữ, không đúp). Ca của người dùng (chấm trắng cách chữ một quãng) không có tệp để dựng lại: CHƯA xác nhận.
- CÒN: dòng có khung OCR cao bất thường trùm hàng trên (accept2 q02) che mất dấu của hàng đó.

**14/10: CHỈ DÒNG CHỮ LÀ ĐƯỜNG DUY NHẤT CỦA MAIN.** Người dùng chốt sau accept2 chạy lại (chữ dọc đã dựng dọc) và phép so trên đề
nhiều icon: đường cũ (ô vỏ S + ô chi tiết I) lưu ở nhánh git `legacy-slots-si`, không vào main nữa. `lonly.LONLY = True` (bỏ cờ
`TEXTFIX_LONLY`), `TEXTFIX_GEO` mặc định `on`, gỡ `scripts/ab_lonly.sh`. Lệnh chạy: `bash scripts/start_server.sh` rồi
`python scripts/dev_batch.py bench/<tập>.json`, không cần đặt cờ. CÒN NỢ: các hàm chỉ đường cũ dùng trong slots.py / render.py /
overlay.py (base_image, SYSTEM, describe, ô S / I) chưa cắt khỏi main -- cắt sau, từng phần, có chạy máy chủ xác nhận.
- [mã] góc nghiêng 4-10 độ: thêm điều kiện có dòng khác nghiêng cùng góc (lệch <= 2,5 độ, `TILT_PAIR`). accept2 q02 (dầu gội):
  "Hương dịu nhẹ" đo 8,4 độ do OCR gộp hai mẩu lệch hàng -> nay thẳng; accept1 b03 (-6,0 / -5,4, một cụm) và k03 (nhãn dán
  14-22 độ) vẫn nghiêng; h04 hai dòng dọc ±90.
- LƯU Ý ĐO: VLM được gọi KHÔNG đặt nhiệt độ / seed -> hai lượt cùng đề có thể khác nhau chỉ vì VLM (cache theo nội dung câu hỏi
  chỉ khoá khi câu hỏi y hệt). So trước / sau bằng số lượt chạy lại là chưa công bằng cho tới khi khoá (chờ người dùng quyết).

**TẬP NGHIỆM THU `bench/final1.json` (32 ca mới, seed 9501-9532):** a đề dài (4), b nút / thẻ / bảng (6), c icon / danh sách (5),
d chữ khó gồm vòm, con dấu, nghiêng, DỌC, viết tay, viền (6), e nền khó (4), p ảnh sản phẩm (3), r form thường gặp (5). Chạy
MỘT lượt trên main, không cờ; chấm đẹp / dùng được / hỏng. Ngưỡng nghiệm thu đã đề xuất: hỏng <= 10%, không mất chữ sản phẩm,
không số / giá / liên hệ bịa. Người dùng quyết KHÔNG khoá nhiệt độ VLM.
- [mã] `audit.flags` thêm cờ của đường chỉ dòng chữ: dòng xoá trống, mã tự lấp >= 2 dòng, VLM viết thiếu phải gọi lại, gỡ chữ
  không dùng LaMa.
- Giới hạn đã biết, KHÔNG sửa trước nghiệm thu (đã thử, hoặc cần phép đo bóng chữ): chữ chạy vòng trên con dấu, vòng tròn mảnh
  cùng màu quanh chữ bị gỡ, nút rất nhỏ, cầu vồng / hình nằm dưới tiêu đề nhoè; VLM đặt câu dài vào ô ngắn (lời dặn đã có, VLM
  không theo); VLM chồng chữ / lẫn cột trong bảng.

**14/10: BỎ BẢN XOÁ FLUX -- Hi-SAM tìm nét chữ + LaMa (chặng A của docs/ROADMAP.md).** Người dùng chốt: TextFix CHỈ sửa chữ; main
chuyển hẳn, không so hai đường (đường cũ: nhánh `legacy-flux-erase`); icon / dấu đầu dòng là của nháp; cụm nét OCR không đọc
để VLM quyết; LaMa một lần mỗi poster.
- [đo trước] bản xoá chỉ còn là bằng chứng cho mặt nạ: dựng mặt nạ có / không bản xoá trên 141 nháp, 1.126 dòng -- 74% dòng như
  nhau, 21% bản xoá làm mặt nạ TO hơn (quầng, nhưng cũng loang trên nền gỗ / ảnh bị vẽ lại), 6% làm NHỎ hơn (chặn luật màu nền).
- [đo máy chủ, 2x A30, 30 nháp accept2] Hi-SAM (SAM ViT-L + `sam_tss_l_textseg.pth`): fp16 0,17 s / ảnh, mặt nạ trùng fp32 >= 0,997,
  đỉnh VRAM 4,4 GB (card trống). 232 dòng cần viết: không dòng nào sót hẳn; yếu nhất chân trang rất nhỏ (e02, phủ 6% ô).
- [mã] `textfix/hisam.py`; `engine.draw` = nháp -> nét chữ (bỏ FLUX xoá + dọn chữ sót, ~6,5 s); `slots.build(draft, ink)`: dòng OCR
  + chữ cong (nét mực = nét Hi-SAM) + `_orphans` (cụm nét chưa dòng nào nhận -> ô L `orphan`) + `_lead_dots`; `lonly.erase_mask` /
  `erase` / `settle`; `lonly.split_far`, `marks_rule`; `fill_gaps` giữ dòng `orphan` VLM không nhắc. Xoá: `cleanup.py`, `overlay.py`,
  `textmask.py`, `flux.erase`, đường luật `_base_rules`, bộ dò dấu đầu dòng, 12 script chỉ đường cũ dùng.
- [đo tại máy] đo dòng mới so với ô đã lưu, 65 nháp (accept2 + mix1): 531 / 532 dòng cũ còn, 7 / 7 dòng cong, thêm 11 cụm nét ở 4
  poster (chữ vòng trên huy hiệu trà, chữ trên ba ảnh sản phẩm), 4 dòng có chấm đầu dòng.
- [mặt nạ, xem mắt 10 nháp khó mỗi vòng] (1) quầng trong dải sát nét so với nền của dòng: nút / thẻ nguyên, con dấu giữ vòng ngoài
  và hạt cà phê (mix1 c01), cầu vồng nguyên (x04), nút đỏ nhỏ nguyên (f02) -- ba giới hạn cũ đã hết; (2) nền phẳng thì lan theo
  đuôi NHẠT của quầng: chữ 3D đỏ có bóng (accept2 e04) hết bóng chữ mờ; (3) gạch nối / dấu câu trong lõi dòng cũng gỡ (e01
  "300 Mbps - 245.000đ"); (4) quầng không lấn vùng dòng khác. ĐÃ THỬ VÀ BỎ: nền ước từ vành ngoài vùng dòng + lan mọi thứ khác
  nền liền với nét (lòng nút, vòng con dấu thành "quầng": mặt nạ 15 -> 30% ảnh); lan cả điểm đậm (nút đỏ, hạt cà phê, đầu cầu vồng).
- [dấu đầu dòng] icon trước dòng: Hi-SAM không coi là chữ, khung OCR không ôm -> để nguyên, mã gỡ icon khỏi lệnh VLM. Chấm tròn
  "•": OCR gộp vào khung dòng và Hi-SAM coi là nét (accept2 r04: 3 / 3) -> gỡ cùng chữ, dòng mới mở đầu "• " (`l["dot"]`).
- [ca người dùng gửi] VLM gộp năm dòng của bốn thẻ vào một lệnh: `split_far` tách theo cụm xếp chồng; 63 lệnh gộp của 65 poster
  đều khe <= 0,6 nên không lệnh nào bị đụng.
- CÒN: chữ neon quầng rộng vẫn để bóng mờ (accept2 h03); mẩu vụn nhỏ sót quanh tiêu đề cong (mix1 f02); phép đo "còn nét" bằng
  Hi-SAM chạy lại cho số > 100%, chưa dùng được. Dựng lại tại máy mới xem 30 poster accept2 với luật quầng đầu, chưa xem đủ 65 với
  luật cuối (người dùng chọn chạy thẳng máy chủ). CHỜ MÁY CHỦ: một lượt accept2, so mốc 21 đẹp / 7 dùng được / 2 hỏng; VRAM khi
  Hi-SAM + LaMa chung card với FLUX.

**accept2 trên main 3ac2285 (14/10, zip 1014_040431 -> `output/accept2h/`):** đủ 30; 18,5 s / poster không tính LLM. Xem 30
poster (nháp | lượt trước | nét chữ + mặt nạ | poster): gỡ chữ sạch ở mọi nền, con dấu cà phê / huy hiệu giữ nguyên; VLM không viết
icon; cụm nét lạ: VLM giữ cả 6 (chữ vòng trên huy hiệu trà, dấu trên máy lọc). LỖI DO CHẶNG A (đã vá, đo tại máy):
- q03 MẤT LOGO Coca-Cola: dòng OCR "eo.mg" (khung 158 x 340 trùm logo) VLM không nhắc, `fill_gaps` xoá trống. [mã] poster có ảnh
  sản phẩm: dòng không ai viết -> giữ (`fill_gaps(product=)`).
- q02 lá trước dòng bị gặm nửa (khung OCR một dòng trùm hai hàng, "cỡ" 127 px cho chữ 20 px -> dải quầng 44 px). [mã] quầng phải
  dính vào nét; cỡ dòng <= 2,2 x phân vị 90 cao mảng nét; dấu câu chỉ gỡ giữa nét đầu và nét cuối; `slots._trim_lead` dời mép trái
  khung tới nét đầu khi khung ôm icon (22 / 538 dòng của 65 nháp), `render._align_left` theo mép các dòng đó.
- h04 "NGÀY • HỘI": `_lead_dots` nhận mẩu nét trước "HOI" là chấm. [mã] cần thêm chữ OCR mở đầu bằng ký tự không phải chữ / số.
- e04 bóng chữ 3D lúc hết lúc không: nền giấy lệch trung vị 4-8 ΔE tuỳ nén JPG. [mã] `HALO_FLAT` 4 -> 8 (65 nháp: mặt nạ 14,9 ->
  15,4% ảnh); neon h03 cũng sạch hơn.
- Không do chặng A: l04 VLM trả rỗng hai lần (mã lấp 8 dòng, poster hỏng như lượt trước); e05 VLM lặp "Gội dưỡng", thiếu "Đính đá";
  r01 "15%" quá nhỏ; k02 câu "HÁI TAY 100%" không có chỗ (chữ vòng trên huy hiệu, VLM giữ nguyên).

**14/10 tối: CHẶNG B -- VLM GHÉP CÂU, CODE CHÉP CHỮ VÀ DỰNG THEO NHÁP** (plan duyệt; đường VLM viết html: nhánh `legacy-ops-html`).
Người dùng chốt: nhóm lấy từ VAI câu khách (brief), không đoán từ ảnh; không cắt chữ, co; nháp thiếu câu thì chịu; các bước theo
thứ tự khách; VLM không cần chữ OCR làm gợi ý.
- [mã] `textfix/fill.py`: `parse` (khuôn `assign` / `styles` / `accent` / `missing`), `resolve` (một dòng một đơn vị; chữ in trên sản
  phẩm luôn giữ; `split` phải nối lại đúng câu; một câu ở nhiều chỗ -> chỗ giống nhất nhận, chỗ khác nhận câu cùng vai còn thiếu
  hoặc thành dòng lấp; dòng lấp có số bịa -> bỏ chữ; VLM chỉ được xoá cụm nét lạ; dòng chưa ai ghép như luật lấp cũ), `order`
  (bước), `layout` (khung nét Hi-SAM, cỡ trung vị, lề theo mép nét, ngắt dòng theo tỉ lệ ký tự OCR, dòng khác cỡ / mẩu cùng hàng,
  một dòng nhiều câu cắt tại khe nét gần chỗ dự kiến theo tỉ lệ chữ, tập cùng vai, kiểu theo vai, màu theo vai trừ khi các dòng
  cùng vai đo ra màu khác hẳn, số bước theo thứ tự khách, ngoặc cho câu trích). `render.render_units` / `check_units` /
  `autofix_units`. Lời dặn VLM 15.400 -> 6.900 ký tự (kể cả danh mục font). Bỏ `_guard`, `validate`, `marks_rule`, `split_far`,
  `fix_ops`, nhóm / cân cỡ / căn lề của render, `_pack_lists`, `_list_order`.
- [đo tại máy, lệnh VLM cũ qua `fill.from_ops`, nền LaMa CPU] mix1c 35 poster: 24 tập ở 19 poster, 0 tập lệch cỡ, 0 chữ lệch câu
  khách, 8 phần tử co < 0,6 (câu dài trong chỗ FLUX vẽ ngắn), 14 câu thiếu -- cả 14 đã thiếu trong lệnh cũ. accept2 (bộ thi, chỉ
  ghi): 17 tập ở 12 poster, 0 lệch, 0 chữ lệch câu, 10 phần tử co < 0,6.
- LƯU Ý: nhiều chi tiết chặng B được phát hiện trên ca accept2 (tên + giá gộp dòng, huy chương, màu chữ viền, ngoặc ruy băng) --
  accept2 không còn là bài thi sạch cho chặng B; chấm chặng B trên `bench/final1.json`.
- CÒN (nguồn lỗi đã dồn về bước ĐO DÒNG): OCR gộp hai cột / một khung hai hàng / đọc tick thành chữ cái (tick sát chữ bị gỡ cùng
  chữ: mix1 m01, có từ khi dùng Hi-SAM; thử luật theo ký tự OCR đầu dòng: bắt 0 / 538 dòng, đã gỡ) / góc nghiêng. Ứng viên: Hi-SAM
  phân cấp (từ / dòng) chia dòng thay OCR. VLM ghép theo khuôn mới chưa đo (cần máy chủ).

**Bộ thi `bench/accept2.json` (30 ca, seed 9001-9030):** e nút / thẻ / huy hiệu (6), h nền khó (5), k chữ khó (5), l danh sách /
bảng / các bước (5), q ảnh sản phẩm (3), r form thường gặp (6). CHỈ chạy và chấm, không sửa mã theo từng ca của bộ này; lỗi
thấy ở đây chỉ sửa khi nó hiện thành một con số trên tập luyện (mix1, accept1, dev2, dev3, form1).
**Phía chữ mới, đếm trên mix1 lượt hai (34 poster, 286 dòng):** bộ kiểm còn 9 too_small + 1 overflow; 8 poster có lệnh co < 0,7,
gần hết vì VLM đặt câu DÀI HƠN chữ nháp vào ô nhỏ (nháp 7 ký tự "Glow Up" -> "Serum Vitamin C Glow Up 30ml"). Chữ cong còn hỏng
(g04, c01) là do bước ĐO bỏ sót / đo sai dòng cong, không phải bước gỡ: 76 poster chỉ có 12 mảng "chữ bỏ sót" ứng viên, luật
theo vùng chữ của bộ dò không cứu được g04 (độ phủ 0,2-0,44).

**Không để trống dòng (13/10 tối, người dùng: "cố gắng bịa vào, đừng bỏ trống"):** đọc lệnh của 34 poster mix1: 13 / 286 dòng
bị bỏ, ở 5 poster -- và phần lớn KHÔNG phải VLM bỏ: 4 poster do luật "lặp chữ -> skip" của mã (i05 huy hiệu "Ưu đãi" nháp vẽ hai
lần, VLM viết cả hai, mã xoá một), 1 poster do VLM trả lời cụt (p02: 1 lệnh cho 10 dòng, autofix đổi 9 dòng còn lại thành skip,
kể cả dòng chữ in trên gói sản phẩm). t02 / t03 không bỏ dòng nào: lỗi là VLM đặt sai nội dung.
- [mã] chế độ chỉ dòng chữ: luật "lặp chữ" không bỏ nữa (`M["lonly"]`); `engine.design` gọi lại VLM một lần khi lệnh phủ < 60%
  số dòng (`lonly.COVER_MIN`); `lonly.fill_gaps` sau `validate`: dòng chưa có lệnh viết / giữ -> chữ in trên sản phẩm thì GIỮ;
  chữ nháp gần trùng một câu khách thì viết câu đó; không thì GIỮ chữ nháp (vỏ không trống). Thử trên lệnh thật: p02 1 -> 10 / 10
  dòng (5 dòng viết câu khách, 4 dòng giữ chữ nháp trong đó 1 dòng sản phẩm), i05 10 -> 11, m03 12 -> 13.
- [tập thử] mix1 f01 thiếu ô bắt buộc `discount` (HTTP 422, không chạy) -> thêm "Giảm 20%".

**Zip báo cáo (13/10 tối, người dùng: tải ít zip nhất, đủ thông tin):** `batch_report.py` gói MỘT tệp
`output/runs_zip/<tập>_report_<giờ>.zip` = mỗi ca nháp + bản xoá + poster (JPG chất lượng 92, tự hạ nếu vượt 23 MB) + plan /
brief / request + manifest; trang HTML / tờ ảnh ghép không gói (xem trên máy chủ). 42 poster accept1: 20,9 MB.
`lonly_replay.py` đọc được .jpg. JPG 92 so với PNG trên 10 poster: 9 / 10 poster giữ nguyên cách xử lý từng dòng, nền lệch trung
bình 1,06 mức xám -- đủ để thử sửa bước gỡ chữ tại máy, số ngưỡng sát biên thì xác nhận lại trên máy chủ.

**Lượt kế (13/10 tối): tập mới `mix1`, chỉ chế độ chỉ dòng chữ, xem mắt:**
```
source ~/textfix_env.sh; export TEXTFIX_GEO=on TEXTFIX_LONLY=on
bash scripts/start_server.sh
python scripts/dev_batch.py bench/mix1.json
```
Xem: bấm đúp các tệp `output/runs/_reports/mix1/sheets/mix1_01.jpg` ... trong JupyterLab. Muốn Claude chấm: tải MỘT tệp
`output/runs_zip/mix1_report_<giờ>.zip` về gốc repo.

**So đường cũ với "chỉ dòng chữ" trên cùng seed (hai lượt, chạy từ terminal):**
```
source ~/textfix_env.sh; export TEXTFIX_GEO=on
bash scripts/start_server.sh                                   # lượt 1: cờ tắt (đường cũ + các sửa 13/10)
python scripts/dev_batch.py bench/accept1.json --fresh
export TEXTFIX_LONLY=on; bash scripts/start_server.sh           # lượt 2: khởi động lại máy chủ với cờ bật
python scripts/dev_batch.py bench/accept1.json --fresh         # bảng in ra so từng ca với lượt 1
```

**Sau cụm sửa 13/10 chiều:** chạy lại accept1 với `--fresh` (cùng seed) để so trước / sau trên đúng 42 nháp:
`export TEXTFIX_GEO=on`, khởi động lại máy chủ + kernel, `%run scripts/dev_batch.py bench/accept1.json --fresh`.

**Bộ nghiệm thu toàn hệ thống: `bench/accept1.json` (42 ca mới, seed cố định, 1 ảnh mỗi ca).** Chạy với chữ cong BẬT:
`export TEXTFIX_GEO=on`, khởi động lại máy chủ và kernel, rồi `%run scripts/dev_batch.py bench/accept1.json`.
Nhóm: a (5) đề bài dài -- FIT, khối, danh sách; b (6) danh sách cùng kiểu, dấu đầu dòng, icon to; c (4) hai cột, bảng giá, hai
thẻ; g (12) chữ khó -- vòm, con dấu, nụ cười, nghiêng, xếp chồng, nhiều cỡ, hai giới hạn đã biết; s (3) vỏ; f (8) form như người
dùng thật; p (2) ảnh sản phẩm; i (2) ảnh thường / đề bài mơ hồ. Mỗi ca có "why" = điều cần xem. Bộ này thay cho các mục 0a, 0, 1
bên dưới.

0a. Đề bài dài sau sửa 13/10 (không cần GEO): chạy lại đúng ba đề nồi cơm / studio chụp ảnh / resort trên giao diện, tải
   "Tải debug (zip)" cả ba một lần. Xem: lệnh chữ nhiều ô có cùng một khối không, dòng có "mark" có dấu ở đầu chữ không, lệnh
   nào còn "co còn" dưới 0.7.
0. Chữ cong sau sửa 13/10: `export TEXTFIX_GEO=on`, chạy lại đúng hai đề bài con dấu tiệm bánh và donut trên giao diện (tải
   "Tải debug (zip)"), rồi `%run scripts/dev_batch.py bench/dev3.json --only g01,g02,g03,g04,g05,g06,g07,g08,g09,g10,g11,g12,g13,g14,g15,g16 --fresh`.
   Xem trong plan.json: log "chữ arc: ... {cx, cy, r ...}", không còn "vỏ ... chứa chữ được viết" trên ô cong, và có / không dòng
   "lệnh gộp ô cong".
1. `%run scripts/dev_batch.py bench/dev3.json` -- tập dev 3 (thay cho dev3_geo: dev3 đã gồm cả 16 ca hình học chữ).
2. Để so trước / sau dev2 (prompt LLM / VLM mới, B2 hai lượt, cỡ / khung mới): xoá `output/dev/dev2_manifest.json` rồi
   `%run scripts/dev_batch.py` -- chạy lại cả 40 ca (seed cố định; d33-d40 là ca mới).
