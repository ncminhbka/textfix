# Kế hoạch: từ "điền ô OCR" sang "dựng lại theo khối cho giống nháp"

Soạn 11/10 sau khi xem lại toàn bộ dev2 + dev3 (120 poster). Trạng thái: **thiết kế, chưa làm**. Mới thử một ảnh (mục 5).
Tài liệu liên quan: docs/DESIGN.md (cách làm hiện tại), docs/TUNING.md (nhật ký tune, quy tắc tune).

## 1. Ý tưởng gốc (giữ nguyên)

Đã thử FLUX chỉ vẽ nền + VLM viết HTML: thất bại vì VLM kém toạ độ và nền không chừa chỗ cho bố cục.
Ý tưởng hiện tại: bảo FLUX vẽ **cả poster kèm chữ** (chữ sẽ sai chính tả), rồi một vòng **xoá chữ**. Ta có:

- **nháp**: một bản thiết kế tham khảo đẹp (bố cục, cỡ, màu, kiểu chữ, vỏ);
- **bản xoá**: nền sạch để dựng chữ đúng lên.

Mục tiêu: **từ nháp ra poster đẹp như nháp, đúng chữ của khách.**

Lớp phủ = nháp − bản xoá: chữ + thẻ / viên thuốc / dải / huy hiệu / icon / sao. Phần còn lại trong bản xoá là ảnh (vật + nền).
Không dùng model dò vật thể ở bất kỳ đâu, và sẽ không dùng (vật thể có muôn kiểu, không tổng quát).

## 2. Cách làm hiện tại

B0 LLM brief -> B1 FLUX nháp -> B2 FLUX xoá + dọn chữ sót -> B2.5 OCR nháp thành các ô (L dòng chữ, S vỏ, I chi tiết) ->
B3 VLM điền mỗi ô một lệnh -> B4 Chromium dựng, đo lỗi, sửa bằng code.

Điểm mấu chốt: **mỗi dòng OCR là một cái khuôn** (khung, cỡ, vị trí), VLM phải điền vào đúng khuôn đó, chữ chỉ được co cho vừa.

## 3. Điểm đau (dev2 + dev3, 120 poster)

**Nháp gần như lúc nào cũng đẹp hơn poster.** Khoảng 1/3 poster đạt (bố cục đơn giản, vài khối chữ thẳng). Phần còn lại thua nháp.

Số đo tự động (`scripts/mold_stats.py`):

| Dấu hiệu | Poster |
|---|---|
| có khối chữ bị co < 0.7 cỡ nháp | **73 / 120** |
| thiếu câu khách | 21 |
| chữ lớn OCR không đọc được (cong, chéo) | 11 |
| nhóm dòng lệch cỡ / lệch lề (bộ dò nhóm chặt, đếm thiếu; bằng mắt ~12) | 4 / 2 |
| dính ít nhất một dấu hiệu | **84 / 120** |

Năm kiểu thua nháp (bằng mắt):

| Kiểu | Ví dụ (output/dev2, output/dev3) | Gốc |
|---|---|---|
| chữ nhỏ, nhạt hơn nháp, thừa khoảng trống (~40-70) | d05 v0, d09 v1, d12, d13, d17, d19, d20, d26, g08 v0, nhóm p | mỗi dòng ép vào khung OCR riêng, chỉ được co |
| chữ cong / chéo vỡ (~12) | g01 v0 (15 ô mỗi ô một chữ cái), g02, g03, g05 v0, g07 v0 (mất "FLASH SALE"), g12 | OCR không đọc -> không có khuôn |
| danh sách / nhóm lặp lệch cỡ, lệch lề, thiếu mục, icon trơ (~12) | d05, d19, d20, f06 (4 thẻ 4 cỡ), p08, p10 (lề so le chép từ nháp) | số hàng / vị trí từng dòng của nháp là nhiễu |
| vỏ và chữ không khớp (~10) | d13 v0, d32 v0, g11 v0, g12 v0, p04 v0 (viên thuốc chia 2 cột) | chữ theo khung dòng, không theo lòng vỏ |
| mất cá tính chữ (~8) | g03 v0, g04 v0, d28 v1, d02 v1 | giới hạn bộ font / CSS -- **giải pháp này không chữa** |

**Chẩn đoán: bám nháp sai tầng.**

| Tầng | Trong nháp là gì | Hiện tại |
|---|---|---|
| DÒNG: khung, cỡ, mép trái từng dòng, số hàng, chỗ ngắt dòng | **nhiễu** của FLUX | bám rất chặt |
| KHỐI: vùng chữ chiếm, độ lớn, vỏ chứa nó, vai, hướng nghiêng / cong, thang cỡ, màu | **thiết kế** | bỏ mất |

Nguyên tắc mới: **nháp là gợi ý ở tầng dòng, là chuẩn ở tầng khối.** OCR xuống vai gợi ý.

Đã loại: VLM agent nhiều vòng tự chỉnh. Vòng duyệt VLM cũ (một vòng) tốn ~20 s, đẹp lên ~2/15, xấu đi ~5/15 -- VLM thấy lỗi nhưng
sửa bằng cách đoán số. Thêm vòng không chữa điểm yếu toạ độ, chỉ thêm ~10 s mỗi vòng.

## 4. Giải pháp: dựng lại theo khối cho giống nháp

Coi nháp là **ảnh đích**. VLM quyết nội dung, không xuất con số nào. Code tìm cỡ và cách dàn bằng dựng thử rồi so với nháp.

**Tầng 1 -- Nhìn (code).** Gom lớp phủ thành **khối**. Mỗi khối: đường bao + hướng (thẳng / nghiêng / cong); khối lượng chữ
(diện tích nét, cao chữ, độ đậm, số dòng); vỏ chứa nó; quan hệ **nhóm lặp** (thẻ / hàng giống nhau xếp đều). OCR chỉ gợi ý khối nói
gì và nhận chữ in trên sản phẩm.

**Tầng 2 -- Quyết (VLM, một lần gọi).** Xem nháp có dán nhãn khối A, B, C. Trả về theo khối: khối là gì; nhận câu khách nào; cấu
trúc nội dung (tiêu đề, danh sách N mục, giá + đơn vị, chỗ nhấn) dưới dạng HTML tự dàn; dáng chữ (nhóm font, đậm, nghiêng, màu).
Nhóm lặp: một mẫu chung cho mọi phần tử. Câu chưa có khối: chọn một trong các chỗ trống code đề xuất (có nhãn). Khối không ứng câu
khách nào: bỏ. Được gộp / tách khối khi code gom sai.

**Tầng 3 -- Khớp (code, vòng dựng-so trong Chromium, vài ms mỗi vòng).** Tìm font trong bộ có sẵn, cỡ, chỗ ngắt dòng, giãn chữ để
bản dựng **lấp đường bao và đạt khối lượng như nháp** -- chữ được **nới lẫn co**. Ràng buộc cứng: trong lòng vỏ trừ lề; cả nhóm lặp
một cỡ một lề; có cỡ sàn. Mặc định chữ ở yên trong đường bao khối. Không khớp được: dàn lại -> nới sang vùng nền **yên** kề bên
(bản xoá phẳng, ít cạnh, cùng tông nền dưới khối, không thuộc khối khác -- phép đo điểm ảnh, không có khái niệm vật thể) -> báo
thiếu. Sai theo hướng thận trọng: thà báo thiếu còn hơn đặt bừa.

**Tầng 4 -- Kiểm (code).** Giữ các phép đo hiện có + điểm giống nháp theo khối. Lỗi nặng còn lại mới gọi VLM **chọn** giữa các phương
án đã dựng sẵn (không tự chỉnh số).

**Vì sao tổng quát:** cong / chéo / to nhỏ trong một dòng chỉ là hình dạng đường bao và cấu trúc, không cần luật riêng; nháp vẽ
thiếu hàng hay OCR sai không còn ảnh hưởng (số mục từ câu khách, hình học từ nét mực); lỗi chữ nhỏ được nhắm trực tiếp. Độ trễ gần
như giữ nguyên (vẫn một lần gọi VLM).

**Giữ lại từ hệ thống hiện tại:** B0, B1, B2 + dọn chữ sót, dò vỏ (`overlay.py`, gồm vỏ nhạt / vỏ viền), dò đường chân chữ cong
(`geo.py`, đang tắt `TEXTFIX_GEO`), renderer (một lệnh đã phủ được nhiều ô, `kind: group`), luật chặn chữ (`slots._guard`,
`uncovered`), phép kiểm đo, sửa bằng code.

## 5. Bằng chứng đã có (ít)

Một ảnh, HTML do Claude viết tay thay VLM: dev3 p08 v0 (`output/dev3/run_20261009_094914_1e4c/v0`), so sánh ở
`output/proto/block_p08.jpg`. Gộp 5 dòng + 5 icon danh sách thành một khối, một đoạn HTML cột, không toạ độ; renderer hiện tại dựng
được ngay.

| | Hiện tại | Theo khối |
|---|---|---|
| danh sách | 5 hàng cho 4 mục, một icon trơ, chữ nhỏ | 4 hàng đều, đúng cỡ nháp, không co |
| tiêu đề | mất "tiền đẻ ra tiền" | đủ câu nhưng co 0.66 (nháp chừa 2 dòng, câu cần 3 -> cần nới khối) |
| câu thiếu | 1 | 0 |

## 6. Rủi ro

- Gom khối sai thì hỏng cả khối (vì thế VLM được gộp / tách).
- HTML do VLM (Qwen 27B) viết trong khối có ổn định không: chưa biết, chỉ biết trên server.
- Đo "nền yên" để nới khối sẽ từ chối cả chỗ designer thật vẫn đặt chữ được (chữ trắng trên ảnh tối nhiều chi tiết): chấp nhận.
- Thừa hưởng lỗi của bước xoá: xoá nhầm chữ sản phẩm, xoá sót, giữ lại vỏ, vẽ thêm (g08 v1).
- Cá tính chữ (3D, viết tay đậm, kim loại) vẫn giới hạn bởi bộ font: việc riêng.

## 7. Lộ trình (mỗi bước đo trên 120 nháp đã có, chưa cần FLUX)

| Bước | Việc | Dừng / đi tiếp |
|---|---|---|
| 1 | **Thử trần**: Claude đóng vai VLM trên ~15 nháp trải 5 kiểu lỗi (nghiêng về ca chữ co): gán câu vào khối, viết HTML, code dàn | không hơn rõ poster hiện tại -> dừng, nghĩ lại |
| 2 | Bộ gom khối + vòng khớp cho **khối chữ thẳng** (nhắm 73 ca chữ co) | tỉ lệ cỡ dựng / cỡ nháp về ~1.0 |
| 3 | Vỏ + **nhóm lặp** (danh sách, các bước, viên thuốc) | đủ mục, một cỡ, một lề |
| 4 | Khối nghiêng, cong (dùng `geo.py`, dựng theo đường chân) | g01-g07, g12 |
| 5 | Câu không có khối, nới khối sang nền yên | số câu thiếu |
| 6 | Prompt VLM theo khối, chạy server A/B cùng seed với hệ thống hiện tại | thắng bằng mắt |

Thước đo: số câu thiếu; tỉ lệ cỡ dựng / cỡ nháp; độ đều trong nhóm; tỉ lệ thắng bằng mắt so poster hiện tại (xem cặp nháp | poster).
Làm sau cờ tắt (như `TEXTFIX_GEO`) cho tới khi cả đường đi xong và đo được -- theo quy tắc tune ở docs/TUNING.md.

## 8. Việc còn treo (cần người dùng quyết)

1. VLM có được viết chữ không có trong câu khách? ("Khai Trương" ở d16 trông ổn; "WEAR CHIC" ở f09 là bịa.) Đề xuất: cấm bằng code.
2. VLM có được rút gọn câu khách? (tiêu đề mất "sau", "GIAO".) Đề xuất: tiêu đề / câu <= 12 chữ không được bỏ chữ.

Trong thiết kế theo khối, cả hai mặc định là **không** (khối không ứng câu khách thì bỏ; nội dung khối là nguyên câu khách).

## 9. Trạng thái repo khi soạn (11/10)

- `main` tới `2e061e0` + commit tài liệu này. Server đang chạy mã cũ hơn (trước `c8bd4a9`); lượt dev3 trên server có 61/62 ảnh.
- Dữ liệu local: `output/dev2/` (62 ảnh), `output/dev3/` (61 ảnh), mỗi ảnh có draft.png, plate.png, plate_raw.jpg, poster.jpg,
  plan.json; manifest `<tập>_manifest.json` ghép ca -> lượt chạy. Bộ đáp án ô `bench/slots_gt`, bộ thử hình học `scripts/geo_synth.py`.
- Công cụ: `scripts/replay_plans.py <thư mục ra> <dev2|dev3> [ca,ca]` dựng lại plan server bằng mã hiện tại (lỗi trước / sau);
  `scripts/mold_stats.py <thư mục ra>` đo các dấu hiệu "OCR làm khuôn" (~15 phút).
- Chưa kiểm trên server: các sửa B3 / B4 của lượt dev3, prompt VLM siết ở lượt dev2, dọn chữ sót lượt 2 bằng FLUX.
