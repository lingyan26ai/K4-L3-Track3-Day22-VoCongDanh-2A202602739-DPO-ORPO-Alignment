# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** _<Họ Tên>_
**Khoá:** _<A20-K4 / ...>_
**Tier đã chạy:** T4
**Ngày:** _<YYYY-MM-DD>_

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

**Tiến độ:** đã chạy NB0–NB3. Các số liệu bên dưới được chép từ output Colab trong ảnh đã lưu tại
`submission/screenshots/`. Phiên Colab đã mất các file trước khi chạy được NB4, nên chưa có kết quả
so sánh SFT với SFT+DPO. Ảnh NB3 đã lưu gồm `03-dpo-training.png`, `03-dpo-reward-curves.png`
và `03-dpo-metrics.png`. Các số DPO trong bảng được làm tròn đến bốn chữ số thập phân từ output.
File `adapters/dpo/dpo_metrics.json`, trọng số và dấu vân tay tập dữ liệu chưa được tải về repo.
Notebook Colab trong repo mới có code NB0 đã điền, chưa chứa output của phiên Colab đã chạy.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4 · log báo bộ nhớ tối đa 14.563 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned (mặc định trong repo, cần đối chiếu cấu hình phiên Colab) · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (xác nhận trong output NB3) · 800 train / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,9% theo số token |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1 · loss sigmoid · 100 bước |
| Giám khảo | _<rm:tên-mô-hình hoặc nhà-cung-cấp:tên-mô-hình; sanity accuracy>_ |
| Chi phí | _<0 đồng (Colab miễn phí) / ...>_ |

**NB0:** `my_dpo_loss` khớp công thức tham chiếu với loss 0,6981. Khi policy bằng reference,
loss bằng 0,6931 và reward của cả chosen lẫn rejected đều bằng 0.

**NB1:** huấn luyện hoàn tất 125 bước trong 10 phút 57 giây. Loss ở bước 10 là 1,884150,
ở bước 120 là 1,283640. Đường loss giảm nhìn chung, có dao động giữa các bước.
Dòng `Final SFT loss: 1.3602` là loss trung bình huấn luyện do `result.training_loss` trả về,
không phải loss riêng của bước cuối. Log xác nhận lưu adapter tại `adapters/sft-mini/`
và mô hình gộp 16-bit tại `models/sft-merged/` trên Colab. Các trọng số chưa được tải về repo.

**NB2:** kiểm tra không trùng câu hỏi đã qua (`no prompt overlap`). Dữ liệu 800 train / 100 eval
đã lưu tại `data/pref/` trên Colab. Trung vị độ dài chosen là 94 token, rejected là 86 token.
Tỉ lệ chosen dài hơn là 65,9%, tiêu đề biểu đồ làm tròn thành 66%.
Các file Parquet và `stats.json` chưa được tải về repo.

Đã đọc ba cặp mẫu, lưu nội dung tại `submission/preference-samples.txt`:

| Cặp | Nhận xét |
|---|---|
| 1 — Tạo 10 yêu cầu thay đổi | Chosen đánh số đủ 1–10 và dùng thống nhất Trước / Yêu cầu / Sau. Rejected cũng có 10 mục nhưng thiếu số thứ tự 8, 9 và dùng nhãn không thống nhất. Có cơ sở ưu tiên chosen về trình bày. |
| 2 — Phân loại bài đăng | Cả hai không dùng nhãn được yêu cầu là hung hăng / không hung hăng. Chosen trả lời Thô bạo, rejected trả lời Bạo lực. Chưa thấy cơ sở rõ để ưu tiên chosen. |
| 3 — Đặt lịch đánh giá giọng nói | Cả hai thêm chi tiết chưa có trong đề và khẳng định đã đặt lịch thành công dù chỉ đang hướng dẫn. Rejected còn thêm URL và AVAR không được cung cấp. Chosen vẫn có lỗi dù thêm ít chi tiết hơn. |

Ba cặp này cho thấy nhãn sở thích có thể có nhiễu. Chưa thể kết luận chosen luôn tốt hơn,
cũng chưa thể suy ra mức nhiễu của toàn bộ tập dữ liệu từ ba mẫu.

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 29 phút 46 giây theo thanh tiến trình train, chưa tính precompute reference và đánh giá cuối riêng |
| VRAM cao nhất | _<...>_ |
| Loss trung bình huấn luyện | 0,6745 |
| Loss đầu tiên được ghi | 0,6919 |
| Reward chosen / rejected cuối trên train | 0,4036 / 0,3076 |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0,0960 |
| Reward chosen / rejected trên held-out | 0,4172 / 0,3292 |
| Độ chính xác reward trên held-out | 67% |
| Margin trên held-out | 0,0880 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | _<... → ... ký tự>_ |

NB3 dùng mô hình tham chiếu `models/sft-merged` và tính sẵn reference log-prob. Ảnh output ghi
9 phút 01 giây cho precompute train và 1 phút 10 giây cho precompute eval. Mức bộ nhớ 4,69 GB
ở cell dọn GPU sau NB3 không phải VRAM cao nhất, nên chưa điền vào hàng đó.
Độ chính xác reward 67% là tỉ lệ chosen có reward cao hơn rejected, chưa phải win rate ở NB4.

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

_Mô tả riêng `rewards/chosen` và `rewards/rejected` trên **train và held-out**. Chosen tăng hay giảm?
Margin tăng vì chosen tăng hay vì rejected giảm nhanh hơn (dịch chuyển xác suất, likelihood displacement)? Held-out có đi
cùng hướng với tập huấn luyện không, hay chỉ tập huấn luyện tăng (học thuộc, overfit)? Chẩn đoán tự động có khớp với điều bạn
thấy không?_

Reward của chosen và rejected đều tăng trên cả train lẫn held-out. Cuối train, chosen đạt 0,4036
và rejected đạt 0,3076, nên margin là 0,0960. Trên held-out, chosen đạt 0,4172 và rejected đạt
0,3292, cho margin 0,0880. Như vậy margin tăng vì chosen tăng nhanh hơn rejected, không phải
vì rejected giảm. Đây không phải likelihood displacement, vì reward chosen vẫn dương.

Đường train có dao động, còn margin held-out tăng qua các lần đánh giá. Hai tập đi cùng hướng
và margin cuối khá gần nhau, nên chưa thấy dấu hiệu rõ là mô hình chỉ học thuộc train. Tuy vậy,
độ chính xác reward held-out tăng từ 59% lên 71%, rồi giảm về 67% ở cuối. Việc margin tăng không
đồng nghĩa độ chính xác tăng đều. Cũng chưa thể dùng 67% này để kết luận DPO trả lời tốt hơn SFT.

Chẩn đoán tự động là INTENDED, phù hợp với quy tắc trong code là chosen dương và margin dương.
Tuy nhiên, biểu đồ không hoàn toàn giống mẫu lý tưởng trong hướng dẫn là chosen tăng và rejected
giảm. Cả hai đều tăng, chỉ khác tốc độ. Dòng chẩn đoán dùng trung bình ba bản ghi cuối nên báo
margin khoảng 0,086, còn margin của lần đánh giá cuối là 0,0880. Với dữ liệu có 65,9% chosen dài
hơn rejected và ba cặp mẫu có nhiễu, vẫn cần NB4 để kiểm tra chất lượng câu trả lời và thiên vị độ dài.
NB4 chưa chạy được do phiên Colab mất file và hết quyền sử dụng GPU.

**Giải thích từ NB0:** DPO tối ưu chênh lệch reward của chosen
và rejected. Trong ví dụ A, reward chosen là +1 và rejected là −1. Trong ví dụ B, chosen là −3
và rejected là −5. Cả hai đều có margin 2 và loss 0,127 khi β = 1. Vì vậy xác suất chosen vẫn
có thể giảm nếu rejected giảm nhanh hơn. RPO thêm NLL của chosen nên phạt ví dụ B nhiều hơn:
loss RPO ở A là 2,027, ở B là 2,427.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | | | | | | | |
| hữu ích — helpfulness (4) | | | | | | | |
| an toàn — safety (4) | | | | | | | |

Giám khảo: ______ · sanity accuracy: ______ · `score_length_spearman` (reward model) hoặc độ nhất quán khi đổi chỗ A/B — position consistency (giám khảo API): ______

_Khoảng tin cậy có chứa 0.5 không? Giám khảo có đáng tin trên tiếng Việt không (xem bộ cặp kiểm tra sanity)? DPO thắng vì câu trả lời tốt
hơn hay vì dài hơn? Hai reward model trong hội đồng (`per_judge`) có cho win rate gần nhau không? Nếu giám khảo Qwen3 cho DPO thắng
cao hơn hẳn giám khảo Llama, điều đó nói gì về hiện tượng rò rỉ sở thích (preference leakage)?
Chọn 2 ví dụ cụ thể (1 câu về độ hữu ích, 1 câu về an toàn) và giải thích._

_Trả lời ở đây._

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

_Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

_Trả lời ở đây._

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
