# Lab 21 — Evaluation Report

**Họ tên**: Đỗ Trung Tuyến  **MSSV**: 2A202602427  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Colab free T4 16GB (dùng cho NB2; sau đó hết quota GPU)

> **Tuyên bố trung thực về phạm vi.** Bài nộp này là **bài nộp một phần**. NB1 (dữ liệu + mask) và NB2 (đóng băng eval + baseline a/b) đã chạy đầy đủ, hợp lệ (`eval_limit = null`, `smoke_mode = false`). NB3 (huấn luyện `correct`), NB4 (ba run đối chứng) và NB5 (phán quyết, autopsy, định tính) **chưa chạy được** vì Colab hết quota GPU. Các mục phụ thuộc vào chúng (4, 5, 6) được ghi rõ là "chưa đo" và **không có con số nào được bịa ra**. Vì vậy `scripts/verify.py` báo FAIL ở các artifact thiếu (`runs.csv`, `verdict.json`, `autopsy.json`, adapter `correct`) — đây là hệ quả đã biết, không phải bị bỏ sót. Các kiểm tra còn lại đều đạt: 119 unit test, mask proof, đủ 50 mẫu target, prompt (b) không bị sửa, (b) hơn (a) (0.000 → 0.765) và bộ eval không đổi.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage (intent, urgency, product, sentiment), bộ mặc định của lab |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 (cấu hình của tier T4) — p95 đo được là 98 token, max 101, `suggested_max_length` = 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` (mask dựng bằng labkit theo offset ký tự) |
| Epochs / max_steps | Chưa huấn luyện nên chưa có số liệu; dùng mặc định của tier T4 nếu chạy lại |

**Lý do chọn.** Tôi giữ bộ mặc định của lab (`unsloth/Qwen3.5-4B`, tier T4, 250 ticket CSKH tiếng Việt → JSON triage) vì ba lý do: (1) 4B với LoRA r=16 vừa với T4 16GB; (2) bài toán JSON bốn khóa có nhãn rõ nên chấm tự động được (target, format) mà không cần người chấm; (3) dùng đúng bộ eval của lab (50 mẫu target + 15 mẫu regression) giúp kết quả đối chiếu được với `results/` và với baseline đã đóng băng. Tôi không dùng dataset riêng nên không có `data/CUSTOM_DATASET.md`.

**Template có giữ khối `<think>` không?** Có — kết luận của `template_check`: "reasoning preserved — safe to train on traces" *(results/template_check.json)*. Template Qwen3.5 sinh sẵn khối `<think></think>` rỗng trong generation prompt, nên mask phải dựng đúng ranh giới này; điều này đã được kiểm chứng ở mục 2.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.415 (39/94 token) |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Chỉ khoảng 41% số token của mỗi mẫu được tính loss: chính là phần JSON trả lời của assistant. Phần system, câu hỏi (ticket) và khung template bị mask. Để đối chứng, chế độ `everything` supervise 94/94 token (100%), tức là mô hình sẽ bị phạt cả khi "học thuộc" câu hỏi — đó là lỗi mà mask proof được thiết kế để chặn. Đoạn token được supervise cụ thể xem trong `results/mask_proof.json`.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3189.6 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1048.3 |
| (c) LoRA fine-tune | chưa chạy | chưa chạy | chưa chạy | chưa chạy |

Tập đánh giá: 50 mẫu target, 15 mẫu regression, đã đóng băng (SHA prompt tối ưu `719e74d3b6232053`) *(results/baselines_frozen.json)*.

**(b) có thật sự mạnh hơn (a) không?** Có, và rất rõ. Với prompt ngây thơ (a), mô hình gần như không bao giờ trả JSON đúng schema: target 0.000 và format 0.000. Chỉ bằng cách đưa schema và chỉ dẫn rõ ràng vào prompt (b), target tăng lên 0.765, format lên 1.000, và độ trễ giảm từ ~3190 ms xuống ~1048 ms (khoảng 3 lần, vì mô hình không còn lan man). Điểm regression của (a) và (b) bằng nhau (0.791), cho thấy prompt không làm đổi kiến thức chung. Tôi **không sửa** `OPTIMIZED_PROMPT`; baseline (b) là mốc mà bản fine-tune phải vượt, và việc (b) đã đạt 0.765 nghĩa là khoảng trống để fine-tune cải thiện không lớn (tối đa ~0.235 trên target).

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| `attn_only` | q,v | *(matched)* | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| `wrong_lr` | text-linear | 16 | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| `qlora` | text-linear | 16 | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |

Toàn bộ mục này **chưa có dữ liệu chất lượng** do không chạy được NB3/NB4/NB5. NB4 đã được thử chạy nhưng dừng ngay ở run đầu tiên (`attn_only`) với lỗi `'functools.partial' object has no attribute '__func__'` trong `SFTTrainer` (runtime không có GPU, chạy ở chế độ CPU fp32), nên không có `runs.csv`. Trước khi lỗi, NB4 đã tính được cấu hình ghép tham số: `text-linear` r=16 có 32,464,896 tham số huấn luyện, còn `attn_only` (q,v) với rank ghép r=283 có 32,456,704 — chênh dưới 0.03%, nên so sánh vị trí adapter sẽ công bằng. Đó chỉ là phép tính cấu hình, không phải kết quả huấn luyện. Tôi không đưa ra nhận định về kết quả thực nghiệm; phần dưới chỉ nêu giả thuyết và điều tôi sẽ kiểm chứng.

**4.1 — `attn_only` so với `correct`.** Chưa đo. Giả thuyết của tôi: khi số tham số huấn luyện được khớp, vị trí gắn adapter (chỉ q,v so với toàn bộ linear của text) có thể quan trọng hơn rank. Tôi sẽ xếp hạng theo cột target của NB5 chứ không theo train loss, vì dùng loss huấn luyện thay cho chỉ số của tác vụ chính là "Lỗi #3" mà lab cảnh báo.

**4.2 — `wrong_lr`.** Chưa đo. Giả thuyết: chỉ đổi một con số LR có thể cho đường loss trông "khỏe" nhưng chất lượng trên target khác hẳn, nên nếu chỉ nhìn loss mà không biết LR thì dễ kết luận sai rằng hai run tương đương.

**4.3 — `qlora`.** Chưa đo VRAM và chất lượng. Tôi không thể xác nhận hay bác bỏ khuyến nghị "không dùng QLoRA cho dòng model này" khi chưa có số đo.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **chưa xác định** — NB5 chưa chạy, không có `verdict.json`.
`target Δ`, `regression Δ`, `valid_trace_rate`: chưa đo. NB5 đã được thử chạy nhưng dừng ở bước nạp adapter vì `adapters/correct/adapter_config.json` không tồn tại (NB3 chưa tạo được adapter). Log của NB5 vẫn xác nhận mốc `baseline (b) target = 0.765`, khớp với mục 3.

Diễn giải: Không có adapter `correct` nên không thể so (c) với (b) và không thể kết luận fine-tune có hơn prompt tốt hay không. Điều duy nhất NB2 cho phép nói chắc chắn là ngưỡng cần vượt: một bản fine-tune chỉ đáng triển khai nếu target vượt 0.765 mà regression không tụt dưới ~0.791 một cách đáng kể và format giữ được 1.000. Rủi ro đáng chú ý là quên kiến thức chung (catastrophic forgetting): tài liệu của lab ghi nhận một lần chạy model nhỏ hơn (0.8B) có target 0.99 so với (b) 0.495, nhưng regression rơi từ 0.644 xuống 0.067. Với model 4B trên T4 điều này chưa được đo, nên tôi coi cổng hồi quy là phần quan trọng nhất cần kiểm chứng khi có GPU và không dự đoán PASSED hay FAILED.

---

## 6. Định tính — bắt buộc có cả ca THUA

Chưa thể lập bảng định tính vì chưa có đầu ra của bản fine-tune (`qualitative.json` không tồn tại). Khi chạy lại, tôi sẽ so sánh đầu ra (b) và (c) trên 50 mẫu target, chọn ít nhất 2 ca FT thắng và 2 ca FT thua, và tìm mẫu chung ở các ca thua (ví dụ nhãn `urgency` hoặc `sentiment` mơ hồ).

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Dựa trên những gì đã đo, tôi chưa thể khuyến nghị triển khai bản fine-tune nào, vì chưa có bản nào được huấn luyện và đánh giá. Điều tôi chắc chắn là: với bài toán này, một prompt được thiết kế tốt (b) đã đạt target 0.765 với format 1.000 và nhanh gấp ba lần prompt ngây thơ, nên bản fine-tune chỉ có giá trị nếu vượt rõ rệt mốc đó mà không làm hỏng kiến thức chung. Đòn bẩy lớn nhất mà thực nghiệm hiện tại chứng minh được là **chất lượng prompt/schema** (từ 0.000 lên 0.765 mà không cần huấn luyện), kế đến là **mask**, vì mask proof cho thấy chỉ 41% token được tính loss và đó phải đúng là phần trả lời. Còn về vị trí adapter, learning rate hay QLoRA, tôi chưa có số đo nên không khẳng định. Nếu phải quyết định hôm nay, tôi sẽ dùng baseline (b) làm bản triển khai tạm thời, vì nó đã được đo và kiểm chứng, và chỉ chuyển sang fine-tune khi cổng hồi quy được xác nhận PASSED trên cùng bộ eval đã đóng băng.

**Ba điều tôi học được** (cụ thể):
1. Phải đóng băng eval và đo baseline (b) trước khi train: nếu không, khó biết fine-tune thật sự hơn prompt tốt bao nhiêu. Ở đây chỉ riêng việc sửa prompt đã đưa target từ 0.000 lên 0.765.
2. Mask proof không phải thủ tục hình thức: supervise toàn bộ (`everything`) sẽ tính loss trên 94/94 token thay vì 39/94 token là câu trả lời, và Qwen3.5 sinh sẵn khối `<think></think>` rỗng nên ranh giới assistant phải được dựng bằng offset ký tự.
3. Về vận hành: tài nguyên GPU miễn phí không đáng tin cho một pipeline dài. Tôi đã tách NB1/NB2 (rẻ, có thể chạy CPU hoặc ít GPU) khỏi NB3–NB5, sao lưu kết quả NB2 lên Drive, và không bao giờ chạy lại NB2 sau khi đã đóng băng — nhờ đó giữ được phần kết quả hợp lệ khi runtime bị mất. Một lần thử huấn luyện trên CPU cũng lỗi (`'functools.partial' object has no attribute '__func__'` ở `SFTTrainer`) và không được tính là một run hợp lệ.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** chạy NB3 với GPU T4 để có adapter `correct`, chạy NB5 để có phán quyết hồi quy và định tính (kèm ca FT thua), rồi chạy NB4 để so `attn_only`, `wrong_lr`, `qlora` bằng chỉ số target thay vì train loss; nếu thiếu thời gian thì đặt `EPOCHS=1` và ghi rõ điều đó trong báo cáo.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link: không làm
