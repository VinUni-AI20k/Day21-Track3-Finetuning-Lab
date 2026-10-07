# Lab 21 — Evaluation Report

**Họ tên**: Dương Minh Hiếu
**MSSV**: 2A202602488
**Ngày chạy full evaluation**: 07/10/2026
**Tier**: T4
**Base model**: `unsloth/Qwen3.5-4B`
**GPU thực tế**: Tesla T4, 14.6 GB VRAM, fp16

> Kết quả trong báo cáo lấy từ lần chạy full evaluation trên Colab: 50 mẫu target
> và 15 mẫu regression. Lần smoke run trước đó chỉ có 8 mẫu và không được dùng làm
> số liệu chính ở đây.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage gồm `intent`, `urgency`, `product`, `sentiment` |
| Train / val | 225 / 25, split seed 42 |
| `max_length` | 1024; p95 đo được 98 token, p99 100, max 101; NB1 gợi ý 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 optimizer steps |

Mục tiêu chọn bài toán ticket CSKH vì nhãn JSON có bốn trường kiểm tra được trực
tiếp, đồng thời có tập regression riêng để đo xem mô hình có quên năng lực phổ thông
không. Tôi giữ nguyên base model, dataset và prompt đã đóng băng giữa các bước để
phép so sánh không bị thay đổi đầu vào.

**Template có giữ khối `<think>` không?** Có. NB1 in verdict `reasoning preserved`;
chuỗi render vẫn có cả `<think>...</think>` và câu trả lời. Tuy nhiên, lần đánh giá
fine-tune cho `valid_trace_rate = 0.0`, nên việc template giữ trace không đồng nghĩa
adapter vẫn tạo trace hợp lệ.

Tôi giữ `max_length=1024` theo tier T4 của cấu hình chạy, dù p95 đo được chỉ 98 token
và giá trị gợi ý là 256. Thống kê cũng cho thấy mẫu dài nhất là 101 token, nên 1024
là cấu hình dư nhiều so với dữ liệu này; với lần chạy sau, tôi sẽ dùng 256 để giảm
padding và ghi nhận rõ việc thay đổi cấu hình. Đây là hạn chế về hiệu quả, không phải
bằng chứng cho thấy câu trả lời bị cắt trong lần chạy hiện tại.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` của mẫu proof | 0.4149 (39/94 token) |
| Tỷ lệ supervised trung bình train được NB3 in | 43.0% (9014/20951 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn supervised được giải mã ngược chứa phần trả lời sau `</think>` cùng JSON, kết
thúc bằng token kết thúc lượt assistant. Phần system/user prompt không nằm trong
đoạn được tính loss. Hai assert `answer_is_supervised` và `question_is_masked` đều
xanh.

```text
</think>
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3214.4 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1017.4 |
| (c) LoRA fine-tune | 0.970 | 0.5444 | 1.000 | 1367.1 |

**(b) có thật sự mạnh hơn (a) không?** Có. Trên 50 ticket, target tăng từ 0.000
lên 0.765, format tăng từ 0 lên 1.000, regression giữ nguyên 0.7911 và latency giảm
từ 3214.4 ms xuống 1017.4 ms. Prompt (b) có schema, tập giá trị hợp lệ và ví dụ JSON
cụ thể; tôi không sửa prompt này giữa các bước.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable params | LR | train loss (NB4) | target (NB5 §4) | train seconds | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6252 | 0.970 | 397.8 | 8.78 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 1e-4 | 0.5376 | 0.970 | 263.6 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.000 | 395.5 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 462.9 | 3.86 |

Cả bốn run dùng 30 optimizer steps. `attn_only` có 32,456,704 tham số trainable,
chỉ lệch khoảng 0.025% so với `correct` (32,464,896), nên đây là phép đối chứng
khớp ngân sách tham số.

### 4.1 — Vị trí adapter so với rank

Trên target, `attn_only` hòa `correct`: cả hai đạt 0.970. Train loss của `attn_only`
thấp hơn (0.5376 so với 0.6252), nhưng chênh lệch đó không tạo ra cải thiện target
đo được trên tập 50 mẫu. Vì ngân sách tham số gần như bằng nhau, kết quả này không
ủng hộ kết luận rằng chỉ tăng rank ở q/v sẽ thắng; trong thí nghiệm này, vị trí
adapter cũng không làm thay đổi điểm target cuối. Kết luận hợp lý là hai thiết kế
hòa theo metric target hiện có, chứ không phải rank hay vị trí đã được chứng minh
là vô ích nói chung.

### 4.2 — `wrong_lr`

`wrong_lr` chỉ giảm learning rate từ 1e-4 xuống 1e-5. Log huấn luyện của nó giảm
chậm hơn: loss cuối là 1.5702, trong khi `correct` là 0.6252; mean token accuracy
ở cuối log cũng thấp hơn (0.7909 so với 0.9965). Trên NB5, `wrong_lr` đạt target
0.000 và format 0.000, trái ngược với `correct` đạt 0.970 và 1.000. Nếu chỉ dựa
vào việc loss có giảm mà không so tốc độ hội tụ và đánh giá ngoài train, có thể
nhầm rằng cấu hình LR thấp vẫn đang học đủ tốt; kết quả cho thấy LR này không phù
hợp với ngân sách 30 bước của phép thử.

### 4.3 — QLoRA

QLoRA dùng 3.86 GB VRAM thay vì 8.78 GB, tiết kiệm 4.92 GB, tương đương khoảng
56%. Đổi lại, train mất 462.9 giây thay vì 397.8 giây và target đạt 0.940, thấp
hơn `correct` 0.030; format vẫn bằng 1.000. Kết quả này cho thấy QLoRA là lựa chọn
có ích khi giới hạn VRAM quan trọng, nhưng trong cấu hình và tập eval này nó chậm
hơn và kém target một chút. Chênh lệch nhỏ trên 50 mẫu chưa đủ để khẳng định mọi
Qwen3.5 đều không nên dùng QLoRA; nó chỉ cung cấp bằng chứng thực nghiệm yếu theo
hướng khuyến nghị đó cho cấu hình đang thử.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.247` · `valid_trace_rate = 0.0`

Fine-tune cải thiện rõ nhiệm vụ đích: target tăng từ 0.765 của base với prompt tối
ưu lên 0.970, còn format giữ ở 1.000. Nhưng điểm regression giảm từ 0.7911 xuống
0.5444, tức giảm 0.2467, vượt xa mức suy giảm tối đa 0.020 mà cổng cho phép. Do đó
không thể gọi adapter là một cải tiến tổng thể, dù nó phân loại ticket tốt hơn.
`valid_trace_rate=0.0` cũng cho thấy model không tạo reasoning trace hợp lệ trong
phép đo này; tôi không xem việc template giữ `<think>` là bằng chứng model đã bảo
tồn năng lực reasoning. Kết quả hợp lý là giữ verdict `FAILED`, không nới ngưỡng
để ép pass. Hướng thử tiếp theo là bổ sung một lượng replay data phổ thông như lab
gợi ý, rồi chạy lại cùng eval đóng băng để xem có thể giữ target cao mà giảm
forgetting hay không. Cho tới khi regression được khắc phục, tôi không khuyến nghị
deploy adapter này như một model đa dụng; nếu chỉ dùng cho triage ticket thì vẫn
cần kiểm thử vận hành và giới hạn phạm vi.

---

## 6. Định tính — gồm trường hợp fine-tune sai và đúng

`qualitative.json` ghi điểm và dự đoán fine-tune, nhưng không có dự đoán baseline
(b) cho từng ticket. Các ca “sai” và “đúng” bên dưới vì vậy được so với ground truth,
không được diễn giải là so sánh trực tiếp với baseline. Với ba ca score 0.75, các
trường `intent`, `product` và nhãn còn lại khớp; dự đoán `urgency=trung_binh` là
sai so với `thap`.

| # | Ticket (rút gọn) | Nhãn đúng (intent / urgency / product / sentiment) | Dự đoán fine-tune quan sát được | Nhận xét |
|---|---|---|---|---|
| 1 | Bình giữ nhiệt VN804124 — chưa thấy tiền | `hoan_tien / thap / bình giữ nhiệt / tich_cuc` | `hoan_tien / trung_binh / bình giữ nhiệt / tich_cuc` | ❌ FT sai urgency (score 0.75) |
| 2 | Nồi chiên không dầu DH249548 — thiếu phụ kiện | `san_pham_loi / thap / nồi chiên không dầu / trung_tinh` | `san_pham_loi / trung_binh / nồi chiên không dầu / trung_tinh` | ❌ FT sai urgency (score 0.75) |
| 3 | Áo khoác gió VN613097 — bị lỗi | `san_pham_loi / thap / áo khoác gió / tich_cuc` | `san_pham_loi / trung_binh / áo khoác gió / tich_cuc` | ❌ FT sai urgency (score 0.75) |
| 4 | Đèn bàn LED VN339109 — vỡ khi nhận, gấp | `san_pham_loi / cao / đèn bàn LED / trung_tinh` | Khớp cả 4 trường (score 1.00) | ✅ FT đúng |
| 5 | Balo laptop DH863123 — đổi size | `doi_tra / thap / balo laptop / tieu_cuc` | Khớp cả 4 trường (score 1.00) | ✅ FT đúng |

Ba lỗi được cung cấp đều là dự đoán urgency `trung_binh` trong khi nhãn là
`thap`, dù nội dung có cụm “khi nào tiện”; đây là dấu hiệu cần kiểm tra calibration
của trường urgency. Hai mẫu score 1.00 cho thấy model vẫn xử lý đúng các ticket có
intent khác nhau, nhưng không xoá được lỗi urgency còn lại trên tập đánh giá. Do
file không chứa prediction-level output của baseline (b), không thể nói liệu
fine-tune thắng hay thua baseline trên từng hàng này. Để lập bảng thắng/thua đối
với (b) theo từng mẫu, cần lưu dự đoán của cả baseline và adapter cho cùng 50 ticket.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Trong thí nghiệm này, fine-tuning có ích cho mục tiêu hẹp là phân loại
ticket: target tăng 20.5 điểm phần trăm so với base model đã có prompt tốt, và định
dạng JSON tiếp tục hợp lệ. Tuy nhiên, model trả giá bằng mức giảm 24.7 điểm phần
trăm trên regression, lớn hơn nhiều so với ngưỡng cho phép, nên kết luận triển khai
phải dựa trên toàn bộ bốn nhóm đánh giá chứ không chỉ target. Tôi chưa deploy adapter
này như một model tổng quát. Nếu sản phẩm chỉ cần triage ticket, adapter có thể là
một ứng viên cho thử nghiệm có giám sát, nhưng cần replay data, kiểm thử trên tập
độc lập lớn hơn và đánh giá các lỗi urgency trước khi dùng thật. Đối chứng `wrong_lr`
cho thấy learning rate có ảnh hưởng lớn trong ngân sách 30 bước; đối chứng
`attn_only` hòa `correct` trên target dù train loss thấp hơn, nên train loss không
thay thế được đánh giá mục tiêu. QLoRA giảm VRAM đáng kể nhưng chưa cải thiện chất
lượng hay tốc độ trong phép thử này. Cuối cùng, mask đúng là điều kiện nền tảng:
nó chứng minh loss học câu trả lời thay vì prompt, nhưng không thể tự ngăn
catastrophic forgetting. Bước tiếp theo nên là thêm replay data và đo lại cả target
lẫn regression, không phải nới gate.

**Ba điều tôi học được**:
1. Prompt được thiết kế tốt đã đưa baseline target từ 0.000 lên 0.765; cần đo prompt baseline trước khi quy công cho fine-tuning.
2. `attn_only` và `correct` khớp ngân sách tham số, nhưng hòa target dù loss train khác nhau; phải xếp hạng bằng metric trên eval.
3. Một model có thể tăng target lên 0.970 mà vẫn không đạt yêu cầu tổng thể vì regression giảm mạnh; quyết định deploy cần xét regression gate.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** thêm 1–5% replay data phổ thông theo khuyến nghị của lab, giữ nguyên tập eval và prompt đã đóng băng, sau đó train lại và kiểm tra xem regression có phục hồi mà target vẫn cao không. Tôi cũng sẽ lưu prediction-level output của baseline (b) và fine-tune cho từng ticket để phân tích chính xác các ca thắng/thua.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link: chưa thực hiện
