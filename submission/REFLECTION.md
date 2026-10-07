# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Baseline (b) mạnh hơn tôi nghĩ. Chỉ bằng cách đổi prompt (đưa schema và chỉ dẫn rõ ràng), mô hình gốc Qwen3.5-4B đi từ target 0.000 / format 0.000 lên target 0.765 / format 1.000, và còn nhanh gấp khoảng 3 lần (3190 ms xuống 1048 ms). Tôi từng cho rằng muốn model trả JSON đúng schema thì phải fine-tune; kết quả NB2 cho thấy khoảng trống mà fine-tune còn phải lấp chỉ còn tối đa khoảng 0.235.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải. Tôi dự đoán thời gian sẽ nằm ở việc huấn luyện và đánh giá, nhưng thực tế phần lớn thời gian mất vào hạ tầng: Colab hết quota GPU nên tôi không chạy được NB3/NB4/NB5 trên GPU, rồi thử NB3 và NB4 trên CPU thì cả hai đều lỗi ở `SFTTrainer` (`'functools.partial' object has no attribute '__func__'`). Tôi cũng mất thời gian vì lỡ chạy NB1/NB3 trên một máy ảo mới mà chưa chép kết quả NB2 sang, nên phải khôi phục `baselines_frozen.json` từ bản sao lưu trên Drive (và không chạy lại NB2 để giữ mốc đã đóng băng).

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng tin fine-tune luôn là bước cần thiết và luôn tốt hơn prompt cho một tác vụ có định dạng riêng. Giờ tôi coi nó là một giả thuyết phải chứng minh: phải đo baseline prompt tốt (b) trước, rồi mới xem bản fine-tune có vượt không, đồng thời phải kiểm tra nó có làm hỏng kiến thức chung (regression) hay không. Tài liệu của lab cũng ghi một lần chạy model nhỏ hơn vừa tăng target vừa làm regression tụt mạnh, nên "target cao" chưa đủ để kết luận.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để đọc repo và hiểu pipeline NB1–NB6, hướng dẫn cấu hình Colab (mount Drive, sao lưu `results/`), lên kế hoạch gói bài nộp và soạn bản nháp báo cáo từ các file kết quả có sẵn. Về chỗ sai: tôi chưa ghi nhận một lỗi sự thật cụ thể nào của AI trong phần kết quả đã đo; các con số trong báo cáo tôi đối chiếu với `results/baselines_frozen.json` và `results/mask_proof.json`. Một hạn chế rõ là AI không thể thay thế GPU: kế hoạch chạy đủ pipeline phụ thuộc vào quota mà nó không kiểm soát được, nên tôi phải chuyển sang nộp bài một phần và ghi rõ phần nào chưa đo.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đóng băng một tập đánh giá đại diện (cả tập target và tập regression), rồi đo baseline prompt tốt nhất có thể trước khi huấn luyện bất kỳ thứ gì. Chỉ khi baseline đó không đáp ứng yêu cầu của khách hàng thì tôi mới fine-tune, và tôi sẽ chạy bằng chứng mask trước, đặt cổng hồi quy làm điều kiện để triển khai.
