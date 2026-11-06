# Ngày 1 — Bài Tập & Phản Ánh (side lab TUỲ CHỌN)

> Side lab này **không** thuộc điểm chính của Day 01 (xem [RUBRIC.md](../RUBRIC.md) mục Bonus). Model: `gpt-6.1-terra` (lớn) và `gpt-6-luna` (nhỏ). `gpt-4o` / `gpt-4o-mini` bị gỡ khỏi API ngày 14/01/2027 ([nguồn](https://developers.openai.com/api/docs/deprecations)). Tên khoá `gpt4o_*` / `mini_*` trong kết quả là hợp đồng của test, giữ nguyên để tương thích: đọc là "model lớn" / "model nhỏ".
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Copy `src/optional_llm_api/template.py` thành `src/optional_llm_api/solution/solution.py`, triển khai mọi TODO, rồi chạy từ gốc repo: `pytest tests/test_solution.py -v`. Chưa có `solution/solution.py` thì test tự bỏ qua.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python src/optional_llm_api/solution/solution.py
```
Bạn sẽ thấy output so sánh phản hồi của `gpt-6.1-terra` và `gpt-6-luna`.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."** (nhớ gửi `reasoning_effort="none"`: model GPT-6 chỉ nhận `temperature` khi tắt suy luận)

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature = 0.0, phản hồi gần như xác định và tập trung vào một sự thật “an toàn”, ít biến thiên. Ở 0.5–1.0, câu trả lời đa dạng hơn, vẫn mạch lạc nhưng chọn chi tiết khác nhau giữa các lần chạy. Ở 1.5, văn phong tự do và bất ngờ hơn, đôi khi lan man hoặc kém chính xác hơn vì model lấy mẫu từ phần đuôi phân phối token.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Nên đặt khoảng 0.2–0.4. Chatbot hỗ trợ cần câu trả lời ổn định, chính xác và dễ dự đoán; temperature thấp giảm rủi ro bịa thông tin và giữ giọng điệu nhất quán. Có thể nâng nhẹ (≈0.5) nếu muốn câu trả lời tự nhiên hơn, nhưng không nên vượt quá 0.7 trong ngữ cảnh hỗ trợ khách hàng.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem `gpt-6.1-terra` đắt hơn `gpt-6-luna` bao nhiêu lần cho workload này** (giá 11/2026: terra $1.00 / $5.00, luna $0.10 / $0.50 mỗi 1M token input / output):
> Tổng request/ngày = 10.000 × 3 = 30.000. Giả sử ~350 token/request (đơn giản hóa, coi phần lớn là output hoặc dùng tỷ lệ input/output tương đương giữa hai model), chi phí tỷ lệ thuận với đơn giá token.  
> - Terra: input $1.00 + output $5.00 → trung bình khoảng $3/1M token nếu input ≈ output, hoặc cao hơn nếu output chiếm phần lớn.  
> - Luna: input $0.10 + output $0.50 → cùng tỷ lệ thì khoảng $0.30/1M token.  
> Hệ số chênh lệch ≈ **10×** (cả input và output terra đều đắt hơn luna đúng 10 lần: 1.00/0.10 và 5.00/0.50). Với 30.000 request × 350 token ≈ 10,5M token/ngày, chi phí terra ≈ 10 lần luna bất kể mix input/output.

**Mô tả một trường hợp mà chi phí cao hơn của `gpt-6.1-terra` là xứng đáng, và một trường hợp `gpt-6-luna` là lựa chọn tốt hơn:**
> **Terra xứng đáng:** soạn thảo báo cáo pháp lý, tóm tắt hồ sơ y tế, hoặc sinh code quan trọng — nơi sai một chi tiết có thể gây hậu quả lớn, cần độ chính xác và suy luận sâu.  
> **Luna tốt hơn:** chatbot FAQ, phân loại ticket đơn giản, gợi ý tìm kiếm, hoặc tạo nội dung marketing số lượng lớn — volume cao, yêu cầu chất lượng “đủ dùng”, tối ưu chi phí quan trọng hơn độ tinh tế tối đa.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất với chatbot hội thoại và công cụ viết dài (soạn email, code, bài viết): người dùng thấy token xuất hiện ngay, cảm giác phản hồi nhanh và có thể dừng sớm nếu nội dung lệch hướng. Non-streaming phù hợp hơn khi cần kết quả đầy đủ trước khi xử lý tiếp (ví dụ: parse JSON, chấm điểm, pipeline batch), hoặc khi UI không hỗ trợ hiển thị từng phần — lúc đó chờ một lần rồi nhận toàn bộ câu trả lời gọn và dễ xử lý hơn.

---

## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/test_solution.py -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [ ] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
