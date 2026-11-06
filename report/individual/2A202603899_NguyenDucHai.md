# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603899
- **Họ và tên: Nguyễn Đức Hải**
- **SEED** (in ở ô đầu notebook): 1632

## 1. Bằng chứng


| Tệp                                                                  | Có? |
| -------------------------------------------------------------------- | --- |
| `[day01_lab.ipynb](day01_lab.ipynb)` — notebook đã chạy, còn output  | ☐   |
| `[lab1_report.json](lab1_report.json)` — do ô "Báo cáo cuối" sinh ra | ☐   |
| `[hand_calc.jpg](hand_calc.jpg)` — ảnh tính tay có các bước          | ☐   |




## 2. Kết quả chính

```
                             precision  recall  F1 (test)  F1 (SUB 40)  
```

1. BoW + LogReg (tự code)            1.000   0.854      0.921        0.919
2. BoW + MLP (deep learning)         1.000   0.896      0.945        0.974

3a. Encoder (MiniLM) + kNN           0.892   0.948      0.919        0.976  
3b. Encoder (MiniLM) + LogReg        0.968   0.948      0.958        1.000  
3c. Encoder (MiniLM) + MLP head      0.968   0.938      0.952        0.974  
4a. LLM nhỏ zero-shot                0.000   0.000      0.000        0.000  
4b. LLM lớn zero-shot                0.000   0.000      0.000        0.000  
4c. LLM nhỏ few-shot                 0.000   0.000      0.000        0.000   

```
                             train (s)  tin/s (cả lô)  p50 ms/tin  
```

1. BoW + LogReg (tự code)            5.333      38910.953       0.057
2. BoW + MLP (deep learning)        19.364      23453.004       0.326

3a. Encoder (MiniLM) + kNN          86.530         42.663      41.377  
3b. Encoder (MiniLM) + LogReg       86.964         52.084      24.818  
3c. Encoder (MiniLM) + MLP head     87.466         49.382      37.850  
4a. LLM nhỏ zero-shot                0.000          1.004     899.972  
4b. LLM lớn zero-shot                0.000          0.912    1061.850  
4c. LLM nhỏ few-shot                 0.000          1.077     853.400   

```
                             p95 ms/tin  USD / 1k tin  
```

1. BoW + LogReg (tự code)             0.095         0.000
2. BoW + MLP (deep learning)          0.413         0.000

3a. Encoder (MiniLM) + kNN           58.754         0.000  
3b. Encoder (MiniLM) + LogReg        35.569         0.000  
3c. Encoder (MiniLM) + MLP head      53.840         0.000  
4a. LLM nhỏ zero-shot              1463.985         0.007  
4b. LLM lớn zero-shot              1373.001         0.066  
4c. LLM nhỏ few-shot               1427.035         0.017  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

**(a) `llm_models`.** Đoán: model lớn hơn nhỏ 0.01–0.03 F1, đắt gấp 5–8 lần, LLM không thắng 3b. Thực tế: F1 = 0.000 ở cả hai model, `invalid = 40/40`. Chi phí gấp ~9,4 lần ($0.066 vs $0.007 mỗi 1k tin). Vì sao lệch: `CLASSIFY_MAX_TOKENS = 3` quá chặt. Khi gọi thử một tin, API trả `content = ''` và `finish_reason = 'length'`, nên `parse_label` trả `None` và mọi tin bị tính là ham. Tôi đã đoán về chất lượng LLM mà không kiểm tra pipeline có chạy đúng không.

**(b) `ml_lr`.** Đoán: lr = 50 làm loss nhảy loạn, không hội tụ. Thực tế: loss vẫn giảm về ~0.003. Vì sao lệch: logistic regression có loss lồi và gradient bị chặn (|p − y| ≤ 1, đầu vào 0/1), dữ liệu gần tách tuyến tính được. Bước lớn chỉ làm trọng số to ra và dự đoán chắc hơn trên train. Tôi đã nhầm "lr lớn" với "mất ổn định" như mạng sâu.

**(c) `emb_tricky`.** Đoán: Encoder + LogReg đoán đúng nhiều hơn BoW. Thực tế: hòa 5/6 và cả hai cùng sai tin "Last chance! Your account rewards expire tonight...". Vì sao lệch: head được train trên spam SMS 2011, không có kiểu lừa đảo tài khoản/phần thưởng. Encoder đóng băng không tự thêm được kiến thức đó. Chỉ 6 case nên không đủ để kết luận gì nhiều.

## 4. Phân tích 50 case khó

Model tụt nhiều nhất là **BoW + LogReg**: F1 0.919 → 0.378 (−0.54), accuracy chỉ 54%. Sau đó là BoW + MLP (0.974 → 0.462). Embedding tụt ít hơn (−0.26 đến −0.36). Nhóm yếu nhất của BoW là Thao túng/lừa đảo (12%), Né từ khoá (17%), Spam giọng tự nhiên (20%).

- **#29** `C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd...`: các từ bị viết lệch thành token lạ ngoài vocab nên bị bỏ qua. BoW chỉ thấy một vài token còn lại, không đủ để đẩy về spam. Embedding đạt 83% ở nhóm này vì WordPiece vẫn cắt chữ thành các mảnh còn mang tín hiệu.
- **#14** `Hi, I'm working from home and making about £300 a day...`: mọi model đều sai. Tin không có từ khoá spam, lại có `i, my` (nhóm từ BoW học là kéo về ham). Với embedding, tin nằm cạnh các tin chat thân mật. Đây là spam mà nội dung không đủ để phân biệt, cần thêm thông tin về người gửi.

BoW đạt 100% ở nhóm ham ("tưởng spam", "nói về spam") chủ yếu vì nó thiên về đoán ham: trong 27 case đúng, khoảng 19–20 là ham. Hàng LLM (4a, 4d) đang là 42% vì lỗi `invalid`, nên chưa phân tích được prompt injection. *(Cập nhật sau khi chạy lại.)*

## 5. Thử thách cá nhân

Đề #1: đổi ngưỡng 0.5 → 0.2 và 0.8 cho BoW + LogReg. Cách làm: tính `predict_proba` trên test một lần, so với từng ngưỡng rồi đếm FP.

| Ngưỡng | Precision | Recall | F1 | Ham bị chặn nhầm |
|---|---|---|---|---|
| 0.2 | 0.883 | 0.948 | 0.915 | 12 |
| 0.5 | 1.000 | 0.854 | 0.921 | 0 |
| 0.8 | 1.000 | 0.583 | 0.737 | 0 |

Hạ ngưỡng thì recall tăng nhưng precision giảm. Nâng ngưỡng thì ngược lại. Ngân hàng sợ chặn nhầm OTP nên cần FP gần 0. Cả 0.5 và 0.8 đều có 0 FP, nhưng 0.8 mất hơn 40% spam (recall 0.583), nên chọn **0.5**. Với 679 tin ham trong test và 0 FP, tỉ lệ chặn nhầm thực vẫn có thể lên tới khoảng 0.4%. Nên chọn ngưỡng trên tập val với ràng buộc FP, và kiểm riêng trên tin OTP/giao dịch (BoW chỉ đúng 83% ở nhóm này trong bộ 50 case).

## 6. Phương án cho sản phẩm thật

Yêu cầu: p95 < 50 ms, 10 triệu SMS/ngày (~116 tin/giây trung bình).

| Phương án | F1 test | F1 50 case khó | p95 | Đạt SLA |
|---|---|---|---|---|
| BoW + LogReg | 0.921 | 0.378 | 0.26 ms | Có |
| BoW + MLP | 0.945 | 0.462 | 0.57 ms | Có |
| Enc + MLP head (3c) | 0.952 | 0.653 | 39 ms | Có |
| Enc + LogReg (3b) | 0.958 | 0.640 | 58 ms | **Không** |
| LLM | chưa có | chưa có | ~1000 ms | Không |

**Chọn:** hệ thống hai tầng. Tầng 1 là BoW + MLP chạy trên mọi tin (p95 0.57 ms, chi phí gần 0). Tin có p(spam) nằm vùng lưỡng lự đi tầng 2 là 3c, hoặc LLM chạy bất đồng bộ. Chi phí LLM nếu chạy cho mọi tin: 300 triệu tin/tháng ≈ $2.100 (nhỏ, zero-shot), ≈ $5.100 (few-shot), ≈ $19.800 (lớn). Đây là ước tính từ lần chạy lỗi, sẽ cập nhật sau khi sửa.

**Đánh đổi:** BoW tầng 1 yếu ở spam giọng tự nhiên và lừa đảo (F1 khó 0.462), nên cần tầng 2 cứu. 3c chỉ nhanh 52 tin/s mỗi instance trên CPU, nên cần ít nhất 3 instance chỉ để phục vụ lưu lượng trung bình, và nhiều hơn cho giờ cao điểm. Vì vậy chỉ gửi một phần nhỏ tin sang tầng 2, tỉ lệ cần đo trên tập val.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời | Tôi sửa/kiểm chứng gì |
|---|---|---|---|
| 4 | Vì sao F1 của cả hai LLM đều 0.00? | Do `invalid = 40/40`, nghi `max_tokens = 3` | Chạy ô chẩn đoán: `content = ''`, `finish_reason = 'length'` → xác nhận. Giả thuyết "reasoning ăn token" của AI không đúng vì `reasoning_tokens = 0` |

**Điều hiểu ra hôm nay:** điểm cao trên test cùng phân phối không chứng minh model hiểu spam; chỉ cần đổi kiểu tin là F1 tụt gần một nửa. Một lỗi kỹ thuật nhỏ (`max_tokens = 3`) có thể làm cả hàng kết quả vô nghĩa mà không báo lỗi, nên phải luôn kiểm tra `n_invalid` trước khi đọc F1. Và bộ test nhỏ viết tay chỉ dùng để tìm kiểu lỗi, không dùng để ước lượng chất lượng thật, tác động của learning rate đối với quá trình training như thế nào, khác biệt giữa các phương pháp ML, DL, Encoder + head, LLM, ưu điểm nhược điểm của từng phương pháp