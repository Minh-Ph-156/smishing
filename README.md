# Vietnamese SMS Smishing & Spam Detection Pipeline

Dự án nghiên cứu và phát triển hệ thống phát hiện tin nhắn rác và tin nhắn lừa đảo (Smishing / Spam) tiếng Việt với bài toán phân loại **3 lớp (3-Class Classification)**, áp dụng các kỹ thuật cân bằng dữ liệu bằng dịch ngược bảo toàn ngữ cảnh (Back-Translation with Masking), kết hợp đa dạng các họ mô hình từ **Machine Learning**, **Deep Learning (Recurrent + Multi-modal)** đến **Transformers (Pre-trained Language Models)**.

---

## 🏛️ Sơ đồ Kiến trúc Pipeline Toàn diện

```text
                     Tập dữ liệu SMS Tiếng Việt (2,991 mẫu)
                                        │
                                        ▼
             [ 1. Gán nhãn 3 lớp: Ham (0) │ Spam (1) │ Smishing (2) ]
                                        │
                                        ▼
             [ 2. Phân tầng Stratified Train / Test Split (80/20) ]
                    (Đóng băng hoàn toàn Test set: 20%)
                                 ┌──────┴──────┐
                                 │             │
                                 ▼             ▼
                          TẬP TRAIN (80%)  TẬP TEST (20% ĐÓNG BĂNG)
                                 │             │
                                 ▼             │
                  [ 3. Back-Translation Masking ]│
               (Capped Oversampling 5-10x Spam) │
                                 │             │
                                 ▼             │
             [ 4. Tiền xử lý văn bản & Trích xuất đặc trưng ] ◄──┘
                   (Chuẩn hóa NFC, Mask URL, 8 Đặc trưng số)
              ┌───────────────┬─────────────────┬──────────────────────┐
              ▼               ▼                 ▼                      ▼
         [ TF-IDF ]   [ 8 Đặc trưng số ] [ SimpleVocab ]     [ Tokenizer Trực tiếp ]
              │               │                 │                      │
              └───────┬───────┘                 │                      │
                      │               ┌─────────┴─────────┐            ├─────────────────────┐
                      │               │                   │            ▼                     ▼
                      │               │                   │   [ Word-Level Seg (pyvi) ] [ Syllable / Subword ]
                      │               │                   │            │                     │
                      ▼               ▼                   ▼            ▼                     ▼
               HỌ MÔ HÌNH ML       HỌ DL RECURRENT (Multi-modal)   PhoBERT-v2          mBERT / DistilBERT
               • SVM (LinearSVC)   • Bi-RNN + 8 Feats          (vinai/phobert-v2)      DeBERTa-v3
               • Decision Tree     • Bi-LSTM + 8 Feats                               (mdeberta-v3-base)
               • KNN               • Bi-GRU + 8 Feats
               • FastText+XGBoost
                      │                         │                              │
                      └────────────────┬────────┴──────────────────────────────┘
                                       ▼
                            [ 5. Unified Benchmark Evaluation ]
          (Acc, Precision, Recall, F1-Score, FPR, FNR, ROC-AUC & Ma trận nhầm lẫn)
                                       │
                      ┌────────────────┼────────────────┐
                      ▼                ▼                ▼
            [ Biểu Đồ Cột Nhóm ] [ Xuất Bảng LaTeX ] [ TensorBoard ]
             (F1, Acc, ROC-AUC)  (Chuẩn booktabs)   (Event Logs)
```

---

## 🏷️ Quy chuẩn Gán nhãn 3 lớp (`label_3class`)

Tập dữ liệu gốc bao gồm **2,991 mẫu** tin nhắn được rà soát và phân loại nghiêm ngặt:

| Nhãn | Tên lớp | Số lượng | Tỷ lệ (%) | Định nghĩa & Tiêu chí nhận diện |
| :---: | :--- | :---: | :---: | :--- |
| **`0`** | **Ham** | 2,188 | 73.15% | Tin nhắn hợp pháp: giao tiếp cá nhân, mã OTP ngân hàng, biến động số dư tài khoản, thông báo cước nhà mạng (`[TB]`), giao hàng Viettel Post (`yk.vtp.vn`). |
| **`1`** | **Spam** | 32 | 1.07% | Tin nhắn rác quảng cáo/tiếp thị thương mại **không có yếu tố lừa đảo**: Bất động sản, Spa/chăm sóc da, Khóa học ngoại ngữ, Bán sim số đẹp, Lắp mạng internet, Khuyến mại shop... |
| **`2`** | **Smishing** | 771 | 25.78% | Tin nhắn độc hại/lừa đảo: Phishing mạo danh ngân hàng (techcombank.shop, vcb-doi-thuong...), Cờ bạc/cá cược (pkwin, 8qbet, g8858), App tín dụng đen đòi nợ đe dọa bôi nhọ (Ví Vay, HDCC), Phạt nguội CSGT/VNeID giả mạo, Bẫy tuyển CTV Shopee/TikTok làm nhiệm vụ, Chiếm đoạt mã OTP. |

---

## ⚖️ Phân tích Kỹ thuật: Mất cân bằng Cực đoan Lớp Spam & Cơ chế Cân bằng

### 1. Thách thức Số liệu Thực tế
Khi chia phân tầng **Stratified Split 80/20** trên 2,991 mẫu:
- **Tập Train (80% - 2,392 mẫu)**: Ham = 1,750 mẫu, Smishing = 617 mẫu, nhưng **Spam chỉ có vỏn vẹn 26 mẫu** (1.09%).
- **Tập Test (20% - 599 mẫu - Đóng băng nguyên bản)**: Ham = 438 mẫu, Smishing = 154 mẫu, **Spam chỉ có đúng 6 mẫu** (1.00%).

### 2. Rủi ro khi Ép Cân bằng 1:1:1 Tuyệt đối (Oversampling 67 lần)
- Nếu ép đưa lớp Spam từ 26 mẫu lên 1,750 mẫu (sinh thêm 1,724 mẫu), tỷ lệ nhân bản lên tới **67 lần**.
- Với dịch ngược (Back-Translation) qua 2 ngôn ngữ trung gian (`['en', 'fr']`), 26 câu gốc chỉ có thể sinh ra tối đa khoảng $26 \times 2 = 52$ câu dịch biến thể có nghĩa khác biệt. Việc ép sinh 1,724 mẫu buộc mô hình lặp lại các biến thể đó hơn 33 lần, dẫn tới **Severe Overfitting (học vẹt)** vào 26 mẫu gốc.
- Trên tập Test, vì Spam chỉ có 6 câu, nếu mô hình đoán sai dù chỉ 1 câu thì Recall lớp Spam sụt giảm ngay $16.67\%$, kéo tụt toàn bộ chỉ số Macro Recall và Macro F1 của hệ thống.

### 3. Giải pháp Kỹ thuật Đã Triển khai
1. **Capped Oversampling kết hợp Translation Cache**: Đặt ngưỡng nhân bản an toàn `max_oversample_factor = 10` (sinh tối đa khoảng 150 – 260 mẫu Spam chất lượng cao) thay vì nhân bản vô tội vạ 67 lần.
2. **Cost-Sensitive Learning (Class Weights)**:
   - Trong ML: Thiết lập `class_weight='balanced'` trong `LinearSVC` và `DecisionTreeClassifier`.
   - Trong DL: Tự động tính toán trọng số nghịch đảo tần suất lớp truyền vào hàm mất mát `nn.CrossEntropyLoss(weight=class_weights)`.
3. **Thuật toán Sinh mẫu Tối ưu**: Thay thế vòng lặp $O(N^2)$ cũ bằng biến đếm $O(1)$ độc lập (`count_spam`, `count_smishing`) và mã định danh tách biệt (`AUG_SPAM_xxxx`, `AUG_SMISHING_xxxx`).

---

## 🔤 Đồng bộ Đặc trưng Tokenizer & Phân đoạn Từ ghép (Word-level vs Syllable-level)

1. **Word-level Segmentation & Class Weights cho PhoBERT (Khắc phục Recall)**:
   - `PhoBERT` (`vinai/phobert-base-v2`) được huấn luyện trên kho ngữ liệu tiếng Việt đã được phân đoạn từ ghép (nối bằng dấu gạch dưới, ví dụ `tài_khoản`, `ngân_hàng`, `tin_nhắn`).
   - Pipeline tích hợp mô-đun phân đoạn qua `pyvi` (`ViTokenizer.tokenize`) riêng biệt cho đầu vào của nhánh PhoBERT, đồng thời bảo vệ các token nhạy cảm không bị chia cắt sai.
   - Bổ sung tham số **Class Weights** nghịch đảo tần suất lớp vào hàm mất mát `nn.CrossEntropyLoss(weight=class_weights)` để phạt nặng mô hình khi bỏ sót các lớp thiểu số (Spam/Smishing), trực tiếp kéo tăng chỉ số Recall và F1.
2. **Chiến lược Tinh chỉnh Tối ưu cho DistilBERT (Cứu DistilBERT)**:
   - Tăng số epoch huấn luyện riêng cho DistilBERT lên **5 epochs** (so với 3 epochs của PhoBERT/mBERT), giúp kiến trúc 6 layers thu nhỏ có đủ số bước cập nhật gradient để học phân tách 3 lớp.
   - Tinh chỉnh Learning Rate xuống mức **`1.5e-5`** kết hợp **Linear Warmup Scheduler (10% warmup steps)** và hàm mất mát phạt theo trọng số lớp để mô hình hội tụ ổn định, tránh tình trạng bỏ quên lớp thiểu số Spam.
3. **Syllable-level cho mBERT**:
   - `bert-base-multilingual-cased` hoạt động trực tiếp trên mức âm tiết (syllable-level WordPiece/BPE) nên nhận chuỗi đã làm sạch không cần tách từ ghép.
4. **Đăng ký Token Đặc biệt (Special Tokens Registration)**:
   - Toàn bộ 9 token định danh chuẩn hóa: `[URL]`, `[MONEY]`, `[TIME]`, `[DATE]`, `[NUMBER]`, `[NAME]`, `[POINT]`, `[TB]`, `[QC]` được đăng ký chính thức qua `tokenizer.add_special_tokens()` và mở rộng ma trận embedding bằng `model.resize_token_embeddings()`. Nhờ đó, các ký tự ngoặc vuông không bị xé nhỏ thành các subwords vô nghĩa.

---

## 📁 Cấu trúc Thư mục Dự án

```text
Smishing/
├── data/                                    # Dữ liệu nguồn và dữ liệu đã phân chia
│   ├── vietnamese_sms_dataset.csv           # Tập dữ liệu gốc đã gán nhãn 3 lớp ('label_3class')
│   ├── vietnamese_sms_dataset.csv.bak       # Bản sao lưu dữ liệu gốc
│   ├── train_balanced.csv                   # Tập Train sau khi tăng cường bằng Back-Translation
│   └── test_frozen.csv                      # Tập Test đóng băng nguyên bản (20% test set)
│
├── src/                                     # Toàn bộ mã nguồn module hóa
│   ├── __init__.py                          # Package initialization
│   ├── Data_preprocessing.py                # Chuẩn hóa NFC, Mask URL/Domain, trích xuất 8 đặc trưng số an toàn index
│   ├── Back_translation.py                  # Dịch ngược an toàn với TextMasker, AccentHandler & Capped Oversampling
│   ├── ML_model.py                          # Huấn luyện ML: SVM (Softmax decision), Decision Tree, KNN (Class Weights)
│   ├── DL_model.py                          # Huấn luyện DL: Bi-RNN, Bi-LSTM, Bi-GRU (Multi-modal Fusion 8 feats)
│   ├── Transformer_model.py                 # Fine-tuning: PhoBERT (ViTokenizer), mBERT, DistilBERT (Special Tokens)
│   ├── metrics.py                           # Bộ metric toàn diện, hỗ trợ Softmax ROC-AUC, xuất LaTeX booktabs
│   ├── pipeline.py                          # Điều phối pipeline tích hợp và xuất kết quả
│   └── utils.py                             # Tiện ích định danh cột và nhãn lớp
│
├── results/                                 # Kết quả nghiên cứu và xuất bản
│   └── performance_table.tex                # Bảng mã LaTeX chuẩn booktabs (\multicolumn{8}{l})
│
├── runs/                                    # TensorBoard event logs
│   └── experiment_sms/                      # Scalar metrics và đồ thị trực quan
│
├── main.py                                  # Entry point thực thi với giao diện dòng lệnh argparse
└── README.md                                # Tài liệu học thuật và báo cáo kỹ thuật toàn diện
```

---

## 📊 Bộ Chỉ số Đánh giá Chuyên sâu (Evaluation Metrics)

Hệ thống tính toán đầy đủ 7 chỉ số học thuật theo chuẩn các bài báo bảo mật & NLP quốc tế:

1. **Accuracy (Acc)**: Độ chính xác tổng thể trên toàn tập kiểm thử.
2. **Precision (P)**: Macro-averaged Precision:
   $$\text{Macro-P} = \frac{1}{C}\sum_{i=1}^C \frac{TP_i}{TP_i + FP_i}$$
3. **Recall (R)**: Macro-averaged Recall:
   $$\text{Macro-R} = \frac{1}{C}\sum_{i=1}^C \frac{TP_i}{TP_i + FN_i}$$
4. **F1-Score (F1)**: Macro-averaged F1-Score (Trung bình cộng $F1_i$ từng lớp theo chuẩn `scikit-learn`):
   $$\text{Macro-F1} = \frac{1}{C} \sum_{i=1}^{C} \text{F1}_i = \frac{1}{C} \sum_{i=1}^{C} \left( \frac{2 \times P_i \times R_i}{P_i + R_i} \right)$$
5. **False Positive Rate (FPR)**: Tỷ lệ cảnh báo nhầm của hệ thống:
   $$\text{FPR}_i = \frac{FP_i}{FP_i + TN_i} \quad \implies \quad \text{Macro-FPR} = \frac{1}{C}\sum_{i=1}^C \text{FPR}_i$$
6. **False Negative Rate (FNR)**: Tỷ lệ bỏ lọt tin nhắn độc hại:
   $$\text{FNR}_i = \frac{FN_i}{FN_i + TP_i} = 1 - \text{Recall}_i \quad \implies \quad \text{Macro-FNR} = \frac{1}{C}\sum_{i=1}^C \text{FNR}_i$$
7. **ROC-AUC (AUC)**: Diện tích dưới đường cong ROC đa lớp (One-vs-Rest Macro Average).
   - Với các mô hình không có `predict_proba` như **LinearSVC**, hệ thống áp dụng hàm Softmax chuẩn hóa trên vector khoảng cách `decision_function` để tạo phân phối xác suất hợp lệ trước khi tính `roc_auc_score(multi_class='ovr')`.

---

## 🚀 Hướng dẫn Sử dụng

### 1. Cân bằng Dữ liệu bằng Back-Translation (Khuyến nghị thực hiện đầu tiên)
Phân chia Stratified 80/20, đóng băng 100% Test set, và sinh mẫu tăng cường cho Spam & Smishing:
```bash
python src/Back_translation.py
```
*Ghi chú: Kết quả sẽ tạo 2 file `data/train_balanced.csv` và `data/test_frozen.csv`.*

### 2. Khởi chạy Toàn diện Hệ thống Huấn luyện & Đánh giá
Chạy toàn bộ tiền xử lý, huấn luyện cả 3 nhóm (ML, DL, Transformers), log TensorBoard và xuất LaTeX:
```bash
python main.py
```

**Các tùy chọn dòng lệnh linh hoạt (`argparse`):**
- Chạy trên máy cá nhân không có GPU hoặc muốn đánh giá nhanh nhóm ML & DL:
  ```bash
  python main.py --skip-transformer
  ```
- Tự động kiểm tra và chạy cân bằng dữ liệu nếu chưa có file `train_balanced.csv`:
  ```bash
  python main.py --balance
  ```
- Tùy chỉnh hệ số oversampling tối đa cho lớp Spam:
  ```bash
  python main.py --max-spam-factor 10
  ```

### 3. Theo dõi Trực quan trên TensorBoard
Khởi chạy TensorBoard web server để quan sát các biểu đồ so sánh:
```bash
tensorboard --logdir runs/
```
Mở trình duyệt truy cập: `http://localhost:6006`

### 4. Kết xuất Bảng LaTeX cho Bài báo Khoa học
Sau khi chạy pipeline, file mã nguồn LaTeX được lưu tại `results/performance_table.tex`. Cú pháp đã được hiệu chỉnh chuẩn `booktabs` với `\multicolumn{8}{l}` để bảng hiển thị đẹp mắt, không làm lệch cột:

```latex
\begin{table}[htbp]
\caption{Performance evaluation of ML, DL, and Transformer methods on Vietnamese SMS dataset}
\label{tab:performance_evaluation}
\centering
\begin{tabular}{lccccccc}
\toprule
Method & Acc & P & R & F1 & FPR & FNR & AUC \\
\midrule
\multicolumn{8}{l}{\textbf{TF-IDF Vectorizer + Numerical Features}} \\
SVM (LinearSVC)           & 0.9683 & 0.9688 & 0.9238 & 0.9435 & 0.0222 & 0.0762 & 0.9850 \\
Decision Tree             & 0.9516 & 0.8985 & 0.9134 & 0.9054 & 0.0321 & 0.0866 & 0.9376 \\
K-Nearest Neighbors (KNN) & 0.9516 & 0.9596 & 0.9023 & 0.9284 & 0.0429 & 0.0977 & 0.9485 \\
\midrule
\multicolumn{8}{l}{\textbf{Deep Learning (Recurrent Models) + Numerical Features}} \\
Bi-RNN                    & 0.9316 & 0.8830 & 0.8918 & 0.8872 & 0.0517 & 0.1082 & 0.9699 \\
Bi-LSTM                   & 0.9399 & 0.8896 & 0.8997 & 0.8944 & 0.0440 & 0.1003 & 0.9849 \\
Bi-GRU                    & 0.9533 & 0.9565 & 0.9100 & 0.9306 & 0.0355 & 0.0900 & 0.9855 \\
\midrule
\multicolumn{8}{l}{\textbf{Transformers (Pre-trained Language Models)}} \\
PhoBERT (RoBERTa)         & 0.9783 & 0.9818 & 0.9270 & 0.9519 & 0.0190 & 0.0730 & 0.9912 \\
BERT (mBERT-base)         & 0.9666 & 0.9678 & 0.9216 & 0.9421 & 0.0243 & 0.0784 & 0.9489 \\
DistilBERT (Multilingual) & 0.9599 & 0.9604 & 0.9186 & 0.9366 & 0.0273 & 0.0814 & 0.9941 \\
\bottomrule
\end{tabular}
\end{table}
```
