# NLP_Classification_Bipolar-Disorder

Dự án này xây dựng một pipeline NLP đa mô hình để phân loại các vấn đề sức khỏe tâm thần từ văn bản và bài đăng trên mạng xã hội. Mục tiêu là tự động nhận diện và phân loại các dấu hiệu bệnh lý dựa trên ngôn ngữ tự nhiên, nhằm hỗ trợ nghiên cứu và ứng dụng trong lĩnh vực y tế tâm thần.

## Tóm tắt

Đề tài tập trung vào việc xây dựng hệ thống phân loại tự động các nội dung liên quan đến sức khỏe tâm thần bằng các mô hình học sâu và kỹ thuật xử lý ngôn ngữ tự nhiên. Hệ thống được thiết kế để nhận diện các lớp bệnh lý phổ biến như ADHD, Anxiety, Bipolar, Depression, PTSD cùng với lớp không thuộc các nhóm này.

Dự án sử dụng dữ liệu từ nhiều nguồn và thực hiện các bước tiền xử lý, biểu diễn văn bản, huấn luyện và đánh giá trên nhiều kiến trúc mô hình khác nhau. Kết quả của đề tài có thể được dùng như một công cụ hỗ trợ sàng lọc ban đầu hoặc nền tảng nghiên cứu tiếp theo trong lĩnh vực NLP y tế.

## Công nghệ sử dụng

### Ngôn ngữ và môi trường
- Python 3.10+
- Jupyter / Python script
- Windows PowerShell / Anaconda Prompt

### Mô hình và thư viện chính
- LSTM
- BERT
- RoBERTa
- PhoBERT
- Transformers
- PyTorch
- scikit-learn
- pandas
- datasets
- matplotlib / seaborn
- pyvi (cho xử lý tiếng Việt)

### Quy trình tổng quát
1. Thu thập và tiền xử lý dữ liệu
2. Chuyển đổi dữ liệu thành tập train/validation/test
3. Tạo token và biểu diễn đặc trưng cho từng mô hình
4. Huấn luyện mô hình và đánh giá kết quả
5. Xuất báo cáo, confusion matrix và kết quả thống kê

## Cấu trúc thư mục

```text
scripts/
├── BERT/               # Tokenize, train, evaluate cho BERT
├── LTSM/               # Tiền xử lý, train, evaluate cho LSTM
├── PhoBERT/            # Tokenize, train, evaluate cho PhoBERT
├── RoBERTa/            # Tokenize, train, evaluate cho RoBERTa
├── preprocessing/      # Tiền xử lý dữ liệu tiếng Anh và tiếng Việt
├── PullData/           # Thu thập dữ liệu
├── config.py           # Cấu hình chung
├── check_cuda.py       # Kiểm tra GPU/CUDA
├── translate_deep.py   # Dịch dữ liệu sang tiếng Việt
├── requirements.txt    # Các dependency cần cài
└── README.md           # Tài liệu hướng dẫn
```

## Yêu cầu môi trường

- Python 3.10+
- GPU hỗ trợ CUDA (khuyến nghị để tăng tốc huấn luyện)
- Windows hoặc môi trường Linux/macOS tương thích

### Cài đặt môi trường

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Kiểm tra GPU

```bash
python check_cuda.py
```

## Cấu hình chính

File [config.py](config.py) chứa các tham số quan trọng như:

- batch size
- learning rate
- số epoch
- số lớp nhãn
- độ dài chuỗi tối đa
- đường dẫn lưu dữ liệu và mô hình

Lưu ý:

- Tiếng Anh: max_length = 512
- Tiếng Việt: max_length = 256

## Quy trình thực hiện

### 1. Tiền xử lý dữ liệu

Chạy các script tiền xử lý tương ứng:

```bash
# Dữ liệu tiếng Anh
python preprocessing/english/preprocessing.py

# Dữ liệu tiếng Việt
python preprocessing/vietnamese/preprocessing_final.py
python preprocessing/vietnamese/preprocessing_15k.py
python preprocessing/vietnamese/preprocessing_pullpush.py
```

Nếu cần dịch dữ liệu sang tiếng Việt, chạy:

```bash
python translate_deep.py
```

### 2. Tạo token và đặc trưng

```bash
python BERT/bert_tokenize.py
python RoBERTa/roberta_tokenize.py
python PhoBERT/phobert_tokenize.py
python LTSM/lstm_preprocess.py
```

### 3. Huấn luyện mô hình

```bash
python BERT/train_bert.py
python RoBERTa/train_roberta.py
python PhoBERT/train_phobert.py
python LTSM/train_lstm.py
```

### 4. Đánh giá mô hình

```bash
python BERT/evaluate_bert.py
python RoBERTa/evaluate_roberta.py
python PhoBERT/evaluate_phobert.py
python LTSM/evaluate_lstm.py
```

## Kết quả thực nghiệm

Sau khi thực hiện quy trình huấn luyện và đánh giá, hệ thống sẽ tạo ra các đầu ra sau:

- dữ liệu đã tiền xử lý tại thư mục data/processed
- mô hình huấn luyện tại thư mục models
- báo cáo đánh giá, confusion matrix và thống kê tại thư mục results
- các file metrics cho từng mô hình để so sánh hiệu năng

Kết quả thực nghiệm có thể được dùng để đối chiếu và chọn ra mô hình phù hợp nhất cho từng ngôn ngữ và từng loại dữ liệu.

### Kết quả hiện tại (Current Result Snapshot)

Dưới đây là hiệu năng của các mô hình trên tập kiểm thử (Test set). Do tính chất mất cân bằng dữ liệu y tế, hệ thống sử dụng Macro-F1 làm thước đo đánh giá cốt lõi.

| Mô hình | Ngôn ngữ | Nguồn dữ liệu | Độ chính xác (Accuracy) | Macro-F1 |
|---|---|---|---|---|
| LSTM | Tiếng Anh | Reddit | 0.9900 | 0.9800 |
| BERT | Tiếng Anh | Reddit | 0.9700 | 0.9600 |
| RoBERTa | Tiếng Anh | Reddit | 0.9700 | 0.9600 |
| PhoBERT | Tiếng Việt | Reddit (Dịch) + VMHQA | 0.8870 | 0.8840 |

> Lưu ý: Mặc dù LSTM đạt điểm số cao trên bề mặt, phân tích Confusion Matrix cho thấy các kiến trúc Transformer (RoBERTa, PhoBERT) vượt trội trong việc thấu hiểu các cấu trúc phụ đỉnh phức tạp và xử lý ranh giới giao thoa của các ca bệnh đồng mắc (comorbidity) như Trầm cảm và Lo âu.


