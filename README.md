# Chương trình phân loại hình ảnh

Đây là một chương trình phân loại hình ảnh được xây dựng dựa trên thiết kế module hóa, hỗ trợ nhiều loại mô hình và phép biến đổi dữ liệu. Chương trình này có thể được sử dụng để huấn luyện, đánh giá và dự đoán nhãn cho hình ảnh.

## Cấu trúc thư mục

```
image_classifier/
├── data/                  # Thư mục chứa dữ liệu
├── models/                # Các mô hình phân loại
│   ├── __init__.py
│   ├── kmeans_model.py    # Mô hình K-means
│   └── svm_model.py       # Mô hình SVM
├── transforms/            # Các phép biến đổi dữ liệu
│   ├── __init__.py
│   ├── normalize.py       # Chuẩn hóa dữ liệu
│   ├── augment.py         # Tăng cường dữ liệu
│   └── transforms.py      # Lớp Transforms chính
├── utils/                 # Các tiện ích
│   ├── __init__.py
│   ├── dataloader.py      # Xử lý dữ liệu
│   └── visualization.py   # Trực quan hóa kết quả
├── main.py                # Điểm khởi đầu chương trình
├── trainer.py             # Huấn luyện mô hình
└── stream.py              # Module streaming dữ liệu
```

## Yêu cầu

Chương trình yêu cầu các thư viện Python sau:

- Python 3.6+
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- OpenCV (cv2)
- tqdm

Bạn có thể cài đặt các thư viện này bằng lệnh:

```bash
pip install scikit-learn numpy matplotlib seaborn tqdm opencv-python
```

## Dữ liệu

Chương trình được thiết kế để hoạt động với bộ dữ liệu CIFAR-10, bạn có thể tải xuống từ:
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Sau khi tải xuống, giải nén vào thư mục `data/`:

```bash
tar -xzf cifar-10-python.tar.gz -C ./data/
```

## Cách sử dụng

### Huấn luyện mô hình

```bash
python main.py --train --model_type svm --data_dir ./data/cifar-10-batches-py --save_model
```

Các tham số:
- `--train`: Chế độ huấn luyện
- `--model_type`: Loại mô hình (`svm` hoặc `kmeans`)
- `--data_dir`: Đường dẫn đến thư mục dữ liệu
- `--save_model`: Lưu mô hình sau khi huấn luyện

Tham số cho SVM:
- `--kernel`: Loại kernel (`linear`, `poly`, `rbf`, `sigmoid`)
- `--C`: Tham số C

Tham số cho KMeans:
- `--n_clusters`: Số lượng cụm

### Đánh giá mô hình

```bash
python main.py --evaluate --model_type svm --data_dir ./data/cifar-10-batches-py --load_model svm_model.pkl
```

Các tham số:
- `--evaluate`: Chế độ đánh giá
- `--load_model`: Tải mô hình đã lưu

### Dự đoán nhãn cho hình ảnh

```bash
python main.py --predict --model_type svm --load_model svm_model.pkl --image_path path/to/image.jpg
```

Các tham số:
- `--predict`: Chế độ dự đoán
- `--image_path`: Đường dẫn đến hình ảnh cần dự đoán

### Sử dụng với dữ liệu từ stream

Chương trình hỗ trợ nhận dữ liệu trực tiếp từ stream server:

```bash
# Khởi động stream server trước
python stream.py --mode server --folder ./data/cifar-10-batches-py --batch-size 32

# Sau đó chạy chương trình chính với chế độ stream
python main.py --train --model_type svm --stream --stream_host localhost --stream_port 6100
```

Các tham số stream:
- `--stream`: Bật chế độ nhận dữ liệu từ stream
- `--stream_host`: Địa chỉ host của stream server (mặc định: localhost)
- `--stream_port`: Cổng kết nối của stream server (mặc định: 6100)

### Streaming dữ liệu

#### Chạy server

```bash
python stream.py --mode server --folder ./data/cifar-10-batches-py --batch-size 32 --split train
```

Các tham số:
- `--mode`: Chế độ hoạt động (`server` hoặc `client`)
- `--host`: Địa chỉ host (mặc định: localhost)
- `--port`: Cổng kết nối (mặc định: 6100)
- `--folder`: Thư mục dữ liệu
- `--batch-size`: Kích thước batch (mặc định: 32)
- `--endless`: Bật chế độ stream liên tục
- `--split`: Phân chia dữ liệu (`train` hoặc `test`, mặc định: train)
- `--sleep`: Thời gian nghỉ giữa các lần gửi (giây, mặc định: 3)

#### Chạy client

```bash
python stream.py --mode client
```

## Ví dụ

### Huấn luyện mô hình SVM

```bash
python main.py --train --model_type svm --kernel linear --C 1.0 --data_dir ./data/cifar-10-batches-py --save_model
```

### Huấn luyện mô hình KMeans

```bash
python main.py --train --model_type kmeans --n_clusters 10 --data_dir ./data/cifar-10-batches-py --save_model
```

### Đánh giá mô hình

```bash
python main.py --evaluate --model_type svm --data_dir ./data/cifar-10-batches-py --load_model svm_model.pkl
```

### Huấn luyện với dữ liệu từ stream

```bash
# Terminal 1: Khởi động stream server
python stream.py --mode server --folder ./data/cifar-10-batches-py --batch-size 32

# Terminal 2: Chạy huấn luyện với dữ liệu từ stream
python main.py --train --model_type svm --stream
```

## Mở rộng

Chương trình được thiết kế theo cấu trúc module hóa, dễ dàng mở rộng:

1. Thêm mô hình mới:
   - Tạo file mới trong thư mục `models/`
   - Cập nhật `models/__init__.py`

2. Thêm phép biến đổi mới:
   - Tạo file mới trong thư mục `transforms/`
   - Cập nhật `transforms/__init__.py`

3. Thêm tiện ích mới:
   - Tạo file mới trong thư mục `utils/`
   - Cập nhật `utils/__init__.py`

## Giấy phép

Mã nguồn này được phân phối theo giấy phép MIT.
