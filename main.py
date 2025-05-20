import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from models import KMeansModel, SVMModel
from transforms import Transforms, RandomHorizontalFlip, Normalize
from utils import DataLoader, plot_images, plot_confusion_matrix, plot_training_history
from trainer import Trainer

def parse_args():
    """
    Xử lý tham số dòng lệnh
    
    Returns:
    --------
    args : Namespace
        Các tham số dòng lệnh
    """
    parser = argparse.ArgumentParser(description='Chương trình phân loại hình ảnh')
    
    # Tham số chung
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Đường dẫn đến thư mục dữ liệu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Đường dẫn để lưu mô hình')
    parser.add_argument('--model_type', type=str, default='svm', choices=['svm', 'kmeans'],
                        help='Loại mô hình (svm hoặc kmeans)')
    
    # Tham số huấn luyện
    parser.add_argument('--train', action='store_true',
                        help='Huấn luyện mô hình')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Kích thước batch')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Số lượng epoch huấn luyện')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Tỷ lệ dữ liệu dùng cho validation')
    
    # Tham số đánh giá
    parser.add_argument('--evaluate', action='store_true',
                        help='Đánh giá mô hình')
    
    # Tham số dự đoán
    parser.add_argument('--predict', action='store_true',
                        help='Dự đoán nhãn cho hình ảnh')
    parser.add_argument('--image_path', type=str,
                        help='Đường dẫn đến hình ảnh cần dự đoán')
    
    # Tham số mô hình
    parser.add_argument('--load_model', type=str,
                        help='Tải mô hình đã lưu')
    parser.add_argument('--save_model', action='store_true',
                        help='Lưu mô hình sau khi huấn luyện')
    
    # Tham số SVM
    parser.add_argument('--kernel', type=str, default='linear',
                        help='Kernel cho SVM (linear, poly, rbf, sigmoid)')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Tham số C cho SVM')
    
    # Tham số KMeans
    parser.add_argument('--n_clusters', type=int, default=10,
                        help='Số lượng cụm cho KMeans')
    
    # Tham số stream
    parser.add_argument('--stream', action='store_true',
                        help='Sử dụng chế độ stream để nhận dữ liệu')
    parser.add_argument('--stream_host', type=str, default='localhost',
                        help='Địa chỉ host của stream server')
    parser.add_argument('--stream_port', type=int, default=6100,
                        help='Cổng kết nối của stream server')
    
    return parser.parse_args()

def main():
    """
    Điểm khởi đầu chương trình
    """
    # Xử lý tham số dòng lệnh
    args = parse_args()
    
    # Tạo thư mục lưu trữ nếu chưa tồn tại
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Khởi tạo transforms
    transforms = Transforms([
        RandomHorizontalFlip(p=0.5),
        Normalize(
            mean=np.array([0.4914, 0.4822, 0.4465]),
            std=np.array([0.2470, 0.2435, 0.2616])
        )
    ])
    
    # Khởi tạo data loader
    data_loader = DataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        transforms=transforms,
        shuffle=True,
        stream_mode=args.stream,
        host=args.stream_host,
        port=args.stream_port
    )
    
    # Khởi tạo mô hình
    if args.model_type == 'svm':
        model = SVMModel(kernel=args.kernel, C=args.C)
    else:  # kmeans
        model = KMeansModel(n_clusters=args.n_clusters)
    
    # Khởi tạo trainer
    trainer = Trainer(model, data_loader, save_dir=args.save_dir)
    
    # Tải mô hình nếu được chỉ định
    if args.load_model:
        trainer.load_model(args.load_model)
    
    # Huấn luyện mô hình nếu được chỉ định
    if args.train:
        print(f"Huấn luyện mô hình {args.model_type}...")
        history = trainer.train(epochs=args.epochs, validation_split=args.validation_split)
        
        # Vẽ lịch sử huấn luyện
        plot_training_history(history)
        
        # Lưu mô hình nếu được chỉ định
        if args.save_model:
            trainer.save_model()
    
    # Đánh giá mô hình nếu được chỉ định
    if args.evaluate:
        print("Đánh giá mô hình...")
        metrics = trainer.evaluate()
    
    # Dự đoán nhãn cho hình ảnh nếu được chỉ định
    if args.predict and args.image_path:
        try:
            # Đọc hình ảnh
            import cv2
            image = cv2.imread(args.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Tiền xử lý hình ảnh
            image = cv2.resize(image, (32, 32))
            image = image.astype(np.float32) / 255.0
            
            # Áp dụng transforms
            if transforms:
                image = transforms(image)
            
            # Dự đoán
            image_flat = image.reshape(1, -1)
            predictions, _ = model.predict(image_flat)
            
            # Hiển thị kết quả
            print(f"Dự đoán: {predictions[0]}")
            
            # Hiển thị hình ảnh
            plt.figure(figsize=(6, 6))
            plt.imshow(image)
            plt.title(f"Dự đoán: {predictions[0]}")
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"Lỗi khi dự đoán hình ảnh: {e}")
    
    # Đóng kết nối stream nếu có
    if args.stream:
        data_loader.close()

if __name__ == '__main__':
    main()
