import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, data_loader, save_dir='./checkpoints'):
        """
        Khởi tạo với mô hình, data loader và thư mục lưu trữ
        
        Parameters:
        -----------
        model : object
            Mô hình cần huấn luyện (KMeansModel hoặc SVMModel)
        data_loader : DataLoader
            Đối tượng DataLoader để tải dữ liệu
        save_dir : str, default='./checkpoints'
            Thư mục lưu trữ mô hình
        """
        self.model = model
        self.data_loader = data_loader
        self.save_dir = save_dir
        self.history = {'accuracy': [], 'loss': [], 'precision': [], 'recall': [], 'f1': []}
        
        # Tạo thư mục lưu trữ nếu chưa tồn tại
        os.makedirs(save_dir, exist_ok=True)
        
    def train(self, epochs=10, validation_split=0.2):
        """
        Huấn luyện mô hình với số epoch và tỷ lệ validation
        
        Parameters:
        -----------
        epochs : int, default=10
            Số lượng epoch huấn luyện
        validation_split : float, default=0.2
            Tỷ lệ dữ liệu dùng cho validation
            
        Returns:
        --------
        history : dict
            Lịch sử huấn luyện
        """
        print("Đang tải dữ liệu huấn luyện...")
        X, y = self.data_loader.load_data(split='train')
        
        if len(X) == 0:
            print("Không thể tải dữ liệu huấn luyện")
            return self.history
        
        # Chia dữ liệu thành tập huấn luyện và validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        print(f"Bắt đầu huấn luyện với {epochs} epochs...")
        
        # Huấn luyện qua các epoch
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Huấn luyện mô hình
            predictions, metrics = self.model.train(X_train, y_train)
            
            # Lưu lịch sử huấn luyện
            self.history['accuracy'].append(metrics.get('accuracy', 0))
            self.history['precision'].append(metrics.get('precision', 0))
            self.history['recall'].append(metrics.get('recall', 0))
            self.history['f1'].append(metrics.get('f1', 0))
            self.history['loss'].append(0)  # Không có loss trong mô hình này
            
            # In kết quả huấn luyện
            print(f"  - Train accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  - Train precision: {metrics.get('precision', 0):.4f}")
            print(f"  - Train recall: {metrics.get('recall', 0):.4f}")
            print(f"  - Train F1: {metrics.get('f1', 0):.4f}")
            
            # Đánh giá trên tập validation nếu có
            if X_val is not None and y_val is not None:
                val_predictions, val_metrics = self.model.predict(X_val, y_val)
                
                # In kết quả validation
                print(f"  - Val accuracy: {val_metrics.get('accuracy', 0):.4f}")
                print(f"  - Val precision: {val_metrics.get('precision', 0):.4f}")
                print(f"  - Val recall: {val_metrics.get('recall', 0):.4f}")
                print(f"  - Val F1: {val_metrics.get('f1', 0):.4f}")
        
        print("Huấn luyện hoàn tất")
        return self.history
        
    def evaluate(self, test_data=None):
        """
        Đánh giá mô hình trên dữ liệu test
        
        Parameters:
        -----------
        test_data : tuple (X_test, y_test), optional
            Dữ liệu test. Nếu None, sẽ tải dữ liệu test từ data_loader
            
        Returns:
        --------
        metrics : dict
            Các độ đo đánh giá
        """
        # Tải dữ liệu test nếu không được cung cấp
        if test_data is None:
            print("Đang tải dữ liệu test...")
            X_test, y_test = self.data_loader.load_data(split='test')
        else:
            X_test, y_test = test_data
        
        if len(X_test) == 0:
            print("Không thể tải dữ liệu test")
            return {}
        
        print("Đang đánh giá mô hình...")
        predictions, metrics = self.model.predict(X_test, y_test)
        
        # In kết quả đánh giá
        print(f"Test accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"Test precision: {metrics.get('precision', 0):.4f}")
        print(f"Test recall: {metrics.get('recall', 0):.4f}")
        print(f"Test F1: {metrics.get('f1', 0):.4f}")
        
        # Vẽ ma trận nhầm lẫn
        if 'confusion_matrix' in metrics:
            plt.figure(figsize=(10, 8))
            plt.imshow(metrics['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Ma trận nhầm lẫn')
            plt.colorbar()
            plt.xlabel('Nhãn dự đoán')
            plt.ylabel('Nhãn thực tế')
            plt.tight_layout()
            plt.show()
        
        return metrics
        
    def save_model(self, filename=None):
        """
        Lưu mô hình đã huấn luyện
        
        Parameters:
        -----------
        filename : str, optional
            Tên file để lưu mô hình. Nếu None, sẽ tạo tên file tự động
        """
        if filename is None:
            model_type = self.model.__class__.__name__
            filename = f"{model_type}_model.pkl"
        
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Đã lưu mô hình tại {filepath}")
        except Exception as e:
            print(f"Lỗi khi lưu mô hình: {e}")
        
    def load_model(self, filename):
        """
        Tải mô hình đã lưu
        
        Parameters:
        -----------
        filename : str
            Tên file chứa mô hình
            
        Returns:
        --------
        model : object
            Mô hình đã tải
        """
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Đã tải mô hình từ {filepath}")
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
        
        return self.model
