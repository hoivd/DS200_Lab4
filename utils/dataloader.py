import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm
import socket
import json
import time
import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.ml.linalg import DenseVector

from transforms import Transforms
from trainer import SparkConfig

class DataLoader:
    def __init__(self, data_dir=None, batch_size=32, transforms=None, shuffle=True, stream_mode=False, host="localhost", port=6100):
        """
        Khởi tạo với đường dẫn dữ liệu hoặc cấu hình stream, kích thước batch, phép biến đổi và shuffle
        
        Parameters:
        -----------
        data_dir : str, optional
            Đường dẫn đến thư mục chứa dữ liệu (None nếu sử dụng stream)
        batch_size : int, default=32
            Kích thước batch
        transforms : callable, optional
            Phép biến đổi áp dụng lên dữ liệu
        shuffle : bool, default=True
            Có xáo trộn dữ liệu hay không
        stream_mode : bool, default=False
            Có sử dụng chế độ stream hay không
        host : str, default="localhost"
            Địa chỉ host của server (chỉ dùng khi stream_mode=True)
        port : int, default=6100
            Cổng kết nối (chỉ dùng khi stream_mode=True)
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms
        self.shuffle = shuffle
        self.stream_mode = stream_mode
        self.host = host
        self.port = port
        self.data = []
        self.labels = []
        self.current_index = 0
        self.stream_connection = None
        
        # Nếu sử dụng chế độ stream, kết nối đến server
        if self.stream_mode:
            self._connect_to_stream()
        
    def _connect_to_stream(self):
        """
        Kết nối đến server stream
        """
        try:
            self.stream_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"Đang kết nối đến stream server tại {self.host}:{self.port}...")
            self.stream_connection.connect((self.host, self.port))
            print("Đã kết nối thành công đến stream server!")
        except Exception as e:
            print(f"Lỗi khi kết nối đến stream server: {e}")
            self.stream_mode = False
            print("Đã chuyển sang chế độ đọc file.")
        
    def load_data(self, split='train'):
        """
        Tải dữ liệu từ thư mục với phân chia train/test
        
        Parameters:
        -----------
        split : str, default='train'
            Phân chia dữ liệu ('train' hoặc 'test')
            
        Returns:
        --------
        X : array, shape (n_samples, height, width, n_channels)
            Dữ liệu hình ảnh
        y : array, shape (n_samples,)
            Nhãn
        """
        # Nếu đang ở chế độ stream, không cần tải dữ liệu từ file
        if self.stream_mode:
            print(f"Đang ở chế độ stream, dữ liệu sẽ được nhận trực tiếp từ server.")
            return np.array([]), np.array([])
            
        # Xác định các file dữ liệu dựa trên split
        if split == 'train':
            batch_files = [
                os.path.join(self.data_dir, f'data_batch_{i}')
                for i in range(1, 6)
            ]
        elif split == 'test':
            batch_files = [os.path.join(self.data_dir, 'test_batch')]
        else:
            raise ValueError("split phải là 'train' hoặc 'test'")
        
        # Tải dữ liệu từ các file
        data = []
        labels = []
        
        for file_path in tqdm(batch_files, desc=f"Đang tải dữ liệu {split}"):
            if not os.path.exists(file_path):
                print(f"Cảnh báo: File {file_path} không tồn tại")
                continue
                
            try:
                with open(file_path, 'rb') as f:
                    batch_data = pickle.load(f, encoding='bytes')
                    data.append(batch_data[b'data'])
                    labels.extend(batch_data[b'labels'])
            except Exception as e:
                print(f"Lỗi khi tải file {file_path}: {e}")
                continue
        
        # Chuyển đổi dữ liệu thành mảng numpy
        if data:
            X = np.vstack(data)
            X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Chuyển đổi sang (n_samples, height, width, channels)
            y = np.array(labels)
            
            # Lưu trữ dữ liệu
            self.data = X
            self.labels = y
            self.current_index = 0
            
            # Xáo trộn dữ liệu nếu cần
            if self.shuffle:
                self._shuffle_data()
            
            return X, y
        else:
            print("Không thể tải dữ liệu")
            return np.array([]), np.array([])
    
    def _receive_stream_data(self):
        """
        Nhận dữ liệu từ stream server
        
        Returns:
        --------
        batch_data : array, shape (batch_size, height, width, n_channels)
            Batch dữ liệu hình ảnh
        batch_labels : array, shape (batch_size,)
            Batch nhãn
        """
        if not self.stream_mode or self.stream_connection is None:
            print("Không ở chế độ stream hoặc chưa kết nối đến server")
            return np.array([]), np.array([])
        
        try:
            # Nhận dữ liệu
            buffer_size = 1024 * 1024  # 1MB buffer
            data = self.stream_connection.recv(buffer_size)
            
            # Kiểm tra nếu không có dữ liệu
            if not data:
                print("Không nhận được dữ liệu từ stream")
                return np.array([]), np.array([])
            
            # Giải mã dữ liệu
            data_str = data.decode('utf-8').strip()
            data_dict = json.loads(data_str)
            
            # Chuyển đổi dữ liệu thành mảng numpy
            images = []
            labels = []
            
            for idx in data_dict:
                # Lấy nhãn
                label = data_dict[idx]['label']
                labels.append(label)
                
                # Lấy đặc trưng
                features = []
                for feature_idx in range(3072):  # 32x32x3 = 3072
                    feature_key = f'feature-{feature_idx}'
                    if feature_key in data_dict[idx]:
                        features.append(data_dict[idx][feature_key])
                
                images.append(features)
            
            # Chuyển đổi thành mảng numpy
            images = np.array(images)
            labels = np.array(labels)
            
            # Reshape lại hình ảnh
            images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            
            return images, labels
            
        except Exception as e:
            print(f"Lỗi khi nhận dữ liệu từ stream: {e}")
            return np.array([]), np.array([])
    
    def get_batch(self):
        """
        Tạo batch dữ liệu tiếp theo
        
        Returns:
        --------
        batch_data : array, shape (batch_size, height, width, n_channels)
            Batch dữ liệu hình ảnh
        batch_labels : array, shape (batch_size,)
            Batch nhãn
        """
        # Nếu đang ở chế độ stream, nhận dữ liệu từ stream
        if self.stream_mode:
            batch_data, batch_labels = self._receive_stream_data()
            
            # Áp dụng phép biến đổi nếu có
            if len(batch_data) > 0 and self.transforms:
                batch_data = np.array([self.transforms(img) for img in batch_data])
            
            return batch_data, batch_labels
        
        # Nếu không ở chế độ stream, lấy dữ liệu từ bộ nhớ
        # Kiểm tra xem đã tải dữ liệu chưa
        if len(self.data) == 0:
            print("Chưa tải dữ liệu. Hãy gọi phương thức load_data trước.")
            return np.array([]), np.array([])
        
        # Tính toán chỉ số bắt đầu và kết thúc của batch
        start_idx = self.current_index
        end_idx = min(start_idx + self.batch_size, len(self.data))
        
        # Lấy batch dữ liệu
        batch_data = self.data[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        # Cập nhật chỉ số hiện tại
        self.current_index = end_idx
        
        # Nếu đã đến cuối dữ liệu, reset chỉ số và xáo trộn dữ liệu nếu cần
        if self.current_index >= len(self.data):
            self.current_index = 0
            if self.shuffle:
                self._shuffle_data()
        
        # Áp dụng phép biến đổi nếu có
        if self.transforms:
            batch_data = np.array([self.transforms(img) for img in batch_data])
        
        return batch_data, batch_labels
    
    def _shuffle_data(self):
        """
        Xáo trộn dữ liệu
        """
        # Tạo chỉ số ngẫu nhiên
        indices = np.random.permutation(len(self.data))
        
        # Xáo trộn dữ liệu và nhãn
        self.data = self.data[indices]
        self.labels = self.labels[indices]
        
    def close(self):
        """
        Đóng kết nối stream nếu có
        """
        if self.stream_mode and self.stream_connection is not None:
            try:
                self.stream_connection.close()
                print("Đã đóng kết nối stream")
            except Exception as e:
                print(f"Lỗi khi đóng kết nối stream: {e}")
            finally:
                self.stream_connection = None
