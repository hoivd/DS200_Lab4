import numpy as np
import cv2

class Normalize:
    def __init__(self, mean=None, std=None):
        """
        Khởi tạo phép biến đổi chuẩn hóa với giá trị trung bình và độ lệch chuẩn
        
        Parameters:
        -----------
        mean : array-like, shape (n_channels,), optional
            Giá trị trung bình cho mỗi kênh màu
        std : array-like, shape (n_channels,), optional
            Độ lệch chuẩn cho mỗi kênh màu
        """
        self.mean = mean
        self.std = std
        
    def __call__(self, image):
        """
        Áp dụng chuẩn hóa lên hình ảnh
        
        Parameters:
        -----------
        image : array-like, shape (height, width, n_channels)
            Hình ảnh đầu vào
            
        Returns:
        --------
        normalized_image : array, shape (height, width, n_channels)
            Hình ảnh đã chuẩn hóa
        """
        # Chuyển đổi hình ảnh sang float32 nếu cần
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Nếu không có giá trị trung bình và độ lệch chuẩn, tính toán từ hình ảnh
        if self.mean is None:
            self.mean = np.mean(image, axis=(0, 1))
        
        if self.std is None:
            self.std = np.std(image, axis=(0, 1))
        
        # Chuẩn hóa hình ảnh
        normalized_image = (image - self.mean) / (self.std + 1e-10)
        
        return normalized_image
    
    def fit(self, images):
        """
        Tính toán giá trị trung bình và độ lệch chuẩn từ tập dữ liệu
        
        Parameters:
        -----------
        images : array-like, shape (n_samples, height, width, n_channels)
            Tập dữ liệu hình ảnh
        """
        # Tính toán giá trị trung bình và độ lệch chuẩn
        self.mean = np.mean(images, axis=(0, 1, 2))
        self.std = np.std(images, axis=(0, 1, 2))
        
        return self
