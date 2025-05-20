import numpy as np
import cv2
import random

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        """
        Khởi tạo phép biến đổi lật ngang ngẫu nhiên với xác suất p
        
        Parameters:
        -----------
        p : float, default=0.5
            Xác suất lật ngang hình ảnh
        """
        self.p = p
        
    def __call__(self, image):
        """
        Áp dụng lật ngang ngẫu nhiên lên hình ảnh
        
        Parameters:
        -----------
        image : array-like, shape (height, width, n_channels)
            Hình ảnh đầu vào
            
        Returns:
        --------
        transformed_image : array, shape (height, width, n_channels)
            Hình ảnh đã biến đổi
        """
        # Lật ngang hình ảnh với xác suất p
        if random.random() < self.p:
            return cv2.flip(image, 1)  # 1 là lật ngang
        
        return image

class RandomRotation:
    def __init__(self, degrees=10):
        """
        Khởi tạo phép biến đổi xoay ngẫu nhiên với góc tối đa
        
        Parameters:
        -----------
        degrees : float, default=10
            Góc xoay tối đa (độ)
        """
        self.degrees = degrees
        
    def __call__(self, image):
        """
        Áp dụng xoay ngẫu nhiên lên hình ảnh
        
        Parameters:
        -----------
        image : array-like, shape (height, width, n_channels)
            Hình ảnh đầu vào
            
        Returns:
        --------
        transformed_image : array, shape (height, width, n_channels)
            Hình ảnh đã biến đổi
        """
        # Chọn góc xoay ngẫu nhiên
        angle = random.uniform(-self.degrees, self.degrees)
        
        # Lấy kích thước hình ảnh
        height, width = image.shape[:2]
        
        # Tính toán ma trận xoay
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Áp dụng phép xoay
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated_image
