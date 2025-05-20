class Transforms:
    def __init__(self, transforms_list=None):
        """
        Khởi tạo với danh sách các phép biến đổi
        
        Parameters:
        -----------
        transforms_list : list, optional
            Danh sách các phép biến đổi cần áp dụng
        """
        self.transforms = transforms_list or []
        
    def __call__(self, image):
        """
        Áp dụng tuần tự các phép biến đổi lên hình ảnh
        
        Parameters:
        -----------
        image : array-like, shape (height, width, n_channels)
            Hình ảnh đầu vào
            
        Returns:
        --------
        transformed_image : array, shape (height, width, n_channels)
            Hình ảnh đã biến đổi
        """
        # Áp dụng tuần tự các phép biến đổi
        transformed_image = image
        for transform in self.transforms:
            transformed_image = transform(transformed_image)
        
        return transformed_image
    
    def add_transform(self, transform):
        """
        Thêm một phép biến đổi vào danh sách
        
        Parameters:
        -----------
        transform : callable
            Phép biến đổi cần thêm
        """
        self.transforms.append(transform)
        
    def remove_transform(self, index):
        """
        Xóa một phép biến đổi khỏi danh sách
        
        Parameters:
        -----------
        index : int
            Chỉ số của phép biến đổi cần xóa
        """
        if 0 <= index < len(self.transforms):
            del self.transforms[index]
