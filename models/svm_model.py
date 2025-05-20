import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SVMModel:
    def __init__(self, kernel='linear', C=1.0, random_state=0):
        """
        Khởi tạo mô hình SVM với kernel, tham số C và trạng thái ngẫu nhiên
        
        Parameters:
        -----------
        kernel : str, default='linear'
            Loại kernel ('linear', 'poly', 'rbf', 'sigmoid')
        C : float, default=1.0
            Tham số điều chỉnh
        random_state : int, default=0
            Trạng thái ngẫu nhiên để tái tạo kết quả
        """
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
        self.model = SVC(
            kernel=kernel,
            C=C,
            random_state=random_state,
            decision_function_shape='ovr',
            probability=True
        )
        
    def train(self, X, y):
        """
        Huấn luyện mô hình với dữ liệu X và nhãn y
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu huấn luyện
        y : array-like, shape (n_samples,)
            Nhãn thực tế của dữ liệu
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Nhãn dự đoán
        metrics : dict
            Các độ đo đánh giá (accuracy, precision, recall, f1)
        """
        # Huấn luyện mô hình
        self.model.fit(X, y)
        
        # Dự đoán nhãn
        predictions = self.model.predict(X)
        
        # Tính toán các độ đo
        metrics = {}
        metrics['accuracy'] = accuracy_score(y, predictions)
        metrics['precision'] = precision_score(y, predictions, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y, predictions, average='macro', zero_division=0)
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-10)
        
        return predictions, metrics
    
    def predict(self, X, y=None):
        """
        Dự đoán nhãn cho dữ liệu mới
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu cần dự đoán
        y : array-like, shape (n_samples,), optional
            Nhãn thực tế của dữ liệu (nếu có)
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Nhãn dự đoán
        metrics : dict
            Các độ đo đánh giá (accuracy, precision, recall, f1, confusion_matrix)
        """
        # Dự đoán nhãn
        predictions = self.model.predict(X)
        
        metrics = {}
        
        # Nếu có nhãn thực tế, tính toán các độ đo
        if y is not None:
            metrics['accuracy'] = accuracy_score(y, predictions)
            metrics['precision'] = precision_score(y, predictions, average='macro', zero_division=0)
            metrics['recall'] = recall_score(y, predictions, average='macro', zero_division=0)
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-10)
            metrics['confusion_matrix'] = confusion_matrix(y, predictions)
        
        return predictions, metrics
    
    def visualize(self, X, y=None, predictions=None):
        """
        Trực quan hóa kết quả phân loại
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu cần trực quan hóa
        y : array-like, shape (n_samples,), optional
            Nhãn thực tế của dữ liệu
        predictions : array-like, shape (n_samples,), optional
            Nhãn dự đoán của dữ liệu
        """
        # Nếu không có nhãn dự đoán, tạo dự đoán mới
        if predictions is None and hasattr(self, 'model'):
            predictions = self.model.predict(X)
        
        # Vẽ ma trận nhầm lẫn nếu có nhãn thực tế và dự đoán
        if y is not None and predictions is not None:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Ma trận nhầm lẫn')
            plt.xlabel('Nhãn dự đoán')
            plt.ylabel('Nhãn thực tế')
            plt.show()
            
            # Tính toán và hiển thị các độ đo
            accuracy = accuracy_score(y, predictions)
            precision = precision_score(y, predictions, average='macro', zero_division=0)
            recall = recall_score(y, predictions, average='macro', zero_division=0)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
        
        # Nếu dữ liệu có 2 chiều, vẽ biên quyết định
        if X.shape[1] == 2:
            self._plot_decision_boundary(X, y, predictions)
    
    def _plot_decision_boundary(self, X, y=None, predictions=None):
        """
        Vẽ biên quyết định cho dữ liệu 2D
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            Dữ liệu 2D
        y : array-like, shape (n_samples,), optional
            Nhãn thực tế
        predictions : array-like, shape (n_samples,), optional
            Nhãn dự đoán
        """
        # Tạo lưới điểm để vẽ biên quyết định
        h = 0.02  # kích thước bước lưới
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Dự đoán nhãn cho lưới điểm
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(12, 5))
        
        # Vẽ biên quyết định và dữ liệu thực tế
        plt.subplot(1, 2, 1)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
        if y is not None:
            plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis')
        else:
            plt.scatter(X[:, 0], X[:, 1], edgecolors='k')
        plt.title('Biên quyết định và dữ liệu thực tế')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        # Vẽ biên quyết định và dự đoán
        plt.subplot(1, 2, 2)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
        if predictions is not None:
            plt.scatter(X[:, 0], X[:, 1], c=predictions, edgecolors='k', cmap='viridis')
        else:
            plt.scatter(X[:, 0], X[:, 1], edgecolors='k')
        plt.title('Biên quyết định và dự đoán')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        plt.tight_layout()
        plt.show()
