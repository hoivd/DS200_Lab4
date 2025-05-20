import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class KMeansModel:
    def __init__(self, n_clusters=10, random_state=0):
        """
        Khởi tạo mô hình K-means với số cụm và trạng thái ngẫu nhiên
        
        Parameters:
        -----------
        n_clusters : int, default=10
            Số lượng cụm (clusters)
        random_state : int, default=0
            Trạng thái ngẫu nhiên để tái tạo kết quả
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=256,
            init_size=1024,
            reassignment_ratio=0.01
        )
        self.pca = PCA(n_components=2)
        self.cluster_to_label = {}
        
    def train(self, X, y=None):
        """
        Huấn luyện mô hình với dữ liệu X và nhãn y (nếu có)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu huấn luyện
        y : array-like, shape (n_samples,), optional
            Nhãn thực tế của dữ liệu
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Nhãn dự đoán
        metrics : dict
            Các độ đo đánh giá (accuracy, precision, recall, f1)
        """
        # Huấn luyện mô hình
        self.model.fit(X)
        
        # Dự đoán cụm
        cluster_predictions = self.model.predict(X)
        
        metrics = {}
        predictions = cluster_predictions
        
        # Nếu có nhãn thực tế, ánh xạ cụm với nhãn
        if y is not None:
            self._map_clusters_to_labels(cluster_predictions, y)
            predictions = self._get_predicted_labels(cluster_predictions)
            
            # Tính toán các độ đo
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
        # Dự đoán cụm
        cluster_predictions = self.model.predict(X)
        
        # Ánh xạ cụm với nhãn
        predictions = self._get_predicted_labels(cluster_predictions)
        
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
        Trực quan hóa kết quả phân cụm
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dữ liệu cần trực quan hóa
        y : array-like, shape (n_samples,), optional
            Nhãn thực tế của dữ liệu
        predictions : array-like, shape (n_samples,), optional
            Nhãn dự đoán của dữ liệu
        """
        # Giảm chiều dữ liệu xuống 2D để trực quan hóa
        X_pca = self.pca.fit_transform(X)
        
        plt.figure(figsize=(12, 10))
        
        # Vẽ biểu đồ phân cụm
        plt.subplot(2, 1, 1)
        if y is not None:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
            plt.title('Phân bố dữ liệu thực tế')
        else:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
            plt.title('Phân bố dữ liệu')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        
        # Vẽ biểu đồ dự đoán
        plt.subplot(2, 1, 2)
        if predictions is not None:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=predictions, cmap='viridis', alpha=0.5)
            plt.title('Phân cụm dự đoán')
        else:
            cluster_predictions = self.model.predict(X)
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_predictions, cmap='viridis', alpha=0.5)
            plt.title('Phân cụm K-means')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        
        plt.tight_layout()
        plt.show()
        
        # Vẽ ma trận nhầm lẫn nếu có nhãn thực tế và dự đoán
        if y is not None and predictions is not None:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Ma trận nhầm lẫn')
            plt.xlabel('Nhãn dự đoán')
            plt.ylabel('Nhãn thực tế')
            plt.show()
    
    def _map_clusters_to_labels(self, cluster_predictions, true_labels):
        """
        Ánh xạ cụm với nhãn thực tế
        
        Parameters:
        -----------
        cluster_predictions : array, shape (n_samples,)
            Dự đoán cụm
        true_labels : array, shape (n_samples,)
            Nhãn thực tế
        """
        # Tạo từ điển ánh xạ từ cụm sang nhãn
        cluster_label_map = {}
        
        # Duyệt qua từng cụm
        for cluster in range(self.n_clusters):
            # Lấy các nhãn thực tế của các điểm trong cụm
            mask = (cluster_predictions == cluster)
            if np.sum(mask) == 0:
                continue
                
            cluster_labels = true_labels[mask]
            
            # Nhãn phổ biến nhất trong cụm
            most_common_label = np.bincount(cluster_labels).argmax()
            
            # Ánh xạ cụm với nhãn phổ biến nhất
            cluster_label_map[cluster] = most_common_label
        
        self.cluster_to_label = cluster_label_map
    
    def _get_predicted_labels(self, cluster_predictions):
        """
        Chuyển đổi dự đoán cụm thành nhãn
        
        Parameters:
        -----------
        cluster_predictions : array, shape (n_samples,)
            Dự đoán cụm
            
        Returns:
        --------
        label_predictions : array, shape (n_samples,)
            Dự đoán nhãn
        """
        # Khởi tạo mảng nhãn dự đoán
        label_predictions = np.zeros_like(cluster_predictions)
        
        # Duyệt qua từng cụm
        for cluster, label in self.cluster_to_label.items():
            # Gán nhãn cho các điểm trong cụm
            label_predictions[cluster_predictions == cluster] = label
        
        return label_predictions
