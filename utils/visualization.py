import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_images(images, labels=None, predictions=None, n_images=5):
    """
    Vẽ một số hình ảnh với nhãn và dự đoán (nếu có)
    
    Parameters:
    -----------
    images : array-like, shape (n_samples, height, width, n_channels)
        Dữ liệu hình ảnh
    labels : array-like, shape (n_samples,), optional
        Nhãn thực tế của dữ liệu
    predictions : array-like, shape (n_samples,), optional
        Nhãn dự đoán của dữ liệu
    n_images : int, default=5
        Số lượng hình ảnh cần hiển thị
    """
    # Giới hạn số lượng hình ảnh
    n_images = min(n_images, len(images))
    
    # Tạo figure với kích thước phù hợp
    plt.figure(figsize=(15, 3 * n_images))
    
    # Hiển thị từng hình ảnh
    for i in range(n_images):
        plt.subplot(n_images, 1, i + 1)
        
        # Hiển thị hình ảnh
        if images[i].shape[-1] == 1:  # Hình ảnh grayscale
            plt.imshow(images[i].squeeze(), cmap='gray')
        else:  # Hình ảnh màu
            plt.imshow(images[i])
        
        # Tạo tiêu đề
        title = ""
        if labels is not None:
            title += f"Nhãn thực tế: {labels[i]}"
        if predictions is not None:
            title += f" | Dự đoán: {predictions[i]}"
        
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Vẽ ma trận nhầm lẫn
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        Nhãn thực tế
    y_pred : array-like, shape (n_samples,)
        Nhãn dự đoán
    class_names : list, optional
        Tên các lớp
    """
    # Tính toán ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)
    
    # Tạo figure với kích thước phù hợp
    plt.figure(figsize=(10, 8))
    
    # Vẽ heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    # Thiết lập tiêu đề và nhãn
    plt.title('Ma trận nhầm lẫn')
    plt.xlabel('Nhãn dự đoán')
    plt.ylabel('Nhãn thực tế')
    
    # Thiết lập tên lớp nếu có
    if class_names is not None:
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    Vẽ lịch sử huấn luyện (độ chính xác, mất mát)
    
    Parameters:
    -----------
    history : dict
        Từ điển chứa lịch sử huấn luyện với các khóa:
        - 'accuracy': list độ chính xác qua các epoch
        - 'loss': list mất mát qua các epoch
        - 'val_accuracy': list độ chính xác validation (nếu có)
        - 'val_loss': list mất mát validation (nếu có)
    """
    # Tạo figure với 2 subplot
    plt.figure(figsize=(12, 5))
    
    # Vẽ đồ thị độ chính xác
    plt.subplot(1, 2, 1)
    plt.plot(history.get('accuracy', []), label='Training')
    if 'val_accuracy' in history:
        plt.plot(history.get('val_accuracy', []), label='Validation')
    plt.title('Độ chính xác qua các epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Vẽ đồ thị mất mát
    plt.subplot(1, 2, 2)
    plt.plot(history.get('loss', []), label='Training')
    if 'val_loss' in history:
        plt.plot(history.get('val_loss', []), label='Validation')
    plt.title('Mất mát qua các epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
