import time
import json
import pickle
import socket
import argparse
import numpy as np
from tqdm import tqdm
import os

class DataStreamer:
    """
    Lớp DataStreamer để gửi dữ liệu từ server đến client qua kết nối TCP
    """
    def __init__(self, host="localhost", port=6100):
        """
        Khởi tạo DataStreamer với host và port
        
        Parameters:
        -----------
        host : str, default="localhost"
            Địa chỉ host của server
        port : int, default=6100
            Cổng kết nối
        """
        self.host = host
        self.port = port
        self.data = []
        self.labels = []
        
    def connect_tcp(self):
        """
        Tạo kết nối TCP và chờ client kết nối
        
        Returns:
        --------
        connection : socket
            Kết nối socket với client
        address : tuple
            Địa chỉ của client
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(1)
        print(f"Đang chờ kết nối trên cổng {self.port}...")
        connection, address = s.accept()
        print(f"Đã kết nối với {address}")
        return connection, address
    
    def data_generator(self, data_file, batch_size):
        """
        Tạo các batch dữ liệu từ file
        
        Parameters:
        -----------
        data_file : str
            Đường dẫn đến file dữ liệu
        batch_size : int
            Kích thước batch
            
        Returns:
        --------
        batches : list
            Danh sách các batch dữ liệu
        """
        batches = []
        
        # Đọc dữ liệu từ file
        with open(data_file, "rb") as batch_file:
            batch_data = pickle.load(batch_file, encoding='bytes')
            self.data.append(batch_data[b'data'])
            self.labels.extend(batch_data[b'labels'])
        
        # Chuyển đổi dữ liệu thành mảng numpy
        data = np.vstack(self.data)
        self.data = list(map(np.ndarray.tolist, data))
        
        # Chia dữ liệu thành các batch
        size_per_batch = (len(self.data) // batch_size) * batch_size
        for ix in range(0, size_per_batch, batch_size):
            image = self.data[ix:ix+batch_size]
            label = self.labels[ix:ix+batch_size]
            batches.append([image, label])
        
        # Cập nhật dữ liệu còn lại
        self.data = self.data[size_per_batch:]
        self.labels = self.labels[size_per_batch:]
        
        return batches
    
    def send_data(self, connection, data_files, batch_size, sleep_time=3, split="train"):
        """
        Gửi dữ liệu từ các file đến client
        
        Parameters:
        -----------
        connection : socket
            Kết nối socket với client
        data_files : list
            Danh sách các file dữ liệu
        batch_size : int
            Kích thước batch
        sleep_time : int, default=3
            Thời gian nghỉ giữa các lần gửi (giây)
        split : str, default="train"
            Phân chia dữ liệu ("train" hoặc "test")
        """
        # Xác định tổng số batch
        if split == "train":
            total_batch = 50000 / batch_size + 1
        else:
            total_batch = 10000 / batch_size + 1
        
        # Tạo thanh tiến trình
        pbar = tqdm(total=total_batch)
        data_sent = 0
        
        # Duyệt qua từng file dữ liệu
        for file in data_files:
            # Tạo các batch từ file
            batches = self.data_generator(file, batch_size)
            
            # Gửi từng batch
            for batch in batches:
                images, labels = batch
                
                # Chuyển đổi hình ảnh thành mảng numpy và làm phẳng
                images = np.array(images)
                images = images.reshape(images.shape[0], -1)
                batch_size, feature_size = images.shape
                images = images.tolist()
                
                # Tạo payload
                payload = dict()
                for batch_idx in range(batch_size):
                    payload[batch_idx] = dict()
                    for feature_idx in range(feature_size):
                        payload[batch_idx][f'feature-{feature_idx}'] = images[batch_idx][feature_idx]
                    payload[batch_idx]['label'] = labels[batch_idx]
                
                # Chuyển đổi payload thành chuỗi và gửi
                payload = (json.dumps(payload) + "\n").encode()
                
                try:
                    connection.send(payload)
                except BrokenPipeError:
                    print("Kết nối bị đóng hoặc kích thước batch quá lớn")
                    return
                except Exception as error_message:
                    print(f"Lỗi: {error_message}")
                    return
                
                # Cập nhật tiến trình
                data_sent += 1
                pbar.update(n=1)
                pbar.set_description(f"Lần: {data_sent} | Đã gửi: {batch_size} hình ảnh")
                
                # Nghỉ giữa các lần gửi
                time.sleep(sleep_time)
    
    def stream_dataset(self, connection, folder, batch_size, split="train"):
        """
        Stream bộ dữ liệu từ thư mục
        
        Parameters:
        -----------
        connection : socket
            Kết nối socket với client
        folder : str
            Đường dẫn đến thư mục dữ liệu
        batch_size : int
            Kích thước batch
        split : str, default="train"
            Phân chia dữ liệu ("train" hoặc "test")
        """
        # Xác định các file dữ liệu
        data_files = [
            os.path.join(folder, 'data_batch_1'),
            os.path.join(folder, 'data_batch_2'),
            os.path.join(folder, 'data_batch_3'),
            os.path.join(folder, 'data_batch_4'),
            os.path.join(folder, 'data_batch_5'),
            os.path.join(folder, 'test_batch'),
        ]
        
        # Chọn file dữ liệu dựa trên split
        if split == 'train':
            data_files = data_files[:-1]  # Tất cả trừ file test
        else:
            data_files = [data_files[-1]]  # Chỉ file test
        
        # Gửi dữ liệu
        self.send_data(connection, data_files, batch_size, sleep_time=3, split=split)


class DataReceiver:
    """
    Lớp DataReceiver để nhận dữ liệu từ server qua kết nối TCP
    """
    def __init__(self, host="localhost", port=6100):
        """
        Khởi tạo DataReceiver với host và port
        
        Parameters:
        -----------
        host : str, default="localhost"
            Địa chỉ host của server
        port : int, default=6100
            Cổng kết nối
        """
        self.host = host
        self.port = port
        
    def connect_to_server(self):
        """
        Kết nối đến server
        
        Returns:
        --------
        socket : socket
            Kết nối socket với server
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Đang kết nối đến {self.host}:{self.port}...")
        s.connect((self.host, self.port))
        print("Đã kết nối thành công!")
        return s
    
    def receive_data(self, connection, buffer_size=4096):
        """
        Nhận dữ liệu từ server
        
        Parameters:
        -----------
        connection : socket
            Kết nối socket với server
        buffer_size : int, default=4096
            Kích thước buffer
            
        Returns:
        --------
        data : dict
            Dữ liệu nhận được
        """
        try:
            # Nhận dữ liệu
            data = connection.recv(buffer_size)
            
            # Kiểm tra nếu không có dữ liệu
            if not data:
                return None
            
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
            
            return {'images': images, 'labels': labels}
            
        except Exception as e:
            print(f"Lỗi khi nhận dữ liệu: {e}")
            return None


def main():
    """
    Hàm chính để chạy module streaming
    """
    parser = argparse.ArgumentParser(description='Stream dữ liệu qua kết nối TCP')
    
    # Tham số chung
    parser.add_argument('--mode', '-m', choices=['server', 'client'], required=True,
                        help='Chế độ hoạt động (server hoặc client)')
    parser.add_argument('--host', default='localhost',
                        help='Địa chỉ host (mặc định: localhost)')
    parser.add_argument('--port', '-p', type=int, default=6100,
                        help='Cổng kết nối (mặc định: 6100)')
    
    # Tham số cho server
    parser.add_argument('--folder', '-f', help='Thư mục dữ liệu')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='Kích thước batch (mặc định: 32)')
    parser.add_argument('--endless', '-e', action='store_true',
                        help='Bật chế độ stream liên tục')
    parser.add_argument('--split', '-s', choices=['train', 'test'], default='train',
                        help='Phân chia dữ liệu (train hoặc test, mặc định: train)')
    parser.add_argument('--sleep', '-t', type=int, default=3,
                        help='Thời gian nghỉ giữa các lần gửi (giây, mặc định: 3)')
    
    args = parser.parse_args()
    
    if args.mode == 'server':
        # Kiểm tra tham số bắt buộc cho server
        if not args.folder:
            parser.error("--folder là bắt buộc khi chạy ở chế độ server")
        
        # Khởi tạo server
        streamer = DataStreamer(host=args.host, port=args.port)
        
        # Kết nối và stream dữ liệu
        connection, _ = streamer.connect_tcp()
        
        try:
            if args.endless:
                # Chế độ stream liên tục
                while True:
                    streamer.stream_dataset(connection, args.folder, args.batch_size, args.split)
            else:
                # Chế độ stream một lần
                streamer.stream_dataset(connection, args.folder, args.batch_size, args.split)
        finally:
            # Đóng kết nối
            connection.close()
            
    else:  # client
        # Khởi tạo client
        receiver = DataReceiver(host=args.host, port=args.port)
        
        # Kết nối đến server
        connection = receiver.connect_to_server()
        
        try:
            # Nhận dữ liệu liên tục
            print("Đang nhận dữ liệu...")
            while True:
                data = receiver.receive_data(connection)
                
                if data is None:
                    print("Kết nối đã đóng hoặc không có dữ liệu")
                    break
                
                # In thông tin dữ liệu nhận được
                print(f"Đã nhận {len(data['images'])} hình ảnh, hình dạng: {data['images'].shape}")
                print(f"Nhãn: {data['labels']}")
                
        except KeyboardInterrupt:
            print("Đã dừng nhận dữ liệu")
        finally:
            # Đóng kết nối
            connection.close()


if __name__ == '__main__':
    main()
