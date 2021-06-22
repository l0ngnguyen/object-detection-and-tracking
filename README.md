# Phát hiện và theo dõi đối tượng
## Module chính main.py
### Vào file main.py sửa đường dẫn video_path đến video và chạy
### Thao tác với video trong lúc đang chạy:
- Nhấn space để dừng video
- Nhấn 'c' để tiếp tục chạy video
- Nhấn '+' để tăng tốc độ video
- Nhấn '-' để giảm tốc độ video
- Nhấn ESC để thoát
- Kéo thả các thanh trackbar để tùy chỉnh các tham số thresh hold của bài toán (threshold của các cái nào tự tìm hiểu :)))

## Module model_detection.py chứa các hàm, phương pháp phát hiện đối tượng bằng mô hình deeplearning
## Module tracker.py chứa 2 phương pháp tracking đối tượng
## Module tracking_multiple_object.py để tracking nhiều đối tượng bằng những hàm sẵn có của opencv
- Đối tượng không được phát hiện tự động và tự chọn bằng tay
- Nhấn 's' để bắt đâu lựa chọn selectionROI các đối tượng
- Nhấn 'r' để reset và tracking lại từ đầu
- Nhấn ESC để thoát
