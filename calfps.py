import time
import torch
from experimental import attempt_load
from utils.general import non_max_suppression

# Tải mô hình YOLOv5
model = attempt_load('best.pt')

# Giả sử bạn có một danh sách các khung hình để xử lý
frames = [torch.randn(1, 3, 640, 640) for _ in range(100)]  # 100 khung hình mẫu

total_time = 0
num_frames = len(frames)

for frame in frames:
    start_time = time.time()
    
    # Thực hiện dự đoán
    with torch.no_grad():
        pred = model(frame)[0]
    
    # Áp dụng non-maximum suppression
    pred = non_max_suppression(pred)
    
    end_time = time.time()
    total_time += (end_time - start_time)

avg_time = total_time / num_frames
fps = 1 / avg_time

print(f"Thời gian trung bình cho mỗi khung hình: {avg_time:.4f} giây")
print(f"FPS: {fps:.2f}")