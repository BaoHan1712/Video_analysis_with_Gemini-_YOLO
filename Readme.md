Mô tả tổng quan

```mermaid
flowchart TD
    A[Bắt đầu] --> B[Đọc video input]
    B --> C[Xử lý từng frame]
    C --> D{Đủ frame_rate frames?}
    D -->|Có| E[Lưu frame để phân tích]
    D -->|Không| F[Tiếp tục xử lý]
    E --> G[YOLO nhận diện đối tượng]
    G --> H[Gemini tạo mô tả]
    H --> I[Hiển thị kết quả]
    F --> I
    I --> J{Còn frame?}
    J -->|Có| C
    J -->|Không| K[Kết thúc]
```

1. **Chi tiết các bước xử lý**

```mermaid
flowchart LR
    A[Load YOLO Model] --> B[Load Gemini Model]
    B --> C[Cấu hình video output]
    C --> D[Khởi tạo annotator]
```

2. **Xử lý frame**:
```mermaid
flowchart TD
    A[Đọc frame] --> B[Resize frame]
    B --> C[YOLO detect]
    C --> D[Vẽ bounding box]
    D --> E[Thêm background mờ]
    E --> F[Render text tiếng Việt]
    F --> G[Lưu frame]
```
3. **Tích hợp LLM**:
```mermaid
flowchart TD
    A[Nhận diện đối tượng] --> B[Tạo mô tả]
    B --> C[Hiển thị kết quả]
```
4. ** Kết quả**:
![result](data\img1.png)

Hệ thống được thiết kế theo hướng module, dễ mở rộng và tùy biến. Mọi thành phần đều có thể được điều chỉnh thông qua các tham số đầu vào.

Liense:[HanBao](https://github.com/BaoHan1712)