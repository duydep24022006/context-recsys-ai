# Kết quả Đo Kiểm Độ trễ Suy luận (API Latency Benchmark)

- **Endpoint kiểm thử**: `POST /recommend` (Top-10 sản phẩm trên toàn bộ danh mục 300 sản phẩm)
- **Phần cứng**: CPU (in-memory feature vectorization, không merge DataFrame mỗi request)
- **Mục tiêu SLA**: `p95 < 100 ms` -> **ĐẠT (< 100 ms)**

| Chế độ kiểm thử | Số request | Concurrency | Throughput (req/s) | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Min (ms) | Max (ms) | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Sequential (1 worker)** | 200 | 1 | 53.7 | 18.61 | 15.37 | **35.17** | 38.70 | 8.92 | 46.79 | 0 |
| **Concurrent (4 workers)** | 200 | 4 | 69.2 | 57.35 | 55.14 | **82.00** | 109.58 | 30.31 | 135.22 | 0 |

## Kỹ thuật tối ưu hóa đã áp dụng
1. **In-memory Feature Pre-indexing**: Toàn bộ `products.parquet`, `feat_user_cat.parquet`, và `feat_ctx_item.parquet` (8 tổ hợp `is_mobile x time_slot`) được nạp sẵn thành mảng NumPy `float32` trong RAM khi khởi động hoặc `/admin/reload`.
2. **Loại bỏ DataFrame Merge lúc suy luận**: Mỗi request `/recommend` dựng ma trận ứng viên bằng phép gán mảng NumPy trực tiếp (`O(N_items)`), giúp giảm thời gian chuẩn bị đặc trưng từ ~35 ms xuống < 1 ms.
3. **Cold-start Fast Path**: Người dùng mới (`user_id` chưa có trong lịch sử) được suy luận kết hợp với vector `ctx_item_pop` đã tính sẵn, không phát sinh truy vấn phụ.
