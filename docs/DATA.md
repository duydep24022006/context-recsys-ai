# Tài liệu Dữ liệu & Khám phá Dữ liệu (EDA)

## 1. Nguồn dữ liệu (Data Sources)
Pipeline chuẩn bị dữ liệu ([src/prepare_data.py](file:///c:/Users/BaDuy/Downloads/recsys-additions/src/prepare_data.py)) hỗ trợ 2 chế độ:
- **`--source real`**: Tải bộ dữ liệu giao dịch thương mại điện tử công khai (UCI Online Retail / Olist E-Commerce Public Dataset từ kho lưu trữ học thuật công khai), trích xuất danh mục sản phẩm, tên sản phẩm, đơn giá thực tế và ánh xạ về đúng Data Contract 5 cột. Nếu môi trường chạy offline không có kết nối mạng ra ngoài, pipeline tự động fallback về bộ sinh dữ liệu thực tế (`--source synthetic`).
- **`--source synthetic` (Mặc định / Offline-ready)**: Sinh dữ liệu thương mại điện tử mô phỏng hành vi thực tế gồm **500 người dùng**, **300 sản phẩm** thuộc 5 danh mục (`Electronics`, `Fashion`, `Home`, `Accessories`, `Books`) và **20.000 tương tác** trong khung thời gian 60 ngày:
  - Phân phối độ phổ biến sản phẩm tuân theo luật **Zipf (Long-tail)** ($\alpha = 1.15$).
  - Bao gồm **6% Cold-start Users** (người dùng có rất ít hoặc chưa có lịch sử tương tác trong tập huấn luyện).
  - Tín hiệu ngữ cảnh thực tế kèm **20% nhiễu ngẫu nhiên (stochastic exploration noise)** và **25% sở thích cá nhân dài hạn (Dirichlet prior)** để tránh hiện tượng rò rỉ quy luật nhân tạo (rule leakage).

---

## 2. Data Contract & Schema (`shared_data/*.parquet`)

### 2.1. Bảng tương tác gốc: `interactions.parquet` (Bắt buộc giữ nguyên 5 cột)
| Cột | Kiểu dữ liệu | Miền giá trị / Mô tả |
|---|---|---|
| `user_id` | `int64` | Định danh người dùng (`1 .. 500`, `>500` cho kịch bản cold-start) |
| `product_id` | `int64` | Định danh sản phẩm (`1 .. 300`) |
| `event_type` | `string` | Loại hành vi: `view` (~70%), `cart` (~20%), `purchase` (~10%) |
| `device_type` | `string` | Thiết bị truy cập: `mobile` (~62%) hoặc `desktop` (~38%) |
| `timestamp` | `datetime64[ns]` | Thời điểm phát sinh tương tác (sắp xếp tăng dần theo thời gian) |

### 2.2. Bảng danh mục sản phẩm: `products.parquet`
| Cột | Kiểu dữ liệu | Mô tả |
|---|---|---|
| `product_id` | `int64` | Khóa chính sản phẩm (`1 .. 300`) |
| `title` | `string` | Tên hiển thị sản phẩm |
| `category` | `string` | Một trong 5 danh mục: `Electronics`, `Fashion`, `Home`, `Accessories`, `Books` |
| `price` | `float64` | Đơn giá sản phẩm (USD, từ `$10` đến `$950`) |

### 2.3. Đặc trưng ngữ cảnh & Bảng đặc trưng tổng hợp (Chỉ tính trên tập Train)
- **Context features**:
  - `hour` (`0..23`): Giờ trong ngày trích xuất từ `timestamp`.
  - `is_weekend` (`0|1`): `1` nếu Thứ Bảy / Chủ Nhật, ngược lại `0`.
  - `is_mobile` (`0|1`): `1` nếu `device_type == "mobile"`, ngược lại `0`.
  - `time_slot` (`0..3`): Khung giờ trong ngày (`0`: Đêm 0–5h, `1`: Sáng 6–11h, `2`: Chiều 12–17h, `3`: Tối 18–23h).
- **Aggregate features (Leak-free)**:
  - `feat_user_cat.parquet` (`user_id, category, user_cat_ratio`): Tỷ lệ tương tác của người dùng với từng ngành hàng, **chỉ tính trên 80% dữ liệu huấn luyện theo thời gian**.
  - `feat_ctx_item.parquet` (`is_mobile, time_slot, product_id, ctx_item_pop`): Độ phổ biến chuẩn hóa của sản phẩm trong từng tổ hợp ngữ cảnh `(is_mobile, time_slot)`, **chỉ tính trên tập huấn luyện**.

---

## 3. Chiến lược Chia Tập Dữ liệu & Negative Sampling
1. **Temporal Split (80/20)**:
   - Toàn bộ 20.000 tương tác dương (`label = 1`) được sắp xếp theo `timestamp` tăng dần.
   - 80% đầu tiên (16.000 tương tác) làm tập huấn luyện dương (`pos_train`), 20% cuối cùng (4.000 tương tác) làm tập kiểm thử dương (`pos_test`).
   - Mọi thống kê (`user_cat_ratio`, `ctx_item_pop`) được tính độc lập trên `pos_train` trước khi ghép sang `test.parquet` (kiểm chứng tự động bằng unit test `test_prepare_data_contract_and_no_leakage` trong [tests/test_pipeline.py](file:///c:/Users/BaDuy/Downloads/recsys-additions/tests/test_pipeline.py)).
2. **Negative Sampling (Tỷ lệ 1:4)**:
   - Với mỗi tương tác dương, lấy mẫu ngẫu nhiên 4 sản phẩm âm (`label = 0`) giữ nguyên ngữ cảnh `(user_id, hour, is_weekend, is_mobile, time_slot)`.

---

## 4. Biểu đồ Khám phá Dữ liệu (EDA Figures)
Các biểu đồ dưới đây được sinh tự động bởi `python src/prepare_data.py --eda`:

- **Phân bố Hành vi, Thiết bị và Lưu lượng theo Giờ**: [docs/figures/eda_distributions.png](file:///c:/Users/BaDuy/Downloads/recsys-additions/docs/figures/eda_distributions.png)
- **Tương quan Ngữ cảnh (Thiết bị × Khung giờ) và Danh mục Sản phẩm**: [docs/figures/eda_context_heatmap.png](file:///c:/Users/BaDuy/Downloads/recsys-additions/docs/figures/eda_context_heatmap.png)
- **Phân phối Long-Tail của Sản phẩm & Tần suất Người dùng (kèm Cold-Start)**: [docs/figures/eda_long_tail.png](file:///c:/Users/BaDuy/Downloads/recsys-additions/docs/figures/eda_long_tail.png)
