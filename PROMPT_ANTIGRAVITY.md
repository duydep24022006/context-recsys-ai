# PROMPT CHO ANTIGRAVITY: HOÀN THIỆN DỰ ÁN "CONTEXT-AWARE E-COMMERCE RECOMMENDER"

Bạn là kỹ sư MLOps/Full-stack senior. Hãy hoàn thiện toàn bộ dự án trong workspace hiện tại (repo `context-recsys-ai`) cho tới khi **chạy được end-to-end bằng một lệnh Docker Compose** và đáp ứng đầy đủ đề bài bên dưới. **Không cần viết báo cáo Word/PDF**, nhưng phải sinh ra các file kết quả tự động (bảng, biểu đồ, ảnh chụp) để người dùng tự đưa vào báo cáo.

Làm việc tự chủ: không hỏi lại trừ khi bị chặn hoàn toàn. Gặp mơ hồ thì chọn phương án hợp lý nhất, ghi lại quyết định trong `docs/DECISIONS.md`.

---

## 1. ĐỀ BÀI (nguyên văn, đây là tiêu chí chấm)

**Học phần:** Phát triển các hệ thống thông minh (INT14151), dự án cuối khóa.

**Yêu cầu chung:**
- Quy trình phát triển: áp dụng MLOps kết hợp CI/CD, Containerization (Docker), Monitoring, Model Registry.
- CLO1 (đánh giá mô hình): thử nghiệm, so sánh tối thiểu 3-4 thuật toán/mô hình khác nhau trước khi chọn phương án tối ưu.
- CLO2 (triển khai hệ thống): thiết kế kiến trúc hoàn chỉnh.

**Đề tài 1: Hệ thống Khuyến nghị Sản phẩm Thương mại Điện tử theo Ngữ cảnh người dùng**
- Nhóm dữ liệu chính: dữ liệu có cấu trúc (lịch sử giao dịch, tương tác người dùng, thuộc tính sản phẩm).
- Mô hình (CLO1): so sánh Collaborative Filtering (Matrix Factorization), Content-Based Filtering, LightGBM/XGBoost và Deep Learning (Neural Collaborative Filtering - NCF).
- Kiến trúc và quy trình (CLO2): quy trình MLOps. Tự động hóa pipeline dữ liệu với Apache Airflow / Prefect. Quản lý thử nghiệm mô hình với MLflow. Đóng gói backend FastAPI và frontend React/Vue.js bằng Docker containers.
- Sản phẩm bàn giao: mã nguồn MLOps pipeline, Web Dashboard giao dịch và gợi ý sản phẩm real-time, báo cáo kỹ thuật (người dùng tự làm, bạn chỉ cần sinh số liệu/biểu đồ/ảnh chụp).

---

## 2. HIỆN TRẠNG REPO (đã có, đừng viết lại từ đầu, hãy tái sử dụng)

Stack: Python 3.11, pandas, scikit-learn, LightGBM, PyTorch (CPU), MLflow, Prefect, FastAPI, Docker Compose.

| File | Vai trò |
|---|---|
| `src/prepare_data.py` | Sinh dữ liệu giả (500 user, 300 sản phẩm, 20.000 tương tác) vào `shared_data/*.parquet`, tạo feature, negative sampling 1:4, split 80/20 theo thời gian |
| `src/benchmark.py` | Huấn luyện/đánh giá 4 mô hình (Content-Based TF-IDF, MF bằng TruncatedSVD, LightGBM, NCF PyTorch), log lên MLflow, đăng ký model tốt nhất với alias `Production` |
| `src/metrics.py` | `evaluate_model`: Precision/Recall/NDCG@10, AUC, thời gian suy luận/query |
| `src/flow_retrain.py` | Prefect flow 3 task (validate + feature, retrain LightGBM, promote nếu NDCG >= 95% bản cũ), chạy `--serve` mỗi 24h |
| `src/recommender.py` | Nạp model `models:/ContextAwareRecommender@Production`, hàm `predict(user_id, device, hour, top_k)` |
| `src/api.py` | FastAPI: `/health`, `/recommend`, `/interactions/recent`, `/metrics`, phục vụ `frontend/index.html` |
| `frontend/index.html` | Dashboard Vue 3 một file (CDN) |
| `tests/test_api.py` | 4 test cho API (mock model) |
| `docker-compose.yml`, `Dockerfile` | Dịch vụ `mlflow-server`, `prefect-orchestrator`, `recsys-api`, `prometheus` |
| `.github/workflows/ci.yml` | lint + pytest + docker build |

**Data contract bắt buộc giữ nguyên** cho `interactions.parquet`: 5 cột `user_id, product_id, event_type (view|cart|purchase), device_type (mobile|desktop), timestamp`. Context features: `hour, is_weekend, is_mobile, time_slot`.

**Điểm yếu đã biết cần xử lý:**
1. Chưa từng chạy thử toàn bộ stack Docker + MLflow + API, nên chắc chắn còn lỗi tích hợp. Việc đầu tiên là chạy thật và sửa.
2. Các class `*Wrapper` định nghĩa ngay trong `benchmark.py` (chạy dưới dạng `__main__`), rủi ro lỗi khi MLflow load lại trong container API. Cần tách ra `src/models.py` và log với `code_paths`.
3. Dữ liệu là dữ liệu giả sinh theo đúng quy luật ngữ cảnh nên kết quả thiên về mô hình context-aware, cần dataset thật để thuyết phục.
4. NCF mới train 5 epoch, không tune; MF dùng TruncatedSVD; chưa có XGBoost; chưa có baseline; chưa lưu kết quả benchmark.
5. `requirements.txt` là `pip freeze` của cả máy (có `pywin32`), cần thay bằng file gọn hoặc xóa.
6. `src/api.py` dùng `@app.on_event` (deprecated), cần chuyển sang `lifespan`.
7. Chưa có monitoring ngoài Prometheus cơ bản, chưa có Grafana, chưa có theo dõi data drift.
8. Frontend mới là một file HTML qua CDN, chưa là container riêng.

---

## 3. CÁC VIỆC CẦN LÀM (theo thứ tự ưu tiên, commit sau mỗi phase)

### Phase 0: Chạy thật và vá lỗi tích hợp
- Chạy: `python src/prepare_data.py` → `docker compose up -d mlflow-server` → chạy `benchmark.py` trong container → `docker compose up -d`. Sửa mọi lỗi gặp phải (đường dẫn, biến môi trường, healthcheck, quyền ghi volume `mlflow_store`, thứ tự khởi động, `depends_on`).
- Tách wrapper models ra `src/models.py`; `benchmark.py`, `flow_retrain.py`, `recommender.py` đều import từ đó; log model bằng `mlflow.pyfunc.log_model(..., code_paths=["src/models.py"], signature=..., input_example=...)`. Đảm bảo API load được model `@Production` sau khi benchmark chạy xong và sau cả khi Prefect retrain.
- Thêm `Makefile` (hoặc `justfile`) với các target: `data`, `benchmark`, `up`, `down`, `test`, `lint`, `smoke`.
- Thêm `.env.example`; mọi URL/đường dẫn đọc từ biến môi trường.

### Phase 1: Dữ liệu (ưu tiên cao)
- Giữ generator giả làm chế độ `--source synthetic` (fallback, chạy offline được).
- Thêm chế độ `--source real`: tải một dataset thương mại điện tử thật công khai (ví dụ Retailrocket e-commerce dataset hoặc Olist Brazilian E-commerce, chọn cái tải được và có giấy phép cho phép dùng học thuật), ánh xạ về đúng data contract 5 cột + bảng `products` (`product_id, title, category, price`). Chọn con số vừa phải (ví dụ 2-5 nghìn user, 1-3 nghìn sản phẩm, top-N category) để benchmark chạy trong vài phút trên CPU. Nếu không tải được dataset thật, ghi rõ trong `docs/DECISIONS.md` và giữ synthetic nhưng giảm tính "rò rỉ quy luật" (thêm nhiễu, cold-start user, long-tail popularity).
- Thêm `docs/DATA.md`: mô tả nguồn, schema, thống kê (số user/item/tương tác, phân bố event/device/giờ), xuất 2-3 biểu đồ EDA vào `docs/figures/`.
- Giữ split theo thời gian, không rò rỉ dữ liệu (feature chỉ tính trên train).

### Phase 2: Đánh giá mô hình (CLO1)
- Có tối thiểu các mô hình: **Popularity baseline**, **Content-Based**, **Collaborative Filtering (MF)** (giữ SVD và có thể thêm ALS/`implicit` nếu cài được), **LightGBM**, **XGBoost**, **NCF (PyTorch)**. Tổng cộng ít nhất 4 mô hình chính + baseline.
- Tune hyperparameter hợp lý cho từng mô hình (Optuna hoặc grid nhỏ, giới hạn thời gian), NCF có early stopping và nhiều epoch hơn. Cố định seed; chạy tối thiểu 3 seed và báo cáo mean ± std.
- Chỉ số: Precision@10, Recall@10, NDCG@10, MAP@10 (thêm), AUC, thời gian suy luận/query (ms), thời gian huấn luyện, kích thước model.
- Ghi tất cả lên MLflow (params, metrics, artifacts, feature importance cho cây). Chọn model tốt nhất theo NDCG@10, đăng ký vào Model Registry với alias `Production`.
- Sinh tự động: `docs/results/benchmark_results.csv`, `docs/results/benchmark_results.md` (bảng markdown), và biểu đồ so sánh `docs/figures/model_comparison.png` (bar chart các metric), ảnh learning curve của NCF.
- Có thêm ablation ngắn: LightGBM có/không có context features, để chứng minh giá trị của "context-aware".

### Phase 3: Pipeline MLOps (Prefect + MLflow)
- Giữ Prefect (không cần Airflow). Flow gồm: ingest → validate data contract → feature engineering → (kiểm tra drift) → train → evaluate → so sánh với Production → promote/skip → gửi log. Mỗi task có retry và log rõ ràng.
- Quy tắc promote: model mới chỉ lên Production nếu NDCG@10 >= 95% bản cũ (giữ), kèm tag MLflow ghi lý do.
- Sau khi promote, API phải tự dùng model mới (endpoint `POST /admin/reload` và/hoặc API tự kiểm tra version định kỳ).
- Prefect chạy được trong Compose theo lịch (24h) và có thể trigger tay; có script `scripts/trigger_retrain.sh`.
- Mô phỏng dữ liệu mới: script `scripts/simulate_new_data.py` thêm tương tác mới (có thể kèm drift) vào `interactions.parquet` để demo retrain.

### Phase 4: Backend FastAPI
- Chuyển `on_event` sang `lifespan`. Thêm: CORS, logging có cấu trúc, `GET /model/info` (tên, version, alias, metrics của run), `POST /admin/reload`, `GET /products`, `GET /users/{id}/profile` (danh mục ưa thích).
- Tối ưu độ trễ: cache bảng feature trong bộ nhớ, tránh merge DataFrame cho cả 300+ sản phẩm mỗi request nếu có thể (vectorize/precompute). Mục tiêu p95 < 100 ms cho top-10 trên CPU; đo bằng script `scripts/load_test.py` (hoặc Locust) và lưu kết quả vào `docs/results/latency.md`.
- Cold-start: user chưa có lịch sử thì trả về popular theo ngữ cảnh, không lỗi 500.
- Prometheus metrics: số request theo device, histogram độ trễ, gauge version model Production, counter lỗi, gauge/histogram điểm số gợi ý.
- Test pytest đầy đủ cho API (validation, cold-start, reload), có test cho `metrics.py` và `prepare_data` (schema, không leakage).

### Phase 5: Frontend (React hoặc Vue, container riêng)
- Chuyển thành project **Vue 3 + Vite** (hoặc React + Vite) trong `frontend/`, build multi-stage Dockerfile, phục vụ bằng nginx, proxy `/api` tới backend. Service `frontend` trong Compose (cổng 8080 hoặc 3000).
- Dashboard gồm các tab: (1) **Gợi ý real-time**: chọn user/thiết bị/giờ/top-K, hiển thị danh sách kèm điểm, độ trễ, và so sánh gợi ý giữa hai ngữ cảnh (ví dụ mobile 20h vs desktop 10h) cạnh nhau; (2) **Giao dịch**: bảng tương tác gần đây tự làm mới, lọc theo user/event/device; (3) **Phân tích**: biểu đồ phân bố danh mục theo giờ/thiết bị, số request, độ trễ theo thời gian; (4) **Mô hình**: bảng so sánh mô hình (đọc từ MLflow/API), model đang Production, lần retrain gần nhất.
- Giao diện sạch, responsive, hỗ trợ dark mode, có trạng thái loading/lỗi/rỗng rõ ràng, tiếng Việt, truy cập bàn phím được.

### Phase 6: Monitoring
- Prometheus (giữ) + **Grafana** với dashboard được provision sẵn bằng file JSON/YAML trong `monitoring/grafana/` (request rate, latency p50/p95/p99, error rate, version model).
- **Data drift**: dùng Evidently AI (hoặc tự cài PSI/KS-test) để so sánh dữ liệu mới và dữ liệu train theo `hour, is_mobile, category`; sinh báo cáo HTML vào `docs/results/drift_report.html`; flow Prefect dùng kết quả drift để quyết định retrain.
- (Tùy chọn) alert rule Prometheus cho latency cao hoặc lỗi nhiều.

### Phase 7: CI/CD
- GitHub Actions: lint (ruff), test (pytest + coverage), build image backend và frontend, **chạy smoke test bằng `docker compose up`** (gọi `/health`, `/recommend`), và push image lên GHCR khi merge vào `main`. Cache pip/docker layer. Badge CI trong README.
- Thêm `pre-commit` (ruff, trailing whitespace). Tách `requirements.docker.txt` (runtime) và `requirements-dev.txt`; xóa hoặc thay `requirements.txt` freeze.

### Phase 8: Tài liệu kỹ thuật ngắn gọn (không phải báo cáo)
- `README.md`: giới thiệu, sơ đồ kiến trúc (Mermaid), cấu trúc thư mục, cách chạy một lệnh (`make up`), URL các dịch vụ, cách chạy benchmark/retrain, cách chạy test.
- `docs/ARCHITECTURE.md`: sơ đồ Mermaid luồng dữ liệu → pipeline → registry → API → frontend → monitoring, mô tả từng thành phần và lý do chọn công nghệ.
- `docs/DECISIONS.md`: các quyết định và giả định.
- `docs/screenshots/`: chụp ảnh dashboard frontend, MLflow (bảng runs, registry), Prefect (flow run), Grafana, nếu có công cụ trình duyệt thì chụp tự động.
- **Không viết báo cáo học thuật.** Người dùng sẽ tự viết, chỉ cần các file số liệu/biểu đồ/ảnh ở trên.

---

## 4. TIÊU CHÍ NGHIỆM THU (tự kiểm tra trước khi kết thúc)

1. `make up` (hoặc `docker compose up -d --build`) trên máy sạch dựng được toàn bộ: mlflow, prefect, api, frontend, prometheus, grafana.
2. Sau `make benchmark`, MLflow có ít nhất 4 mô hình chính + baseline cùng metrics, registry có `ContextAwareRecommender@Production`.
3. `curl -X POST localhost:8000/recommend -d '{"user_id":15,"device":"mobile","hour":20,"top_k":10}'` trả về 10 sản phẩm hợp lệ; p95 latency theo `docs/results/latency.md` ghi rõ.
4. Kết quả gợi ý **thay đổi hợp lý theo ngữ cảnh** (mobile 20h khác desktop 10h) và hiển thị được trên dashboard.
5. Trigger retrain (Prefect) → có run mới trên MLflow → promote/skip đúng quy tắc → API dùng model mới mà không cần restart container.
6. Báo cáo drift sinh được; Grafana hiển thị metric thật.
7. `pytest` pass, `ruff` sạch, CI xanh.
8. Các file kết quả có sẵn: `docs/results/benchmark_results.{csv,md}`, `docs/figures/*.png`, `docs/screenshots/*`.
9. Không commit secret, không commit thư mục `mlflow_store/`, `.venv/`, `__pycache__/`.

## 5. NGUYÊN TẮC LÀM VIỆC
- Bắt đầu bằng việc đọc toàn bộ repo và lập kế hoạch ngắn; làm tuần tự theo phase, mỗi phase một commit với message rõ ràng (Conventional Commits).
- Tái sử dụng code sẵn có; chỉ refactor khi cần. Giữ data contract và tên model đăng ký `ContextAwareRecommender`, alias `Production`.
- Mọi thứ phải chạy được trên CPU, không cần GPU, không cần dịch vụ trả phí.
- Sau mỗi phase, chạy lại test/smoke để chắc không làm hỏng phần trước.
- Kết thúc bằng bản tóm tắt: những gì đã làm, những gì chưa làm được và lý do, lệnh để chạy, vị trí các file kết quả.
