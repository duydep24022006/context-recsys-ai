# Kết quả Đánh giá & So sánh Mô hình (CLO1)

Bảng dưới đây tổng hợp kết quả thực nghiệm trên tập kiểm thử theo thời gian (Temporal Split 80/20, không rò rỉ dữ liệu) qua **3 random seeds (`42, 100, 2026`)**, báo cáo dưới dạng `mean ± std`.

| Mô hình | Context-Aware | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | ROC-AUC | Latency (ms/query) | Train Time (s) | Size (KB) |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **NCF_PyTorch_CPU** | Có | 0.0685 ± 0.0016 | 0.4859 ± 0.0098 | **0.3156 ± 0.0022** | 0.2394 ± 0.0033 | 0.8540 ± 0.0109 | 2.60 | 28.73 | 72.5 |
| **XGBoost_Context_Aware** | Có | 0.0702 ± 0.0002 | 0.5000 ± 0.0044 | **0.3043 ± 0.0018** | 0.2225 ± 0.0030 | 0.8935 ± 0.0001 | 17.50 | 2.00 | 760.5 |
| **LightGBM_Context_Aware** | Có | 0.0702 ± 0.0002 | 0.4953 ± 0.0012 | **0.3041 ± 0.0029** | 0.2236 ± 0.0034 | 0.8926 ± 0.0002 | 14.73 | 2.11 | 610.9 |
| **Collaborative_Filtering_MF** | Không | 0.0455 ± 0.0004 | 0.3373 ± 0.0027 | **0.1811 ± 0.0016** | 0.1215 ± 0.0014 | 0.6704 ± 0.0005 | 0.68 | 0.11 | 75.6 |
| **Popularity_Baseline** | Không | 0.0495 ± 0.0000 | 0.3645 ± 0.0000 | **0.1759 ± 0.0000** | 0.1059 ± 0.0000 | 0.8222 ± 0.0000 | 0.40 | 0.00 | 1.4 |
| **LightGBM_No_Context** | Không | 0.0443 ± 0.0005 | 0.3228 ± 0.0031 | **0.1663 ± 0.0009** | 0.1071 ± 0.0011 | 0.8124 ± 0.0005 | 13.81 | 1.65 | 609.2 |
| **Content_Based_Filtering** | Không | 0.0085 ± 0.0000 | 0.0422 ± 0.0000 | **0.0228 ± 0.0000** | 0.0125 ± 0.0000 | 0.5469 ± 0.0000 | 0.89 | 0.65 | 219.9 |

## Nghiên cứu Ablation: Giá trị của Ngữ cảnh (Context Features)

- **LightGBM (Có Context - `hour, is_weekend, is_mobile, time_slot, ctx_item_pop`)**: NDCG@10 = `0.3041`
- **LightGBM (Không Context - chỉ `user_id, product_id, category, price, user_cat_ratio`)**: NDCG@10 = `0.1663`
- **Mức cải thiện tương đối**: **+82.86%** trên chỉ số NDCG@10 khi tích hợp tín hiệu ngữ cảnh thời gian thực.
