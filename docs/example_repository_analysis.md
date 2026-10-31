# Delhivery Delay Intelligence Pipeline — Reference Repository Analysis

> **Reference Repository:** [https://github.com/aayupunera8-cloud/delhivery-delay-pipeline](https://github.com/aayupunera8-cloud/delhivery-delay-pipeline)  
> **Purpose:** Detailed architectural, algorithmic, and data breakdown of the benchmark implementation for the IIT Roorkee Capstone Project.

---

## 1. Repository Layout & Structural Architecture

The reference repository strictly adheres to the required flat and modular directory structure:

```text
delhivery-delay-pipeline/
├── README.md                            # Comprehensive report answering all rubric questions
├── requirements.txt                     # Pinned dependencies (pandas, numpy, scikit-learn, joblib)
├── generate_data.py                     # Synthetic dataset generator (committed verbatim)
├── data/
│   ├── hubs.csv                         # 24 regional delivery hubs
│   ├── raw_shipments.csv                # 1,224 messy historical shipments
│   ├── new_shipments.csv                # 48 unlabelled parcels awaiting dispatch
│   └── modelling_table.csv              # 1,200 cleaned shipments with engineered features
├── src/
│   ├── __init__.py
│   └── shipment_prep.py                 # Core reusable preprocessing module
├── part_a_clean_eda.py                  # Part A pipeline & EDA metrics generation
├── part_b_regression.py                 # Part B delivery hours regression models
├── part_c_classification.py             # Part C delay prediction, feature selection & thresholding
├── part_d_segments.py                   # Part D K-Means hub segmentation & playbook allocation
├── package_model.py                     # Model serialization & artifact verification
├── predict_delay.py                     # Production batch scoring script
├── models/
│   ├── delay_model.pkl                  # Serialized scikit-learn classifier pipeline
│   ├── delay_threshold.csv              # Alert probability decision threshold (0.5)
│   ├── feature_order.csv                # Pinned 17 feature sequence
│   └── impute_values.csv                # Precomputed historical column medians
└── outputs/
    ├── part_a_checks.csv                # Data cleaning verification metrics
    ├── part_a_delay_by_group.csv        # Segmented delay statistics by dimension
    ├── part_a_correlations.csv          # Pearson correlation matrix of numeric columns
    ├── part_b_checks.csv                # Split sizes and leakage-prevention check
    ├── part_b_tree_depth.csv            # Decision tree depth diagnostic (1..12, unlimited)
    ├── part_b_metrics.csv               # Regression performance (MAE, MSE, RMSE, R2)
    ├── part_b_gb_learning_rate.csv      # Gradient boosting learning rate study (0.01, 0.1, 1.0)
    ├── part_b_cv.csv                    # 5-fold CV comparison between top candidates
    ├── part_c_models.csv                # Baseline and classifier metrics (default 0.5 threshold)
    ├── part_c_feature_scores.csv        # ANOVA F-scores and Random Forest importances
    ├── part_c_selectk.csv               # Validation F1 across k in [5, 7, 10, 12, 17]
    ├── part_c_tuning.csv                # Grid vs Randomized search vs Default configuration
    ├── part_c_thresholds.csv            # Precision-recall trade-off across thresholds [0.1..0.9]
    ├── part_c_final_config.csv          # Winning classifier parameters
    ├── part_c_final_test.csv            # Final model evaluation on held-out test set
    ├── hub_profile.csv                  # Aggregated operating profile for 24 hubs
    ├── part_d_separation.csv            # Across-over-within distance ratios for K in 2..5
    ├── hub_segments.csv                 # Cluster mapping for each hub
    ├── segment_profile.csv              # Mean metrics per segment (Standard, Watch, Escalate)
    ├── tier_vs_segment_mismatch.csv     # Detected discrepancies between assigned and actual tiers
    ├── part_d_reload_check.csv          # Pickle reload parity verification
    └── new_shipment_actions.csv         # Flagged status and operations action for 48 new parcels
```

---

## 2. Shared Pipeline Module: `src/shipment_prep.py`

This module serves as the single source of truth for preprocessing logic to prevent train-serving skew.

### Constants
- **`PROMISED_HOURS`**: Mapping service types to contractual delivery promises:
  - `Express`: 36 hours
  - `Standard`: 60 hours
  - `Economy`: 96 hours
- **`FEATURES`**: Exactly 17 pinned features in order:
  1. `promised_hours`, 2. `distance_km`, 3. `parcel_weight_kg`, 4. `log_declared_value`, 5. `n_stops`,
  6. `rain_flag`, 7. `sale_event_flag`, 8. `is_cod`, 9. `is_weekend`, 10. `dispatch_hour`,
  11. `hub_tier`, 12. `hub_daily_capacity`, 13. `label_print_batch`, 14. `scanner_build_no`,
  15. `tape_roll_no`, 16. `sticker_serial_mod`, 17. `shelf_slot_no`
- **`NOISE_COLUMNS`**: The 5 non-causal bookkeeping columns:
  `["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]`
- **`MEDIAN_COLUMNS`**: Columns requiring robust median imputation:
  `["distance_km", "parcel_weight_kg", "declared_value_inr"]`

### Core Functions
1. **`clean_shipments(raw, medians=None) -> (clean_df, medians)`**:
   - Drops exact duplicate rows (`raw.drop_duplicates()`).
   - Cleans `service_type` by stripping whitespace and converting to title case.
   - Replaces missing `payment_mode` with `"Unknown"`.
   - Cleans `declared_value` (removes `₹` and commas) into numeric `declared_value_inr`, dropping text.
   - Parses `dispatch_time` as datetime.
   - Replaces sentinel weights (`parcel_weight_kg > 100`) with `np.nan`.
   - Fills missing values in `MEDIAN_COLUMNS` using calculated or passed medians.
2. **`build_features(clean, hubs) -> enriched_df`**:
   - Performs a left merge with `hubs` on `hub_id`.
   - Derives `promised_hours` from `service_type`.
   - Derives binary `is_cod` (`1` if `COD`, else `0`).
   - Derives binary `is_weekend` (`1` if Saturday or Sunday, else `0`).
   - Extracts `dispatch_hour` (integer 6 to 21).
   - Computes `log_declared_value = np.log(declared_value_inr)`.

---

## 3. Data Processing & EDA: `part_a_clean_eda.py`

### Data Transformation & Checks (`outputs/part_a_checks.csv`)
- **Raw Data Ingestion:** 1,224 rows &rarr; 24 duplicates removed &rarr; 1,200 unique records.
- **Pre-Cleaning Missingness:**
  - `missing_distance_km_before`: 36
  - `missing_parcel_weight_before`: 24 (plus 5 sentinel values `> 100 kg`)
  - `missing_declared_value_before`: 12
  - `missing_payment_mode_before`: 18
  - `service_type_spellings`: reduced from 12 messy variations to 3 clean values.
- **Computed Historical Medians:**
  - `median_distance_km`: 314.95 km
  - `median_parcel_weight_kg`: 1.83 kg
  - `median_declared_value_inr`: INR 1,121.50
- **Integrity Validation:** 0 null values across all 17 features, `delivery_hours`, and `is_delayed`.
- **Target Metrics:** 278 delayed shipments out of 1,200 (Historical Delay Rate = **23.17%**).
- **Outlier Diagnostic:** Delivery hours IQR bounds are `[-0.71, 116.52]`, revealing only 1 high outlier.

### Delay Patterns by Group (`outputs/part_a_delay_by_group.csv`)
1. **Service Type:**
   - `Express`: 243 shipments, 119 delayed (**48.97%** delay rate) — tightest SLA (36 hrs) leaves no margin for error.
   - `Standard`: 642 shipments, 132 delayed (**20.56%** delay rate).
   - `Economy`: 315 shipments, 27 delayed (**8.57%** delay rate) — 96 hr SLA absorbs delays.
2. **Hub Tier:**
   - `Tier 1`: 400 shipments, 16 delayed (**4.00%**).
   - `Tier 2`: 400 shipments, 86 delayed (**21.50%**).
   - `Tier 3`: 400 shipments, 176 delayed (**44.00%**).
3. **Distance Bands:**
   - `0-200 km`: 274 shipments, 6 delayed (**2.19%**).
   - `201-400 km`: 563 shipments, 77 delayed (**13.68%**).
   - `401-600 km`: 275 shipments, 120 delayed (**43.64%**).
   - `601+ km`: 88 shipments, 75 delayed (**85.23%**).

### Correlations (`outputs/part_a_correlations.csv`)
Top linear correlations with `delivery_hours`:
- `promised_hours`: `+0.8372`
- `distance_km`: `+0.4013`
- `n_stops`: `+0.3247`
- `hub_daily_capacity`: `-0.3237`
- `hub_tier`: `+0.3175`

---

## 4. Regression Analysis: `part_b_regression.py`

Predicts expected delivery time in hours (`delivery_hours`) at the moment of dispatch.

### Split & Leakage Prevention
- **Split Ratio:** 70% Train (840) / 15% Val (180) / 15% Test (180) via nested `train_test_split(random_state=42)`.
- **Data Leakage Proof (`outputs/part_b_checks.csv`):**
  - Scaler fitted *strictly* on training rows.
  - `train_scaled_max_abs_mean` = `0.0000` (unrounded `~4e-16`).
  - `test_scaled_max_abs_mean` = `0.2187` (> 0.01), confirming test set was completely unseen.

### Diagnostic Experiments
1. **Decision Tree Depth Diagnostic (`outputs/part_b_tree_depth.csv`):**
   - Depth 1 (High Bias / Underfitting): Train RMSE = 11.63 hrs, Val RMSE = 12.13 hrs (Gap = 0.50 hrs).
   - Unlimited Depth (High Variance / Overfitting): Train RMSE = 0.00 hrs, Val RMSE = 7.84 hrs (Gap = 7.84 hrs).
   - **Optimal Depth:** Depth 6 yields lowest validation RMSE (6.28 hrs, Val R² = 0.8867).
2. **Gradient Boosting Learning Rate (`outputs/part_b_gb_learning_rate.csv`):**
   - `lr = 0.01`: Val RMSE = 9.99 hrs, Val R² = 0.7127 (too slow / underfits within 100 trees).
   - `lr = 0.1`: Val RMSE = 4.92 hrs, Val R² = 0.9306 (optimal convergence).
   - `lr = 1.0`: Val RMSE = 7.37 hrs, Val R² = 0.8440 (overshoots and chases training noise).

### Model Selection & Evaluation (`outputs/part_b_metrics.csv` & `outputs/part_b_cv.csv`)
- **Validation RMSE Ranking:**
  1. `linear_regression`: Val RMSE = 4.8912 hrs, R² = 0.9313
  2. `gradient_boosting`: Val RMSE = 4.9165 hrs, R² = 0.9306
  3. `random_forest`: Val RMSE = 5.5095 hrs, R² = 0.9128
  4. `decision_tree`: Val RMSE = 6.2796 hrs, R² = 0.8867
  5. `mean_baseline`: Val RMSE = 18.8552 hrs, R² = -0.0212
- **5-Fold Cross Validation on Train+Val (1,020 rows):**
  - `gradient_boosting`: Mean R² = **0.9353** (std = 0.0045)
  - `linear_regression`: Mean R² = **0.9261** (std = 0.0048)
  - *Architectural Insight:* Gradient Boosting captures non-linear feature interactions (such as sale window × Express and rain × long distance) better than Linear Regression across 5-fold evaluation.
- **Final Model on Test:** Linear Regression evaluated once on test set &rarr; RMSE = **4.5235 hrs**, MAE = **3.5671 hrs**, R² = **0.9493**.

---

## 5. Classification Pipeline: `part_c_classification.py`

Flags parcels at risk of missing their delivery SLA (`is_delayed = 1`).

### Stratified Split
- Train: 840 (195 delayed) | Val: 180 (42 delayed) | Test: 180 (41 delayed).

### Feature Signal Check (`outputs/part_c_feature_scores.csv`)
- **Signal Columns (Top 7 by F-Score):**
  1. `distance_km` (F-score = 369.34)
  2. `hub_tier` (F-score = 111.45)
  3. `hub_daily_capacity` (F-score = 80.52)
  4. `n_stops` (F-score = 75.31)
  5. `promised_hours` (F-score = 67.24)
  6. `sale_event_flag` (F-score = 56.42)
  7. `rain_flag` (F-score = 36.73)
- **Noise Columns:**
  - `label_print_batch`, `scanner_build_no`, `tape_roll_no`, `sticker_serial_mod`, `shelf_slot_no`.
  - All noise columns have F-scores `< 10.0` (ranks 8, 12, 14, 16, 17).
  - Despite zero causal relationship, noise features consumed **22.44%** of the untuned Random Forest's feature importance.
- **Selecting k (`outputs/part_c_selectk.csv`):**
  - `k = 5`: Val F1 = 0.6988
  - `k = 7`: Val F1 = **0.8333** (Peak signal-to-noise ratio)
  - `k = 10`: Val F1 = 0.7857
  - `k = 12`: Val F1 = 0.8049
  - `k = 17`: Val F1 = 0.7848

### Tuning Diagnostics (`outputs/part_c_tuning.csv`)
- `grid_search` (27 candidates, 135 fits): Val F1 = 0.7750, CV Acc = 0.8952
- `randomized_search` (9 candidates, 45 fits): Val F1 = 0.7848, CV Acc = 0.8905
- `default_forest_same_7_features` (100 trees, no depth limit, min_samples_leaf=1): Val F1 = **0.8333**
- *Takeaway:* Hyperparameter grid searches on small validation sets (180 samples) tend to overfit validation randomness. A default forest on well-selected features remains the most robust choice.

### Operational Threshold Selection (`outputs/part_c_thresholds.csv`)
Business requirement: Missed delays are more expensive than proactive customer calls &rarr; target validation recall $\ge 0.80$.
- Threshold 0.6: Recall = 0.7619 (fails 80% SLA)
- **Threshold 0.5**: Recall = **0.8333** (catches 35/42 delays), Precision = **0.7955**, F1 = **0.8140**
- Threshold 0.4: Recall = 0.8810, Precision = 0.7400
- **Chosen Alert Threshold:** `0.5`

### Test Set Performance (`outputs/part_c_final_test.csv`)
Scored once at threshold 0.5:
- **Accuracy:** 86.67%
- **Precision:** 68.89% (approx. 7 out of 10 flagged parcels are truly delayed)
- **Recall:** 75.61% (catches 31 of 41 delayed parcels)
- **F1 Score:** 0.7209

---

## 6. Hub Clustering & Operations Playbook: `part_d_segments.py`

Segments the 24 logistics hubs into data-driven operating tiers using K-Means clustering.

### Hub Profile Features (`outputs/hub_profile.csv`)
4 normalized operational metrics across all 24 hubs (sorted by `hub_id`):
1. `mean_delivery_hours`
2. `delay_rate`
3. `mean_distance_km`
4. `mean_n_stops`

### Separation Diagnostics (`outputs/part_d_separation.csv`)
Measured via across-to-within Euclidean distance ratio:
- $K = 2$: Ratio = 2.8940, Sizes = `15-9`
- **$K = 3$**: Ratio = **5.3928**, Sizes = `9-8-7` (Largest separation jump: $+2.4988$)
- $K = 4$: Ratio = 6.3047, Sizes = `8-7-5-4` (Contains cluster `< 6` hubs)
- $K = 5$: Ratio = 6.8866, Sizes = `8-7-4-3-2` (Contains cluster `< 6` hubs)
- *Business Decision:* $K=3$ is chosen because subsequent clusters create micro-groups too small for dedicated operations playbooks, while $K=3$ yields balanced groups aligned with existing tier structures.

### Segment Profiles (`outputs/segment_profile.csv`)
Clusters ordered by mean delay rate:
1. **Standard (7 hubs):** Delay Rate = **0.0257**, Mean Hours = 47.24, Mean Dist = 188.93 km, Mean Stops = 55.37.
2. **Watch (8 hubs):** Delay Rate = **0.1550**, Mean Hours = 56.81, Mean Dist = 314.18 km, Mean Stops = 70.72.
3. **Escalate (9 hubs):** Delay Rate = **0.4600**, Mean Hours = 64.65, Mean Dist = 470.48 km, Mean Stops = 84.34.

### Tier Label Mismatches (`outputs/tier_vs_segment_mismatch.csv`)
Identifies four hubs operating contrary to their administrative tier label:
| Hub ID | City | Stated Tier | Operational Segment | Operations Impact |
| :--- | :--- | :--- | :--- | :--- |
| **HUB07** | Kolkata | Tier 1 | **Watch** | Overstates reliability; requires active oversight |
| **HUB13** | Nagpur | Tier 2 | **Escalate** | Severe operational strain; needs hub manager escalation |
| **HUB15** | Kochi | Tier 2 | **Escalate** | Stretched lane performance; needs escalation |
| **HUB22** | Dehradun | Tier 3 | **Watch** | Label is too pessimistic; operates above Tier 3 norms |

---

## 7. Packaging & Operational Scoring: `package_model.py` & `predict_delay.py`

### Production Model Artifacts (`models/`)
- `delay_model.pkl`: Pipeline containing `SelectKBest(k=7)` and `RandomForestClassifier`.
- `delay_threshold.csv`: Contains scalar threshold `0.5`.
- `feature_order.csv`: Pinned 17 feature sequence.
- `impute_values.csv`: Exact precomputed historical medians.
- **Reload Parity Check (`outputs/part_d_reload_check.csv`):**
  - `reload_max_abs_probability_diff`: `0.0` (difference $\le 10^{-9}$), proving exact mathematical reproducibility upon reload.

### Batch Scoring Rules on New Parcels (`outputs/new_shipment_actions.csv`)
Applied to 48 unlabelled incoming parcels in `data/new_shipments.csv`:
$$\text{Action} = \begin{cases} 
\text{Escalate to hub manager} & \text{if } \text{flagged} = 1 \text{ and } \text{segment} = \text{Escalate} \\ 
\text{Proactive customer call} & \text{if } \text{flagged} = 1 \text{ and } \text{segment} \neq \text{Escalate} \\ 
\text{No action} & \text{if } \text{flagged} = 0 
\end{cases}$$

**Scoring Results on 48 Incoming Parcels:**
- **No action:** 40 parcels (83.3%)
- **Escalate to hub manager:** 6 parcels (12.5%)
- **Proactive customer call:** 2 parcels (4.2%)
- Total flagged parcels: 8 out of 48.

