# Delhivery Capstone — Class Walkthrough & Lecture Notes

> **Source Notebook:** [Google Colab — Delhivery Last-Mile Delay Intelligence Pipeline](https://colab.research.google.com/drive/1IFwyc-pJbcNH9VqvKahxLUneoxSNx71L?usp=sharing)  
> **Topic:** Data Generation, Ingestion, Exploration, Data Cleaning (`clean_shipments`), and Feature Engineering (`build_features`) for Part A.

---

## 1. Setup & Environment Pinning

In the session, dependencies and their versions were verified to guarantee reproducibility across development and grading environments:

```python
import numpy as np
import pandas as pd
import sklearn
import joblib

print("Pandas:", pd.__version__)        # e.g., 2.2.3
print("Numpy:", np.__version__)         # e.g., 2.1.3 / 1.26.x
print("Scikit-learn:", sklearn.__version__)  # e.g., 1.5.x / 1.6.x
print("Joblib:", joblib.__version__)    # e.g., 1.4.x / 1.6.x
```

### Requirements Reference
`requirements.txt` should pin these four core packages using exact versions (`==`):
```text
pandas==<version>
numpy==<version>
scikit-learn==<version>
joblib==<version>
```

---

## 2. Core Constants

The capstone enforces standard dictionary mappings and fixed feature column ordering:

```python
PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
]
```

---

## 3. Data Ingestion & Initial Inspection

The simulation generates three CSV datasets in the `data/` directory:
- `data/hubs.csv` (24 rows, 5 columns: `hub_id`, `hub_city`, `region`, `hub_tier`, `hub_daily_capacity`)
- `data/raw_shipments.csv` (1,224 rows, 17 columns: historical shipments with `delivery_hours`)
- `data/new_shipments.csv` (48 rows, 16 columns: unlabelled shipments to be scored)

### Exploration Observations from Class
1. **Duplicate Records:**
   ```python
   raw_ship_df.duplicated().sum()  # 24 duplicate rows (approx. 1.96%)
   len(raw_ship_df.drop_duplicates())  # 1,200 unique shipments
   ```
2. **Missing Values (`raw_ship_df.isna().sum()`):**
   - `payment_mode`: 18 missing values
   - `parcel_weight_kg`: 25 missing values (prior to sentinel filtering)
   - `distance_km`: 36 missing values
   - `declared_value`: 12 missing values

---

## 4. Data Cleaning Steps (`clean_shipments`)

The classroom implementation demonstrated the exact sequential transformations required for `clean_shipments(raw, medians=None)`:

### Step 1: Deduplication
Drop exact duplicates so that only the 1,200 unique shipments remain:
```python
shipments = raw_ship_df.drop_duplicates().copy()
```

### Step 2: Service Type Standardization
`service_type` contains mixed casing and whitespace (e.g. `'express'`, `'  Express '`, `'STANDARD'`):
```python
# Before: 12 distinct variations
shipments["service_type"] = shipments["service_type"].str.strip().str.title()
# After: exactly 3 variations ('Standard': 642 (53.5%), 'Economy': 315 (26.25%), 'Express': 243 (20.25%))
```

### Step 3: Payment Mode Imputation
Replace missing values in `payment_mode` with `"Unknown"`:
```python
shipments["payment_mode"] = shipments["payment_mode"].fillna("Unknown")
# Distribution: Prepaid (729), COD (453), Unknown (18)
```

### Step 4: Declared Value Cleaning
Strip currency symbol (`₹`) and commas, cast to float as `declared_value_inr`, and drop `declared_value`:
```python
shipments["declared_value_inr"] = (
    shipments["declared_value"]
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)
shipments = shipments.drop(columns=["declared_value"])
```

### Step 5: Datetime Conversion
Parse `dispatch_time` string into datetime objects:
```python
shipments["dispatch_time"] = pd.to_datetime(shipments["dispatch_time"])
```

### Step 6: Scanner Sentinel Handling
Parcels with weights > 100 kg indicate scanner hardware faults (the network only accepts parcels up to 40 kg):
```python
sentinel_count = (shipments["parcel_weight_kg"] > 100).sum()  # 5 rows
shipments.loc[shipments["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
```

### Step 7: Robust Imputation via Medians
**Concept Taught:** The mean is sensitive to outliers, whereas the median is robust.
Example from lecture:
- Normal series: `[1.0, 2.0, NaN, 3.0, 4.0]` &rarr; Mean = 2.5, Median = 2.5
- Series with outlier: `[1.0, 2.0, NaN, 3.0, 4.0, 100.0]` &rarr; Mean = 22.0, Median = 3.0

Medians calculated from the historical dataset (1,200 deduped rows after sentinel removal):
```python
medians = {
    "distance_km": round(float(shipments["distance_km"].median()), 2),        # 314.95
    "parcel_weight_kg": round(float(shipments["parcel_weight_kg"].median()), 2), # 1.83
    "declared_value_inr": round(float(shipments["declared_value_inr"].median()), 2) # 1121.50
}
```
Impute missing values using dictionary unpacking:
```python
for column, median_value in medians.items():
    shipments[column] = shipments[column].fillna(median_value)
```

---

## 5. Feature Engineering (`build_features`)

The second core function enriches the cleaned shipments with hub network data and temporal/business features:

### 1. Hub Metadata Join
Left merge on `hub_id` to attach hub attributes:
```python
result = shipments.merge(
    hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
    on="hub_id",
    how="left",
    validate="many_to_one"
)
```

### 2. Promised Hours
Derived by mapping service levels:
```python
result["promised_hours"] = result["service_type"].map(PROMISED_HOURS).astype(int)
```

### 3. Payment Mode Flag (`is_cod`)
Binary indicator for cash on delivery (`Unknown` becomes 0):
```python
result["is_cod"] = (result["payment_mode"] == "COD").astype(int)
```

### 4. Weekend Flag (`is_weekend`)
Indicates dispatches on Saturday (5) or Sunday (6):
```python
result["is_weekend"] = (result["dispatch_time"].dt.dayofweek >= 5).astype(int)
```

### 5. Dispatch Hour (`dispatch_hour`)
Extract the integer hour of dispatch (between 6 and 21):
```python
result["dispatch_hour"] = result["dispatch_time"].dt.hour.astype(int)
```

### 6. Log-Transformed Value (`log_declared_value`)
Declared value is log-normally distributed. Applying natural log normalizes the skewness:
```python
result["log_declared_value"] = np.log(result["declared_value_inr"])
```

---

## 6. Target Definition for Part A & Beyond

In `part_a_clean_eda.py`, the binary classification target `is_delayed` is computed:
```python
result["is_delayed"] = (result["delivery_hours"] > result["promised_hours"]).astype(int)
```
- Total delayed shipments: **278** out of 1,200
- Overall historical delay rate: **23.17%** (0.2317)

