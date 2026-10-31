"""Package Model Pipeline.

Rebuilds Part C's winning pipeline configuration, fits on training rows,
and serializes artifacts into models/:
- models/delay_model.pkl
- models/delay_threshold.csv
- models/feature_order.csv
- models/impute_values.csv

Verifies exact reload parity and writes outputs/part_d_reload_check.csv.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 1. Read winning configuration from Part C
cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
selector_k = int(cfg["selector_k"])
n_estimators = int(cfg["n_estimators"])
max_depth = None if str(cfg["max_depth"]).lower() == "unlimited" else int(cfg["max_depth"])
min_samples_leaf = int(cfg["min_samples_leaf"])
threshold_val = float(cfg["threshold"])

# 2. Rebuild Part C's stratified split
table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["is_delayed"]

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
)

# 3. Fit final pipeline on training rows
model = Pipeline([
    ("selector", SelectKBest(f_classif, k=selector_k)),
    ("model", RandomForestClassifier(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    ))
])
model.fit(X_train, y_train)

# 4. Serialize artifacts to models/
joblib.dump(model, "models/delay_model.pkl")

# models/delay_threshold.csv (one column: threshold)
pd.DataFrame({"threshold": [threshold_val]}).to_csv("models/delay_threshold.csv", index=False)

# models/feature_order.csv (one column: feature)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

# models/impute_values.csv (columns: column, median)
_, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))
impute_df = pd.DataFrame({
    "column": list(medians.keys()),
    "median": [round(float(v), 4) for v in medians.values()]
})
impute_df.to_csv("models/impute_values.csv", index=False)

# 5. Reload verification test
reloaded_model = joblib.load("models/delay_model.pkl")
in_memory_probs = model.predict_proba(X_test)[:, 1]
reloaded_probs = reloaded_model.predict_proba(X_test)[:, 1]
max_diff = float(np.abs(reloaded_probs - in_memory_probs).max())

reload_check = pd.DataFrame({
    "metric": [
        "test_rows_scored",
        "reload_max_abs_probability_diff",
        "threshold"
    ],
    "value": [
        len(X_test),
        max_diff,
        threshold_val
    ]
})
reload_check["value"] = reload_check["value"].round(4)
reload_check.to_csv("outputs/part_d_reload_check.csv", index=False)

print(f"Model successfully packaged to models/. Reload max probability difference: {max_diff}.")

