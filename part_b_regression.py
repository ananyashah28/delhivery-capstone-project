"""Part B: Delivery-Hours Regression.

Evaluates regression models for delivery_hours prediction:
1. Leakage-free train/val/test split (840 / 180 / 180).
2. StandardScaler verification on train vs test.
3. Mean baseline evaluation.
4. Tree depth diagnostics.
5. Evaluation of four candidate models.
6. Gradient boosting learning rate experiment.
7. 5-fold cross-validation of top 2 candidates and final test evaluation.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)

# Load modelling table
table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["delivery_hours"]

# 1. Split into training (70%), validation (15%), and test (15%)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# 2. Scaling without leakage
scaler = StandardScaler().fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)

b_checks = pd.DataFrame({
    "metric": [
        "train_rows",
        "val_rows",
        "test_rows",
        "train_scaled_max_abs_mean",
        "test_scaled_max_abs_mean"
    ],
    "value": [
        len(X_train),
        len(X_val),
        len(X_test),
        float(np.abs(train_scaled.mean(axis=0)).max()),
        float(np.abs(test_scaled.mean(axis=0)).max())
    ]
})
b_checks["value"] = b_checks["value"].round(4)
b_checks.to_csv("outputs/part_b_checks.csv", index=False)


def evaluate_predictions(y_true, pred):
    """Compute MAE, MSE, RMSE, and R2."""
    mse = float(mean_squared_error(y_true, pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, pred))
    r2 = float(r2_score(y_true, pred))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# 3. Baseline model: predicts mean of y_train for every validation row
metrics_rows = [{
    "model": "mean_baseline",
    "split": "val",
    **evaluate_predictions(y_val, np.full(len(y_val), y_train.mean()))
}]

# 4. Tree-depth diagnostic
depths = [1, 2, 3, 4, 6, 8, 12, None]
depth_records = []
best_depth = None
best_val_rmse = float("inf")

for d in depths:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    tr_rmse = float(np.sqrt(mean_squared_error(y_train, tree.predict(X_train))))
    va_rmse = float(np.sqrt(mean_squared_error(y_val, tree.predict(X_val))))
    depth_label = "unlimited" if d is None else str(d)
    depth_records.append({
        "max_depth": depth_label,
        "train_rmse": tr_rmse,
        "val_rmse": va_rmse
    })
    # Tie-breaking: smaller depth chosen on ties
    if va_rmse < best_val_rmse - 1e-12:
        best_val_rmse = va_rmse
        best_depth = d

tree_depth_df = pd.DataFrame(depth_records).round(4)
tree_depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)

# 5. Four candidate models
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}

val_rmse_map = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    scores = evaluate_predictions(y_val, preds)
    val_rmse_map[name] = scores["RMSE"]
    metrics_rows.append({"model": name, "split": "val", **scores})

# 6. Learning-rate experiment
lr_records = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    gb_preds = gb.predict(X_val)
    lr_records.append({
        "learning_rate": lr,
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, gb_preds))),
        "val_r2": float(r2_score(y_val, gb_preds))
    })

lr_df = pd.DataFrame(lr_records).round(4)
lr_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7. Cross-validated comparison and final model
# Rank candidates by validation RMSE
ranked_candidates = sorted(val_rmse_map, key=val_rmse_map.get)
top2 = ranked_candidates[:2]

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_records = []
for name in top2:
    scores = cross_val_score(candidates[name], X_train_val, y_train_val, cv=kf)
    cv_records.append({
        "model": name,
        "mean_r2": float(scores.mean()),
        "std_r2": float(scores.std())
    })

cv_df = pd.DataFrame(cv_records).round(4)
cv_df.to_csv("outputs/part_b_cv.csv", index=False)

# Final model is the candidate with lowest validation RMSE
final_name = ranked_candidates[0]
final_model = candidates[final_name]
test_preds = final_model.predict(X_test)
test_scores = evaluate_predictions(y_test, test_preds)
metrics_rows.append({"model": final_name, "split": "test", **test_scores})

metrics_df = pd.DataFrame(metrics_rows)[["model", "split", "MAE", "MSE", "RMSE", "R2"]].round(4)
metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)

print(f"Part B complete. Best tree depth: {best_depth}. Ranked candidates: {ranked_candidates}. Final model: {final_name}.")
