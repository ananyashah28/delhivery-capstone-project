"""Part C: Delay Classifier and Alert Threshold.

Evaluates delay classification models and selects the operational alert threshold:
1. Stratified split (840 / 180 / 180).
2. Blind baseline and 3 candidate models on validation (0.5 rule).
3. Feature signal and noise diagnostic (F-score vs Random Forest importance), SelectKBest k-search.
4. Hyperparameter tuning (GridSearchCV, RandomizedSearchCV, and default forest on top 7 features).
5. Final model pipeline configuration.
6. Alert threshold optimization on validation probabilities (target recall >= 0.80).
7. Single evaluation of the final model on the held-out test set.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

# Load data
table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["is_delayed"]

# 1. Stratified split (two steps)
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
)

print(f"Split sizes: Train={len(X_train)} (delayed={int(y_train.sum())}), "
      f"Val={len(X_val)} (delayed={int(y_val.sum())}), "
      f"Test={len(X_test)} (delayed={int(y_test.sum())})")


def compute_cm_metrics(y_true, pred):
    """Compute TN, FP, FN, TP, accuracy, precision, recall, and f1."""
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0))
    }


# 2. Blind baseline and three classifiers on validation (default 0.5 decision rule)
model_rows = [{
    "model": "blind_all_on_time",
    **compute_cm_metrics(y_val, np.zeros(len(y_val), dtype=int))
}]

# Logistic regression pipeline
logit_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
]).fit(X_train, y_train)
model_rows.append({
    "model": "logistic_regression",
    **compute_cm_metrics(y_val, logit_pipe.predict(X_val))
})

# Decision tree with optimal depth chosen by validation F1
depth_candidates = [2, 3, 4, 5, 6, 8, 10]
best_depth = None
best_val_f1 = -1.0

for d in depth_candidates:
    dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    val_f1 = f1_score(y_val, dt.predict(X_val), zero_division=0)
    # Tie-breaking: smaller depth chosen on ties
    if val_f1 > best_val_f1 + 1e-12:
        best_val_f1 = val_f1
        best_depth = d

best_tree = DecisionTreeClassifier(criterion="gini", max_depth=best_depth, random_state=42).fit(X_train, y_train)
model_rows.append({
    "model": f"decision_tree_depth_{best_depth}",
    **compute_cm_metrics(y_val, best_tree.predict(X_val))
})

# Default Random Forest
rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
model_rows.append({
    "model": "random_forest_default",
    **compute_cm_metrics(y_val, rf_default.predict(X_val))
})

models_df = pd.DataFrame(model_rows).round(4)
models_df.to_csv("outputs/part_c_models.csv", index=False)

# 3. Which columns carry signal?
sk_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": sk_all.scores_,
    "rf_importance": rf_default.feature_importances_
})
fs_df["f_rank"] = fs_df["f_score"].rank(ascending=False, method="first").astype(int)
fs_df["is_noise"] = fs_df["feature"].isin(NOISE_COLUMNS)
fs_df = fs_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
fs_df.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

noise_share = float(fs_df.loc[fs_df["is_noise"], "rf_importance"].sum())
print(f"Noise columns' share of RF importance: {round(noise_share, 4)}")

# Evaluate SelectKBest for k in [5, 7, 10, 12, 17]
k_records = []
for k in [5, 7, 10, 12, 17]:
    k_pipe = Pipeline([
        ("sel", SelectKBest(f_classif, k=k)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ]).fit(X_train, y_train)
    k_val_f1 = float(f1_score(y_val, k_pipe.predict(X_val), zero_division=0))
    k_records.append({"k": k, "val_f1": k_val_f1})

selectk_df = pd.DataFrame(k_records).round(4)
selectk_df.to_csv("outputs/part_c_selectk.csv", index=False)

# 4. Tuning on top 7 features (selector fitted on training rows only)
K = 7
selector = SelectKBest(f_classif, k=K).fit(X_train, y_train)
X_train_k7 = selector.transform(X_train)
X_val_k7 = selector.transform(X_val)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

# GridSearchCV
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5).fit(X_train_k7, y_train)
# RandomizedSearchCV
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5, random_state=42).fit(X_train_k7, y_train)
# Untuned default forest on same 7 features
default_forest_k7 = RandomForestClassifier(random_state=42).fit(X_train_k7, y_train)

DEFAULT_PARAMS = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}

tuning_candidates = {
    "grid_search": (
        gs.best_params_,
        float(gs.best_score_),
        len(gs.cv_results_["params"]),
        5,
        float(f1_score(y_val, gs.best_estimator_.predict(X_val_k7), zero_division=0))
    ),
    "randomized_search": (
        rs.best_params_,
        float(rs.best_score_),
        len(rs.cv_results_["params"]),
        5,
        float(f1_score(y_val, rs.best_estimator_.predict(X_val_k7), zero_division=0))
    ),
    "default_forest_same_7_features": (
        DEFAULT_PARAMS,
        np.nan,
        1,
        0,
        float(f1_score(y_val, default_forest_k7.predict(X_val_k7), zero_division=0))
    ),
}

tuning_records = []
for name, (params, cv_acc, n_cand, folds, vf1) in tuning_candidates.items():
    formatted_params = json.dumps({
        k: (None if params[k] is None else int(params[k])) for k in sorted(params)
    })
    tuning_records.append({
        "search": name,
        "candidates": n_cand,
        "cv_folds": folds,
        "total_fits": (n_cand * folds) if folds else 1,
        "best_params": formatted_params,
        "best_cv_accuracy": cv_acc,
        "val_f1": vf1
    })

tuning_df = pd.DataFrame(tuning_records).round(4)
tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)

# 5. Choosing the final configuration
priority_order = ["default_forest_same_7_features", "grid_search", "randomized_search"]
best_val_score = tuning_df["val_f1"].max()
chosen_name = next(
    name for name in priority_order
    if tuning_df.loc[tuning_df["search"] == name, "val_f1"].iloc[0] == best_val_score
)
chosen_params = tuning_candidates[chosen_name][0]

final_classifier = RandomForestClassifier(random_state=42, **chosen_params)
final_pipeline = Pipeline([
    ("selector", SelectKBest(f_classif, k=K)),
    ("model", final_classifier)
]).fit(X_train, y_train)

# 6. Alert threshold optimization on validation probabilities
threshold_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_probs = final_pipeline.predict_proba(X_val)[:, 1]

threshold_records = []
for t in threshold_candidates:
    preds_t = (val_probs >= t).astype(int)
    threshold_records.append({"threshold": t, **compute_cm_metrics(y_val, preds_t)})

threshold_df = pd.DataFrame(threshold_records).round(4)
threshold_df.to_csv("outputs/part_c_thresholds.csv", index=False)

# Operations rule: highest threshold with recall >= 0.80
eligible_thresholds = threshold_df.loc[threshold_df["recall"] >= 0.80, "threshold"]
chosen_threshold = float(eligible_thresholds.max()) if len(eligible_thresholds) else 0.1

# 7. One single test evaluation
test_probs = final_pipeline.predict_proba(X_test)[:, 1]
final_test_preds = (test_probs >= chosen_threshold).astype(int)
test_record = [{"threshold": chosen_threshold, **compute_cm_metrics(y_test, final_test_preds)}]

final_test_df = pd.DataFrame(test_record).round(4)
final_test_df.to_csv("outputs/part_c_final_test.csv", index=False)

max_depth_val = "unlimited" if chosen_params.get("max_depth") is None else str(chosen_params["max_depth"])
config_record = [{
    "selector_k": K,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": max_depth_val,
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_name
}]
config_df = pd.DataFrame(config_record)
config_df.to_csv("outputs/part_c_final_config.csv", index=False)

print(f"Part C complete. Chosen configuration: {chosen_name} with parameters {chosen_params}.")
print(f"Chosen Alert Threshold: {chosen_threshold}.")
