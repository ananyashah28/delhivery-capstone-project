"""Part A: Data Foundation and Delay Evidence.

Reads raw shipments and hubs, runs cleaning and feature engineering,
exports data/modelling_table.csv, and computes EDA outputs in outputs/.
"""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 1. Ingest raw shipments and hubs metadata
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# 2. Compute "before" measurements (after de-duplication, before any filling)
dedup = raw.drop_duplicates()
declared_num = pd.to_numeric(
    dedup["declared_value"].astype("string")
    .str.replace("\u20b9", "", regex=False)
    .str.replace(",", "", regex=False),
    errors="coerce"
)
weight_before = dedup["parcel_weight_kg"]
spell_before = dedup["service_type"].nunique()

# 3. Clean shipments and build features
clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)

assert len(table) == len(clean), f"Hub merge changed row count: {len(table)} != {len(clean)}"
table.to_csv("data/modelling_table.csv", index=False)

# 4. Outlier diagnostics and metrics
h = table["delivery_hours"]
q1, q3 = float(h.quantile(0.25)), float(h.quantile(0.75))
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
dv = table["declared_value_inr"]
ldv = table["log_declared_value"]

metrics = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", int(dedup["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(weight_before.isna().sum())),
    ("sentinel_weight_rows", int((weight_before > 100).sum())),
    ("missing_declared_value_before", int(declared_num.isna().sum())),
    ("missing_payment_mode_before", int(dedup["payment_mode"].isna().sum())),
    ("service_type_spellings_before", int(spell_before)),
    ("service_type_spellings_after", int(table["service_type"].nunique())),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(table)),
    ("nulls_in_features_and_target", int(table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(table["is_delayed"].sum())),
    ("delay_rate", float(table["is_delayed"].mean())),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lo),
    ("hours_upper_bound", hi),
    ("hours_outlier_count", int(((h < lo) | (h > hi)).sum())),
    ("declared_value_mean_inr", float(dv.mean())),
    ("declared_value_median_inr", float(dv.median())),
    ("log_declared_value_mean", float(ldv.mean())),
    ("log_declared_value_median", float(ldv.median())),
]

checks = pd.DataFrame(metrics, columns=["metric", "value"])
checks["value"] = checks["value"].astype(float).round(4)
checks.to_csv("outputs/part_a_checks.csv", index=False)

# 5. Delay evidence table by group
table["distance_band"] = pd.cut(
    table["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True
)

rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = table.groupby(dim, observed=True).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    )
    for grp, r in g.iterrows():
        rows.append([dim, str(grp), int(r["shipments"]), int(r["delayed"]), r["delay_rate"], r["mean_hours"]])

by_group = pd.DataFrame(rows, columns=["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"])
by_group[["delay_rate", "mean_hours"]] = by_group[["delay_rate", "mean_hours"]].round(4)
by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# 6. Correlation matrix of numeric columns
numeric_cols = table.drop(columns=["distance_band"]).select_dtypes(include=[np.number])
corr = numeric_cols.corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv")

print(f"Part A complete: {len(table)} rows in modelling table; {int(table['is_delayed'].sum())} delayed parcels.")

