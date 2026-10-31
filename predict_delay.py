"""Production Delay Prediction & Action Script.

Loads 48 unlabelled parcels from data/new_shipments.csv,
preprocesses with shared pipeline using precomputed medians,
generates delay probabilities with the packaged RandomForest model,
and assigns deterministic operational actions based on segment playbooks.
"""
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

# 1. Ingest inputs and serialized model artifacts
new_raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute_df = pd.read_csv("models/impute_values.csv")
medians = {r["column"]: float(r["median"]) for _, r in impute_df.iterrows()}
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

# 2. Clean and feature-engineer using shared pipeline
clean, _ = clean_shipments(new_raw, medians=medians)
feats = build_features(clean, hubs)
X = feats[feature_order]

assert X.isna().sum().sum() == 0, "Missing values detected in prediction feature set!"

# 3. Model inference and threshold application
delay_probs = model.predict_proba(X)[:, 1].round(4)
flagged = (delay_probs >= threshold).astype(int)

# 4. Construct output action list
out = feats[["shipment_id", "hub_id"]].copy()
out["delay_probability"] = delay_probs
out["flagged"] = flagged
out = out.merge(segments, on="hub_id", how="left")

# Assign operational action
out["action"] = "No action"
out.loc[out["flagged"] == 1, "action"] = "Proactive customer call"
out.loc[(out["flagged"] == 1) & (out["segment"] == "Escalate"), "action"] = "Escalate to hub manager"

out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)

print("Scoring complete. Action distribution:")
print(out["action"].value_counts().to_string())

