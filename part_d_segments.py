"""Part D: Hub Segments and Tier Verification.

Clusters the 24 regional delivery hubs into operating segments:
1. Computes hub profile on the 24 hubs from data/modelling_table.csv.
2. Fits K-Means (K=3) on scaled operational metrics in ascending hub_id order.
3. Computes across-over-within distance separation ratio for K in [2, 3, 4, 5].
4. Labels clusters as Standard, Watch, and Escalate (ordered by mean delay rate).
5. Detects mismatches between administrative tier labels and real operating segments.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs("outputs", exist_ok=True)

# 1. Load data and compute hub profile
table = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# Groupby hub_id (naturally sorts by ascending hub_id)
profile = table.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")
).reset_index().sort_values("hub_id").reset_index(drop=True)

profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# 2. K-Means clustering on the scaled four profile columns
PROFILE_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
scaler = StandardScaler()
Z = scaler.fit_transform(profile[PROFILE_COLS])

km3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Z)


# 3. Separation evidence for K = 2, 3, 4, 5
def compute_across_over_within(Z_matrix, cluster_labels):
    """Mean Euclidean distance between hubs in different clusters divided by that in same cluster."""
    diff = Z_matrix[:, None, :] - Z_matrix[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
    triu_idx = np.triu_indices(len(Z_matrix), k=1)
    
    pairwise_distances = dist_matrix[triu_idx]
    same_cluster = (cluster_labels[:, None] == cluster_labels[None, :])[triu_idx]
    
    across_mean = pairwise_distances[~same_cluster].mean()
    within_mean = pairwise_distances[same_cluster].mean()
    return float(across_mean / within_mean)


separation_records = []
for k in [2, 3, 4, 5]:
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)
    sizes = sorted(np.bincount(km_k.labels_), reverse=True)
    sizes_str = "-".join(str(int(s)) for s in sizes)
    ratio = compute_across_over_within(Z, km_k.labels_)
    separation_records.append({
        "K": k,
        "across_over_within": ratio,
        "cluster_sizes": sizes_str
    })

separation_df = pd.DataFrame(separation_records).round(4)
separation_df.to_csv("outputs/part_d_separation.csv", index=False)

# 4. Segment naming and tier check
profile["cluster"] = km3.labels_

# Order clusters by mean delay rate: lowest -> Standard, middle -> Watch, highest -> Escalate
cluster_delay_means = profile.groupby("cluster")["delay_rate"].mean().sort_values()
cluster_order = cluster_delay_means.index.tolist()
segment_map = dict(zip(cluster_order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(segment_map)

# hub_segments.csv (hub_id, segment, hub_city, hub_tier)
hub_segments = profile[["hub_id", "segment"]].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]],
    on="hub_id",
    how="left"
)
hub_segments.to_csv("outputs/hub_segments.csv", index=False)

# segment_profile.csv (segment, hubs, mean_delivery_hours, delay_rate, mean_distance_km, mean_n_stops)
segment_profile = profile.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")
).reset_index()

# Sort by standard segment order
segment_profile["segment"] = pd.Categorical(segment_profile["segment"], ["Standard", "Watch", "Escalate"])
segment_profile = segment_profile.sort_values("segment").reset_index(drop=True)
segment_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

# tier_vs_segment_mismatch.csv (Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate)
tier_implied_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatches = hub_segments[hub_segments["segment"] != hub_segments["hub_tier"].map(tier_implied_segment)]
mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print(f"Part D segmentation complete. Hubs: {len(hub_segments)}. Mismatches detected: {len(mismatches)}.")
