
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


# --------------------------------------------------
# 1. Paths and load data
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(DATA_DIR / "modelling_table.csv")


# --------------------------------------------------
# 2. Create the hub profile
# --------------------------------------------------

hub_profile = (
    data.groupby("hub_id", as_index=False, sort=True)
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    .sort_values("hub_id")
    .reset_index(drop=True)
)

# Confirm the required number of hubs and shipments.
assert len(hub_profile) == 24
assert (hub_profile["shipments"] == 50).all()

hub_profile.to_csv(
    OUTPUT_DIR / "hub_profile.csv",
    index=False,
    float_format="%.4f",
)

print("Hub profile created.")
print(hub_profile.head())


# --------------------------------------------------
# 3. Scale the four hub profile features
# --------------------------------------------------

PROFILE_FEATURES = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]

# Keep the hubs in ascending hub_id order.
hub_profile = hub_profile.sort_values(
    "hub_id"
).reset_index(drop=True)

scaler = StandardScaler()

scaled_profiles = scaler.fit_transform(
    hub_profile[PROFILE_FEATURES]
)


# --------------------------------------------------
# 4. Calculate separation ratios for K = 2, 3, 4, 5
# --------------------------------------------------

# Calculate all pairwise Euclidean distances on the scaled matrix.
distance_matrix = pairwise_distances(
    scaled_profiles,
    metric="euclidean",
)

separation_rows = []

for k in [2, 3, 4, 5]:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    labels = kmeans.fit_predict(scaled_profiles)

    within_distances = []
    across_distances = []

    n_hubs = len(labels)

    # Visit every unordered pair exactly once.
    for i in range(n_hubs):
        for j in range(i + 1, n_hubs):
            distance = distance_matrix[i, j]

            if labels[i] == labels[j]:
                within_distances.append(distance)
            else:
                across_distances.append(distance)

    mean_within = float(np.mean(within_distances))
    mean_across = float(np.mean(across_distances))

    across_over_within = mean_across / mean_within

    cluster_sizes = sorted(
        [
            int(np.sum(labels == cluster_id))
            for cluster_id in range(k)
        ],
        reverse=True,
    )

    separation_rows.append(
        {
            "K": k,
            "across_over_within": across_over_within,
            "cluster_sizes": "-".join(
                str(size) for size in cluster_sizes
            ),
        }
    )


separation_df = pd.DataFrame(
    separation_rows,
    columns=["K", "across_over_within", "cluster_sizes"],
)

separation_df.to_csv(
    OUTPUT_DIR / "part_d_separation.csv",
    index=False,
    float_format="%.4f",
)

print("\nK-Means separation results:")
print(separation_df.to_string(index=False))


# --------------------------------------------------
# 5. Fit the required final K = 3 clustering
# --------------------------------------------------

final_kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

cluster_labels = final_kmeans.fit_predict(
    scaled_profiles
)

hub_profile["cluster_id"] = cluster_labels


# --------------------------------------------------
# 6. Name segments by mean hub-level delay rate
# --------------------------------------------------

cluster_delay_rates = (
    hub_profile.groupby("cluster_id")["delay_rate"]
    .mean()
    .sort_values()
)

ordered_cluster_ids = cluster_delay_rates.index.tolist()

segment_names = {
    ordered_cluster_ids[0]: "Standard",
    ordered_cluster_ids[1]: "Watch",
    ordered_cluster_ids[2]: "Escalate",
}

hub_profile["segment"] = (
    hub_profile["cluster_id"].map(segment_names)
)


# --------------------------------------------------
# 7. Create hub_segments.csv
# --------------------------------------------------

hub_details = (
    data[["hub_id", "hub_city", "hub_tier"]]
    .drop_duplicates(subset=["hub_id"])
)

hub_segments = hub_profile[
    ["hub_id", "segment"]
].merge(
    hub_details,
    on="hub_id",
    how="left",
    validate="one_to_one",
)

hub_segments = hub_segments[
    ["hub_id", "segment", "hub_city", "hub_tier"]
].sort_values(
    "hub_id"
).reset_index(drop=True)

hub_segments.to_csv(
    OUTPUT_DIR / "hub_segments.csv",
    index=False,
)

print("\nHub segments:")
print(hub_segments.to_string(index=False))


# --------------------------------------------------
# 8. Create segment-level profiles
# --------------------------------------------------

segment_profile = (
    hub_profile.groupby("segment", as_index=False)
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
)

# Display segments in the operational order.
segment_order = {
    "Standard": 0,
    "Watch": 1,
    "Escalate": 2,
}

segment_profile["_order"] = (
    segment_profile["segment"].map(segment_order)
)

segment_profile = (
    segment_profile.sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

segment_profile.to_csv(
    OUTPUT_DIR / "segment_profile.csv",
    index=False,
    float_format="%.4f",
)

print("\nSegment profiles:")
print(segment_profile.to_string(index=False))


# --------------------------------------------------
# 9. Identify tier-versus-segment mismatches
# --------------------------------------------------

expected_segment_by_tier = {
    1: "Standard",
    2: "Watch",
    3: "Escalate",
}

hub_segments["expected_segment"] = (
    pd.to_numeric(hub_segments["hub_tier"], errors="coerce")
    .map(expected_segment_by_tier)
)

mismatches = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
].copy()

mismatches = mismatches[
    ["hub_id", "segment", "hub_city", "hub_tier"]
].reset_index(drop=True)

mismatches.to_csv(
    OUTPUT_DIR / "tier_vs_segment_mismatch.csv",
    index=False,
)

print("\nTier-versus-segment mismatches:")
print(mismatches.to_string(index=False))


# --------------------------------------------------
# 10. Summary
# --------------------------------------------------

print("\nPart D segmentation completed successfully.")

print("\nFiles created:")
print("outputs/hub_profile.csv")
print("outputs/part_d_separation.csv")
print("outputs/hub_segments.csv")
print("outputs/segment_profile.csv")
print("outputs/tier_vs_segment_mismatch.csv")
