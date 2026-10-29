
from pathlib import Path

import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the input data
# --------------------------------------------------

raw = pd.read_csv(DATA_DIR / "raw_shipments.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")

raw_rows = len(raw)

# Remove duplicate rows before calculating cleaning metrics.
raw_deduplicated = raw.drop_duplicates().copy()

duplicate_rows_removed = raw_rows - len(raw_deduplicated)


# --------------------------------------------------
# 3. Record data-quality checks before cleaning
# --------------------------------------------------

missing_distance_km_before = int(
    raw_deduplicated["distance_km"].isna().sum()
)

missing_parcel_weight_before = int(
    raw_deduplicated["parcel_weight_kg"].isna().sum()
)

missing_declared_value_before = int(
    raw_deduplicated["declared_value"].isna().sum()
)

missing_payment_mode_before = int(
    raw_deduplicated["payment_mode"].isna().sum()
)

sentinel_weight_rows = int(
    (raw_deduplicated["parcel_weight_kg"] > 100).sum()
)

service_type_spellings_before = int(
    raw_deduplicated["service_type"].nunique(dropna=True)
)


# --------------------------------------------------
# 4. Clean the shipment data
# --------------------------------------------------

cleaned, medians = clean_shipments(raw_deduplicated)

service_type_spellings_after = int(
    cleaned["service_type"].nunique(dropna=True)
)


# --------------------------------------------------
# 5. Build features and merge hub information
# --------------------------------------------------

modelling = build_features(cleaned, hubs)

rows_after_hub_merge = len(modelling)

# Delivery time and delayed status are the target columns.
modelling["delivery_hours"] = pd.to_numeric(
    modelling["delivery_hours"], errors="coerce"
)

modelling["is_delayed"] = (
    modelling["delivery_hours"] > modelling["promised_hours"]
).astype(int)


# --------------------------------------------------
# 6. Calculate data-quality and descriptive statistics
# --------------------------------------------------

feature_columns = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

required_columns = feature_columns + [
    "delivery_hours",
    "is_delayed",
]

nulls_in_features_and_target = int(
    modelling[required_columns].isna().sum().sum()
)

hours_q1 = modelling["delivery_hours"].quantile(0.25)
hours_q3 = modelling["delivery_hours"].quantile(0.75)
hours_iqr = hours_q3 - hours_q1

hours_lower_bound = hours_q1 - 1.5 * hours_iqr
hours_upper_bound = hours_q3 + 1.5 * hours_iqr

hours_outlier_count = int(
    (
        (modelling["delivery_hours"] < hours_lower_bound)
        | (modelling["delivery_hours"] > hours_upper_bound)
    ).sum()
)

declared_value_mean_inr = modelling["declared_value_inr"].mean()
declared_value_median_inr = modelling["declared_value_inr"].median()

log_declared_value_mean = modelling["log_declared_value"].mean()
log_declared_value_median = modelling["log_declared_value"].median()

delay_rate = modelling["is_delayed"].mean()
delayed_count = int(modelling["is_delayed"].sum())


# --------------------------------------------------
# 7. Save Part A data-quality checks
# --------------------------------------------------

checks = {
    "raw_rows": raw_rows,
    "duplicate_rows_removed": duplicate_rows_removed,
    "rows_after_dedup": len(raw_deduplicated),
    "missing_distance_km_before": missing_distance_km_before,
    "missing_parcel_weight_before": missing_parcel_weight_before,
    "missing_declared_value_before": missing_declared_value_before,
    "missing_payment_mode_before": missing_payment_mode_before,
    "sentinel_weight_rows": sentinel_weight_rows,
    "service_type_spellings_before": service_type_spellings_before,
    "service_type_spellings_after": service_type_spellings_after,
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": rows_after_hub_merge,
    "nulls_in_features_and_target": nulls_in_features_and_target,
    "delayed_count": delayed_count,
    "delay_rate": round(float(delay_rate), 4),
    "declared_value_mean_inr": round(float(declared_value_mean_inr), 4),
    "declared_value_median_inr": round(float(declared_value_median_inr), 4),
    "log_declared_value_mean": round(float(log_declared_value_mean), 4),
    "log_declared_value_median": round(float(log_declared_value_median), 4),
    "hours_q1": round(float(hours_q1), 4),
    "hours_q3": round(float(hours_q3), 4),
    "hours_iqr": round(float(hours_iqr), 4),
    "hours_lower_bound": round(float(hours_lower_bound), 4),
    "hours_upper_bound": round(float(hours_upper_bound), 4),
    "hours_outlier_count": hours_outlier_count,
}

checks_df = pd.DataFrame(
    list(checks.items()),
    columns=["metric", "value"],
)

checks_df.to_csv(
    OUTPUT_DIR / "part_a_checks.csv",
    index=False,
)


# --------------------------------------------------
# 8. Save delay rates by group
# --------------------------------------------------

group_results = []


def add_group_results(df, dimension, group_column):
    """Calculate shipment counts and delay rates for each group."""

    grouped = df.groupby(
        group_column,
        dropna=False,
        observed=True,
    )

    for group_name, group_df in grouped:
        group_results.append(
            {
                "dimension": dimension,
                "group": str(group_name),
                "shipments": int(len(group_df)),
                "delayed": int(group_df["is_delayed"].sum()),
                "delay_rate": round(
                    float(group_df["is_delayed"].mean()), 4
                ),
                "mean_hours": round(
                    float(group_df["delivery_hours"].mean()), 4
                ),
            }
        )


# Service type
add_group_results(
    modelling,
    "service_type",
    "service_type",
)

# Hub tier
add_group_results(
    modelling,
    "hub_tier",
    "hub_tier",
)

# Distance bands
modelling["distance_band"] = pd.cut(
    modelling["distance_km"],
    bins=[-np.inf, 200, 400, 600, np.inf],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
)

add_group_results(
    modelling,
    "distance_band",
    "distance_band",
)

group_results_df = pd.DataFrame(group_results)

group_results_df.to_csv(
    OUTPUT_DIR / "part_a_delay_by_group.csv",
    index=False,
)


# --------------------------------------------------
# 9. Save the modelling table
# --------------------------------------------------

modelling.to_csv(
    DATA_DIR / "modelling_table.csv",
    index=False,
)


# --------------------------------------------------
# 10. Save the numeric correlation matrix
# --------------------------------------------------

correlation_matrix = modelling.corr(
    numeric_only=True
).round(4)

correlation_matrix.index.name = "column"

correlation_matrix.to_csv(
    OUTPUT_DIR / "part_a_correlations.csv",
    index=True,
)


# --------------------------------------------------
# 11. Print a short completion summary
# --------------------------------------------------

print("Part A completed successfully.")
print(f"Raw rows: {raw_rows}")
print(f"Duplicate rows removed: {duplicate_rows_removed}")
print(f"Rows after deduplication: {len(raw_deduplicated)}")
print(f"Rows after hub merge: {rows_after_hub_merge}")
print(f"Delayed shipments: {delayed_count}")
print(f"Delay rate: {delay_rate:.4f}")
print(f"Nulls in features and target: {nulls_in_features_and_target}")

print("\nFiles created:")
print("data/modelling_table.csv")
print("outputs/part_a_checks.csv")
print("outputs/part_a_delay_by_group.csv")
print("outputs/part_a_correlations.csv")

print("\nTop correlations with delivery_hours:")

target_correlations = (
    correlation_matrix["delivery_hours"]
    .drop(labels=["delivery_hours"], errors="ignore")
    .dropna()
    .sort_values(key=lambda values: values.abs(), ascending=False)
)

print(target_correlations.head(3))


