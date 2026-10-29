
from pathlib import Path

import numpy as np
import pandas as pd

from src.shipment_prep import (
    FEATURES,
    PROMISED_HOURS,
    clean_shipments,
    build_features,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


def main():
    OUTPUTS.mkdir(exist_ok=True)

    # Load source data
    raw = pd.read_csv(DATA / "raw_shipments.csv")
    hubs = pd.read_csv(DATA / "hubs.csv")

    # Record initial quality metrics after deduplication
    deduplicated = raw.drop_duplicates().copy()

    raw_rows = len(raw)
    rows_after_dedup = len(deduplicated)
    duplicate_rows_removed = raw_rows - rows_after_dedup

    service_spellings_before = deduplicated["service_type"].nunique()
    missing_distance_before = deduplicated["distance_km"].isna().sum()

    weight_before = deduplicated["parcel_weight_kg"].copy()
    missing_weight_before = (
        weight_before.isna() & (weight_before <= 100)
    ).sum()

    missing_declared_before = deduplicated["declared_value"].isna().sum()
    missing_payment_before = deduplicated["payment_mode"].isna().sum()

    # Clean and engineer features
    clean, medians = clean_shipments(raw)
    service_spellings_after = clean["service_type"].nunique()

    modelling = build_features(clean, hubs)

    if len(modelling) != rows_after_dedup:
        raise ValueError("Hub merge changed the shipment row count.")

    modelling["is_delayed"] = (
        modelling["delivery_hours"] > modelling["promised_hours"]
    ).astype(int)

    modelling.to_csv(DATA / "modelling_table.csv", index=False)

    # Data-quality checks
    value = modelling["declared_value_inr"]
    log_value = np.log(value)
    q1 = modelling["delivery_hours"].quantile(0.25)
    q3 = modelling["delivery_hours"].quantile(0.75)
    iqr = q3 - q1

    checks = {
        "raw_rows": raw_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "rows_after_dedup": rows_after_dedup,
        "missing_distance_before": missing_distance_before,
        "missing_parcel_weight_before": missing_weight_before,
        "missing_declared_value_before": missing_declared_before,
        "missing_payment_mode_before": missing_payment_before,
        "service_type_spellings_before": service_spellings_before,
        "service_type_spellings_after": service_spellings_after,
        "median_distance_km": medians["distance_km"],
        "median_parcel_weight_kg": medians["parcel_weight_kg"],
        "median_declared_value_inr": medians["declared_value_inr"],
        "nulls_in_features_and_target": modelling[
            FEATURES + ["delivery_hours", "is_delayed"]
        ].isna().sum().sum(),
        "declared_value_mean_inr": value.mean(),
        "declared_value_median_inr": value.median(),
        "log_declared_value_mean": log_value.mean(),
        "log_declared_value_median": log_value.median(),
        "hours_q1": q1,
        "hours_q3": q3,
        "hours_iqr": iqr,
        "hours_lower_bound": q1 - 1.5 * iqr,
        "hours_upper_bound": q3 + 1.5 * iqr,
        "hours_outlier_count": (
            (modelling["delivery_hours"] < q1 - 1.5 * iqr)
            | (modelling["delivery_hours"] > q3 + 1.5 * iqr)
        ).sum(),
    }

    checks_df = pd.DataFrame(
        list(checks.items()), columns=["metric", "value"]
    )
    checks_df["value"] = pd.to_numeric(
        checks_df["value"], errors="coerce"
    ).round(4)
    checks_df.to_csv(OUTPUTS / "part_a_checks.csv", index=False)

    # Delay rates by service type and hub tier
    group_tables = []

    for dimension in ["service_type", "hub_tier"]:
        grouped = modelling.groupby(dimension, observed=True).agg(
            shipments=("shipment_id", "size"),
            delayed=("is_delayed", "sum"),
            mean_hours=("delivery_hours", "mean"),
        ).reset_index()

        grouped["delay_rate"] = grouped["delayed"] / grouped["shipments"]
        grouped = grouped.rename(columns={dimension: "group"})
        grouped.insert(0, "dimension", dimension)
        group_tables.append(
            grouped[
                [
                    "dimension", "group", "shipments",
                    "delayed", "delay_rate", "mean_hours",
                ]
            ]
        )

    # Distance bands: right edge inclusive
    modelling["distance_band"] = pd.cut(
        modelling["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
        include_lowest=True,
    )

    grouped = modelling.groupby(
        "distance_band", observed=False
    ).agg(
        shipments=("shipment_id", "size"),
        delayed=("is_delayed", "sum"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index()

    grouped["delay_rate"] = grouped["delayed"] / grouped["shipments"]
    grouped = grouped.rename(columns={"distance_band": "group"})
    grouped.insert(0, "dimension", "distance_band")
    group_tables.append(
        grouped[
            [
                "dimension", "group", "shipments",
                "delayed", "delay_rate", "mean_hours",
            ]
        ]
    )

    delay_groups = pd.concat(group_tables, ignore_index=True)
    delay_groups["delay_rate"] = delay_groups["delay_rate"].round(4)
    delay_groups["mean_hours"] = delay_groups["mean_hours"].round(4)
    delay_groups.to_csv(
        OUTPUTS / "part_a_delay_by_group.csv", index=False
    )

    # Correlation matrix for numeric columns
    correlations = modelling.select_dtypes(
        include="number"
    ).corr().round(4)
    correlations.to_csv(OUTPUTS / "part_a_correlations.csv")

    print("Part A processing complete.")
    print(f"Rows after deduplication: {len(modelling)}")
    print(f"Feature columns: {len(FEATURES)}")
    print("Created data/modelling_table.csv")
    print("Created outputs/part_a_checks.csv")
    print("Created outputs/part_a_delay_by_group.csv")
    print("Created outputs/part_a_correlations.csv")


if __name__ == "__main__":
    main()
