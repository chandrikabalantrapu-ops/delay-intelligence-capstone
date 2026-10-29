
from pathlib import Path

import joblib
import pandas as pd

from src.shipment_prep import (
    clean_shipments,
    build_features,
    FEATURES,
)


# --------------------------------------------------
# File paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

NEW_SHIPMENTS_FILE = DATA_DIR / "new_shipments.csv"
HUBS_FILE = DATA_DIR / "hubs.csv"

MODEL_FILE = MODEL_DIR / "delay_model.pkl"
THRESHOLD_FILE = MODEL_DIR / "delay_threshold.csv"
FEATURE_ORDER_FILE = MODEL_DIR / "feature_order.csv"
IMPUTE_VALUES_FILE = MODEL_DIR / "impute_values.csv"

HUB_SEGMENTS_FILE = OUTPUT_DIR / "hub_segments.csv"
OUTPUT_FILE = OUTPUT_DIR / "new_shipment_actions.csv"


# --------------------------------------------------
# Load the trained model and supporting files
# --------------------------------------------------

model = joblib.load(MODEL_FILE)

threshold_df = pd.read_csv(THRESHOLD_FILE)
threshold = float(threshold_df["threshold"].iloc[0])

feature_order_df = pd.read_csv(FEATURE_ORDER_FILE)
feature_order = feature_order_df.iloc[:, 0].tolist()

impute_df = pd.read_csv(IMPUTE_VALUES_FILE)
impute_values = dict(
    zip(impute_df["column"], impute_df["median"])
)

new_shipments = pd.read_csv(NEW_SHIPMENTS_FILE)
hubs = pd.read_csv(HUBS_FILE)
hub_segments = pd.read_csv(HUB_SEGMENTS_FILE)


# --------------------------------------------------
# Clean the new shipments using saved training medians
# --------------------------------------------------


# Clean new shipments using the saved training medians.
cleaned, _ = clean_shipments(
    new_shipments,
    medians=impute_values,
)


# --------------------------------------------------
# Build features using the shared preparation function
# --------------------------------------------------

features = build_features(cleaned, hubs)

# Preserve the exact feature order used during training.
X = features[feature_order]


# --------------------------------------------------
# Predict delay probabilities
# --------------------------------------------------

# Original probabilities are retained for threshold decisions.
probabilities = model.predict_proba(X)[:, 1]

# Displayed probabilities are rounded to four decimal places.
results = pd.DataFrame({
    "shipment_id": cleaned["shipment_id"].values,
    "hub_id": cleaned["hub_id"].values,
    "delay_probability": probabilities.round(4),
})


# --------------------------------------------------
# Flag shipments using ORIGINAL probabilities
# --------------------------------------------------

# IMPORTANT:
# Do not use results["delay_probability"] here because
# those values have already been rounded for display.
results["flagged"] = (
    probabilities >= threshold
).astype(int)


# --------------------------------------------------
# Add hub segment information
# --------------------------------------------------

segment_columns = ["hub_id", "segment"]

results = results.merge(
    hub_segments[segment_columns],
    on="hub_id",
    how="left",
    validate="many_to_one",
)


# --------------------------------------------------
# Assign operational actions
# --------------------------------------------------

results["action"] = "No action"

flagged_mask = results["flagged"] == 1

escalate_mask = (
    flagged_mask
    & (results["segment"] == "Escalate")
)

proactive_mask = (
    flagged_mask
    & (results["segment"] != "Escalate")
)

results.loc[
    escalate_mask, "action"
] = "Escalate to hub manager"

results.loc[
    proactive_mask, "action"
] = "Proactive customer call"


# --------------------------------------------------
# Select required columns and preserve shipment order
# --------------------------------------------------

results = results[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action",
    ]
]


# --------------------------------------------------
# Save the output
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results.to_csv(OUTPUT_FILE, index=False)

print("Shipment scoring completed successfully.")
print(f"Shipments scored: {len(results)}")
print(f"Delay threshold: {threshold}")
print(f"Flagged shipments: {results['flagged'].sum()}")

print("\nAction summary:")
print(results["action"].value_counts().to_string())

print(f"\nOutput created: {OUTPUT_FILE.relative_to(ROOT)}")
