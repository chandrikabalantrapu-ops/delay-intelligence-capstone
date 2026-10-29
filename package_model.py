
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the modelling table and saved configuration
# --------------------------------------------------

data = pd.read_csv(DATA_DIR / "modelling_table.csv")
config = pd.read_csv(OUTPUT_DIR / "part_c_final_config.csv")

if len(config) != 1:
    raise ValueError("Expected exactly one row in part_c_final_config.csv.")

config_row = config.iloc[0]

selector_k = int(config_row["selector_k"])
n_estimators = int(config_row["n_estimators"])
min_samples_leaf = int(config_row["min_samples_leaf"])
threshold = float(config_row["threshold"])

max_depth_value = config_row["max_depth"]

if str(max_depth_value).strip().lower() == "unlimited":
    max_depth = None
else:
    max_depth = int(max_depth_value)


# --------------------------------------------------
# 3. Reproduce Part C's exact stratified split
# --------------------------------------------------

X = data[FEATURES].copy()
y = data["is_delayed"].astype(int).copy()

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

assert len(X_train) == 840
assert len(X_val) == 180
assert len(X_test) == 180


# --------------------------------------------------
# 4. Rebuild and fit the final classifier
# --------------------------------------------------

model = Pipeline(
    [
        ("selector", SelectKBest(f_classif, k=selector_k)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 5. Save the reusable model and metadata
# --------------------------------------------------

model_path = MODELS_DIR / "delay_model.pkl"
joblib.dump(model, model_path)

pd.DataFrame(
    {"threshold": [threshold]}
).to_csv(
    MODELS_DIR / "delay_threshold.csv",
    index=False,
)

pd.DataFrame(
    {"feature": FEATURES}
).to_csv(
    MODELS_DIR / "feature_order.csv",
    index=False,
)

# Obtain the three imputation medians from the shared cleaning function.
raw_data = pd.read_csv(DATA_DIR / "raw_shipments.csv")
_, medians = clean_shipments(raw_data)

median_rows = [
    {
        "column": column,
        "median": medians[column],
    }
    for column in [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    ]
]

pd.DataFrame(median_rows).to_csv(
    MODELS_DIR / "impute_values.csv",
    index=False,
)


# --------------------------------------------------
# 6. Reload the saved model and compare probabilities
# --------------------------------------------------

reloaded_model = joblib.load(model_path)

original_probabilities = model.predict_proba(X_test)[:, 1]
reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

reload_max_abs_probability_diff = float(
    np.max(
        np.abs(
            original_probabilities - reloaded_probabilities
        )
    )
)

reload_check = pd.DataFrame(
    [
        {
            "metric": "test_rows_scored",
            "value": len(X_test),
        },
        {
            "metric": "reload_max_abs_probability_diff",
            "value": reload_max_abs_probability_diff,
        },
        {
            "metric": "threshold",
            "value": threshold,
        },
    ]
)

reload_check.to_csv(
    OUTPUT_DIR / "part_d_reload_check.csv",
    index=False,
)

print("Model packaging completed successfully.")
print(f"Model saved to: {model_path}")
print(f"Test rows scored: {len(X_test)}")
print(
    "Maximum probability difference after reload: "
    f"{reload_max_abs_probability_diff:.12g}"
)

print("\nFiles created:")
print("models/delay_model.pkl")
print("models/delay_threshold.csv")
print("models/feature_order.csv")
print("models/impute_values.csv")
print("outputs/part_d_reload_check.csv")
