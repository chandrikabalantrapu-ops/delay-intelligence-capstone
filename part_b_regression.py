
from pathlib import Path

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


# --------------------------------------------------
# 1. Paths and data loading
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(DATA_DIR / "modelling_table.csv")

X = data[FEATURES].copy()
y = data["delivery_hours"].copy()

if len(FEATURES) != 17:
    raise ValueError(
        f"Expected 17 FEATURES, but found {len(FEATURES)}."
    )

if X.isna().any().any() or y.isna().any():
    raise ValueError("Missing values found in model features or target.")


# --------------------------------------------------
# 2. Split into training, validation and test sets
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
)

print("Dataset split sizes:")
print(f"Training:   {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test:       {len(X_test)}")

assert len(X_train) == 840
assert len(X_val) == 180
assert len(X_test) == 180


# --------------------------------------------------
# 3. Scaling diagnostic: fit on training rows only
# --------------------------------------------------

diagnostic_scaler = StandardScaler()

X_train_scaled = diagnostic_scaler.fit_transform(X_train)
X_test_scaled = diagnostic_scaler.transform(X_test)

train_scaled_max_abs_mean = float(
    np.abs(X_train_scaled.mean(axis=0)).max()
)

test_scaled_max_abs_mean = float(
    np.abs(X_test_scaled.mean(axis=0)).max()
)

checks = pd.DataFrame(
    [
        {"metric": "train_rows", "value": len(X_train)},
        {"metric": "val_rows", "value": len(X_val)},
        {"metric": "test_rows", "value": len(X_test)},
        {
            "metric": "train_scaled_max_abs_mean",
            "value": train_scaled_max_abs_mean,
        },
        {
            "metric": "test_scaled_max_abs_mean",
            "value": test_scaled_max_abs_mean,
        },
    ]
)

checks.to_csv(
    OUTPUT_DIR / "part_b_checks.csv",
    index=False,
)

print("\nScaling diagnostic:")
print(checks.to_string(index=False))


# --------------------------------------------------
# 4. Metric calculation helper
# --------------------------------------------------

def calculate_metrics(y_true, predictions):
    """Calculate regression metrics for a set of predictions."""

    mae = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, predictions)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


metric_rows = []


def add_metric_row(model_name, split_name, y_true, predictions):
    """Append one row of regression evaluation metrics."""

    metrics = calculate_metrics(y_true, predictions)

    metric_rows.append(
        {
            "model": model_name,
            "split": split_name,
            "MAE": metrics["MAE"],
            "MSE": metrics["MSE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
        }
    )


# --------------------------------------------------
# 5. Baseline: predict the training target mean
# --------------------------------------------------

mean_baseline_prediction = np.full(
    shape=len(y_val),
    fill_value=y_train.mean(),
)

add_metric_row(
    "mean_baseline",
    "val",
    y_val,
    mean_baseline_prediction,
)


# --------------------------------------------------
# 6. Decision tree depth diagnostic
# --------------------------------------------------

depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_depth_rows = []
tree_models = {}

for depth in depths:
    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42,
    )

    tree.fit(X_train, y_train)

    train_predictions = tree.predict(X_train)
    val_predictions = tree.predict(X_val)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_predictions)
    )

    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_predictions)
    )

    depth_label = "unlimited" if depth is None else depth

    tree_depth_rows.append(
        {
            "max_depth": depth_label,
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
        }
    )

    tree_models[depth_label] = tree


tree_depth_df = pd.DataFrame(tree_depth_rows)

tree_depth_df.to_csv(
    OUTPUT_DIR / "part_b_tree_depth.csv",
    index=False,
)

# Find the lowest validation RMSE.
# In a tie, select the smaller numerical depth.
# Unlimited depth comes after all numerical depths.
best_tree_row = min(
    tree_depth_rows,
    key=lambda row: (
        row["val_rmse"],
        float("inf")
        if row["max_depth"] == "unlimited"
        else row["max_depth"],
    ),
)

best_depth = best_tree_row["max_depth"]
best_tree = tree_models[best_depth]

print("\nDecision tree depth diagnostic:")
print(tree_depth_df.to_string(index=False))
print(f"\nSelected decision tree depth: {best_depth}")


# --------------------------------------------------
# 7. Define the four candidate models
# --------------------------------------------------

candidate_models = {
    "linear_regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    ),

    "decision_tree": best_tree,

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    ),
}


# --------------------------------------------------
# 8. Fit candidates and evaluate on validation rows
# --------------------------------------------------

fitted_models = {}

for model_name, model in candidate_models.items():
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)

    add_metric_row(
        model_name,
        "val",
        y_val,
        val_predictions,
    )

    fitted_models[model_name] = model


# --------------------------------------------------
# 9. Gradient boosting learning-rate experiment
# --------------------------------------------------

learning_rate_rows = []

for learning_rate in [0.01, 0.1, 1.0]:
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=learning_rate,
        random_state=42,
    )

    gb_model.fit(X_train, y_train)

    val_predictions = gb_model.predict(X_val)

    metrics = calculate_metrics(y_val, val_predictions)

    learning_rate_rows.append(
        {
            "learning_rate": learning_rate,
            "val_rmse": metrics["RMSE"],
            "val_r2": metrics["R2"],
        }
    )


learning_rate_df = pd.DataFrame(learning_rate_rows)

learning_rate_df.to_csv(
    OUTPUT_DIR / "part_b_gb_learning_rate.csv",
    index=False,
)

print("\nGradient boosting learning-rate experiment:")
print(learning_rate_df.to_string(index=False))


# --------------------------------------------------
# 10. Rank candidates by validation RMSE
# --------------------------------------------------

validation_metrics = pd.DataFrame(metric_rows)

candidate_validation_metrics = validation_metrics[
    (validation_metrics["split"] == "val")
    & (validation_metrics["model"] != "mean_baseline")
].copy()

candidate_validation_metrics = candidate_validation_metrics.sort_values(
    by="RMSE",
    ascending=True,
    kind="stable",
)

print("\nCandidates ranked by validation RMSE:")
print(
    candidate_validation_metrics[
        ["model", "RMSE", "R2"]
    ].to_string(index=False)
)

best_model_name = candidate_validation_metrics.iloc[0]["model"]
best_model = fitted_models[best_model_name]

print(f"\nWinning validation model: {best_model_name}")


# --------------------------------------------------
# 11. Cross-validation for the two best candidates
# --------------------------------------------------

# Combine training and validation rows.
X_train_val = pd.concat(
    [X_train, X_val],
    axis=0,
)

y_train_val = pd.concat(
    [y_train, y_val],
    axis=0,
)

# One shared KFold object for both models.
shared_kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

top_two_model_names = candidate_validation_metrics.head(2)[
    "model"
].tolist()

cv_rows = []

for model_name in top_two_model_names:
    # Use a fresh, unfitted clone of the model configuration.
    model = candidate_models[model_name]

    cv_scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=shared_kfold,
        # Default scoring is R²; deliberately do not set scoring.
    )

    cv_rows.append(
        {
            "model": model_name,
            "mean_r2": float(cv_scores.mean()),
            "std_r2": float(cv_scores.std()),
        }
    )


cv_df = pd.DataFrame(cv_rows)

cv_df.to_csv(
    OUTPUT_DIR / "part_b_cv.csv",
    index=False,
)

print("\nFive-fold cross-validation:")
print(cv_df.to_string(index=False))


# --------------------------------------------------
# 12. Final test evaluation — exactly once
# --------------------------------------------------

# The winning candidate remains fitted on the training rows.
# Do not refit it using validation or test data.
test_predictions = best_model.predict(X_test)

add_metric_row(
    best_model_name,
    "test",
    y_test,
    test_predictions,
)


# --------------------------------------------------
# 13. Save the final metrics file
# --------------------------------------------------

metrics_df = pd.DataFrame(
    metric_rows,
    columns=["model", "split", "MAE", "MSE", "RMSE", "R2"],
)

# Keep metric values precise enough for RMSE = sqrt(MSE).
metrics_df.to_csv(
    OUTPUT_DIR / "part_b_metrics.csv",
    index=False,
    float_format="%.10f",
)


# --------------------------------------------------
# 14. Final summary
# --------------------------------------------------

print("\nPart B completed successfully.")

print("\nFinal validation metrics:")
print(
    metrics_df[metrics_df["split"] == "val"].to_string(index=False)
)

print("\nFinal test metric:")
print(
    metrics_df[metrics_df["split"] == "test"].to_string(index=False)
)

print("\nFiles created:")
print("outputs/part_b_checks.csv")
print("outputs/part_b_tree_depth.csv")
print("outputs/part_b_metrics.csv")
print("outputs/part_b_gb_learning_rate.csv")
print("outputs/part_b_cv.csv")
