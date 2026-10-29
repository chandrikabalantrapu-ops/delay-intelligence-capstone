
from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.shipment_prep import FEATURES, NOISE_COLUMNS


# --------------------------------------------------
# 1. Paths and data loading
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(DATA_DIR / "modelling_table.csv")

X = data[FEATURES].copy()
y = data["is_delayed"].astype(int).copy()

if len(FEATURES) != 17:
    raise ValueError(f"Expected 17 features, got {len(FEATURES)}.")

if X.isna().any().any() or y.isna().any():
    raise ValueError("Missing values found in model features or target.")


# --------------------------------------------------
# 2. Stratified train / validation / test split
# --------------------------------------------------

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

assert int(y_train.sum()) == 195
assert int(y_val.sum()) == 42
assert int(y_test.sum()) == 41

print("Split sizes and delayed counts:")
print(
    f"Train: {len(X_train)} rows, {int(y_train.sum())} delayed"
)
print(
    f"Validation: {len(X_val)} rows, {int(y_val.sum())} delayed"
)
print(
    f"Test: {len(X_test)} rows, {int(y_test.sum())} delayed"
)


# --------------------------------------------------
# 3. Metric helper
# --------------------------------------------------

def calculate_classification_metrics(y_true, predictions):
    """Return confusion-matrix cells and classification metrics."""

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
    }


def append_metric_row(rows, model_name, y_true, predictions):
    metrics = calculate_classification_metrics(
        y_true,
        predictions,
    )
    metrics["model"] = model_name
    rows.append(metrics)


# --------------------------------------------------
# 4. Blind baseline
# --------------------------------------------------

model_rows = []

blind_predictions = np.zeros(len(y_val), dtype=int)

append_metric_row(
    model_rows,
    "blind_all_on_time",
    y_val,
    blind_predictions,
)

# --------------------------------------------------
# 5. Logistic regression
# --------------------------------------------------

logistic_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ]
)

logistic_model.fit(X_train, y_train)

logistic_val_predictions = logistic_model.predict(X_val)

append_metric_row(
    model_rows,
    "logistic_regression",
    y_val,
    logistic_val_predictions,
)


# --------------------------------------------------
# 6. Select the best decision-tree depth by validation F1
# --------------------------------------------------

tree_depth_results = []
tree_candidates = {}

for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42,
    )

    tree.fit(X_train, y_train)
    predictions = tree.predict(X_val)
    metrics = calculate_classification_metrics(y_val, predictions)

    tree_candidates[depth] = tree
    tree_depth_results.append(
        {
            "depth": depth,
            "val_f1": metrics["f1"],
        }
    )

# Highest F1 wins; smaller depth breaks ties.
best_tree_depth = min(
    tree_depth_results,
    key=lambda row: (-row["val_f1"], row["depth"]),
)["depth"]

best_tree = tree_candidates[best_tree_depth]

append_metric_row(
    model_rows,
    f"decision_tree_depth_{best_tree_depth}",
    y_val,
    best_tree.predict(X_val),
)

print(f"\nSelected decision-tree depth: {best_tree_depth}")


# --------------------------------------------------
# 7. Default random forest using all 17 features
# --------------------------------------------------

default_forest_all = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

default_forest_all.fit(X_train, y_train)

append_metric_row(
    model_rows,
    "random_forest_default",
    y_val,
    default_forest_all.predict(X_val),
)


# --------------------------------------------------
# 8. Save Part C model comparison
# --------------------------------------------------

models_df = pd.DataFrame(
    model_rows,
    columns=[
        "model",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ],
)

models_df.to_csv(
    OUTPUT_DIR / "part_c_models.csv",
    index=False,
    float_format="%.10f",
)


# --------------------------------------------------
# 9. Feature scores and random-forest importances
# --------------------------------------------------

forest_importances = default_forest_all.feature_importances_

selector_all = SelectKBest(
    score_func=f_classif,
    k="all",
)

selector_all.fit(X_train, y_train)

f_scores = np.nan_to_num(
    selector_all.scores_,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

# Sort descending by F-score, using original feature order for ties.
feature_order = sorted(
    range(len(FEATURES)),
    key=lambda i: (-f_scores[i], i),
)

feature_ranks = np.empty(len(FEATURES), dtype=int)

for rank, index in enumerate(feature_order, start=1):
    feature_ranks[index] = rank

feature_scores_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "f_score": f_scores,
        "f_rank": feature_ranks,
        "rf_importance": forest_importances,
        "is_noise": [
            feature in NOISE_COLUMNS
            for feature in FEATURES
        ],
    }
)

feature_scores_df.to_csv(
    OUTPUT_DIR / "part_c_feature_scores.csv",
    index=False,
    float_format="%.10f",
)

noise_importance_share = float(
    feature_scores_df.loc[
        feature_scores_df["is_noise"],
        "rf_importance",
    ].sum()
)

print(
    "\nCombined importance of noise columns: "
    f"{noise_importance_share:.4f}"
)


# --------------------------------------------------
# 10. SelectKBest comparison: k = 5, 7, 10, 12, 17
# --------------------------------------------------

selectk_rows = []

for k in [5, 7, 10, 12, 17]:
    selectk_model = Pipeline(
        [
            ("selector", SelectKBest(f_classif, k=k)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )

    selectk_model.fit(X_train, y_train)
    predictions = selectk_model.predict(X_val)

    selectk_rows.append(
        {
            "k": k,
            "val_f1": float(
                f1_score(
                    y_val,
                    predictions,
                    zero_division=0,
                )
            ),
        }
    )

selectk_df = pd.DataFrame(selectk_rows)

selectk_df.to_csv(
    OUTPUT_DIR / "part_c_selectk.csv",
    index=False,
    float_format="%.10f",
)

print("\nSelectKBest comparison:")
print(selectk_df.to_string(index=False))


# --------------------------------------------------
# 11. Fit the k=7 selector on training data only
# --------------------------------------------------

selector_7 = SelectKBest(
    score_func=f_classif,
    k=7,
)

X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)

# X_test is deliberately not transformed or evaluated here.


# --------------------------------------------------
# 12. Grid search and randomized search
# --------------------------------------------------

parameter_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=parameter_grid,
    cv=5,
)

grid_search.fit(X_train_7, y_train)

grid_predictions = grid_search.best_estimator_.predict(X_val_7)

grid_val_f1 = f1_score(
    y_val,
    grid_predictions,
    zero_division=0,
)

grid_candidates = (
    len(parameter_grid["n_estimators"])
    * len(parameter_grid["max_depth"])
    * len(parameter_grid["min_samples_leaf"])
)

grid_total_fits = grid_candidates * 5


randomized_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=parameter_grid,
    n_iter=9,
    cv=5,
    random_state=42,
)

randomized_search.fit(X_train_7, y_train)

randomized_predictions = (
    randomized_search.best_estimator_.predict(X_val_7)
)

randomized_val_f1 = f1_score(
    y_val,
    randomized_predictions,
    zero_division=0,
)

randomized_candidates = 9
randomized_total_fits = randomized_candidates * 5


# --------------------------------------------------
# 13. Untuned forest on the same seven features
# --------------------------------------------------

default_forest_7 = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=None,
    min_samples_leaf=1,
)

default_forest_7.fit(X_train_7, y_train)

default_7_predictions = default_forest_7.predict(X_val_7)

default_7_val_f1 = f1_score(
    y_val,
    default_7_predictions,
    zero_division=0,
)


tuning_rows = [
    {
        "search": "grid_search",
        "candidates": grid_candidates,
        "cv_folds": 5,
        "total_fits": grid_total_fits,
        "best_params": json.dumps(
            grid_search.best_params_,
            sort_keys=True,
        ),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1,
    },
    {
        "search": "randomized_search",
        "candidates": randomized_candidates,
        "cv_folds": 5,
        "total_fits": randomized_total_fits,
        "best_params": json.dumps(
            randomized_search.best_params_,
            sort_keys=True,
        ),
        "best_cv_accuracy": randomized_search.best_score_,
        "val_f1": randomized_val_f1,
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": default_7_val_f1,
    },
]

tuning_df = pd.DataFrame(
    tuning_rows,
    columns=[
        "search",
        "candidates",
        "cv_folds",
        "total_fits",
        "best_params",
        "best_cv_accuracy",
        "val_f1",
    ],
)

tuning_df.to_csv(
    OUTPUT_DIR / "part_c_tuning.csv",
    index=False,
    float_format="%.10f",
)

print("\nTuning comparison:")
print(tuning_df.to_string(index=False))


# --------------------------------------------------
# 14. Choose the winning configuration
# --------------------------------------------------

# Highest validation F1 wins.
# Tie preference: default forest, grid search, randomized search.
tie_priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2,
}

winning_row = min(
    tuning_rows,
    key=lambda row: (
        -row["val_f1"],
        tie_priority[row["search"]],
    ),
)

chosen_from = winning_row["search"]

if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_
elif chosen_from == "randomized_search":
    chosen_params = randomized_search.best_params_
else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
    }

final_model = Pipeline(
    [
        ("selector", SelectKBest(f_classif, k=7)),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                **chosen_params,
            ),
        ),
    ]
)

# Fit only on the training rows.
final_model.fit(X_train, y_train)

# Use validation probabilities to select the alert threshold.
val_probabilities = final_model.predict_proba(X_val)[:, 1]


# --------------------------------------------------
# 15. Alert threshold experiment
# --------------------------------------------------

thresholds = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9,
]

threshold_rows = []

for threshold in thresholds:
    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    metrics = calculate_classification_metrics(
        y_val,
        predictions,
    )

    threshold_rows.append(
        {
            "threshold": threshold,
            **metrics,
        }
    )

thresholds_df = pd.DataFrame(
    threshold_rows,
    columns=[
        "threshold",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ],
)

thresholds_df.to_csv(
    OUTPUT_DIR / "part_c_thresholds.csv",
    index=False,
    float_format="%.10f",
)

eligible_thresholds = thresholds_df[
    thresholds_df["recall"] >= 0.80
]

if eligible_thresholds.empty:
    chosen_threshold = 0.1
else:
    chosen_threshold = float(
        eligible_thresholds["threshold"].max()
    )

print(f"\nChosen alert threshold: {chosen_threshold}")


# --------------------------------------------------
# 16. Evaluate the final model on the test set once
# --------------------------------------------------

# Do not use the test set for model or threshold selection.
test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

test_metrics = calculate_classification_metrics(
    y_test,
    test_predictions,
)

final_test_df = pd.DataFrame(
    [
        {
            "threshold": chosen_threshold,
            **test_metrics,
        }
    ],
    columns=[
        "threshold",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ],
)

final_test_df.to_csv(
    OUTPUT_DIR / "part_c_final_test.csv",
    index=False,
    float_format="%.10f",
)


# --------------------------------------------------
# 17. Save the final configuration
# --------------------------------------------------

final_config_df = pd.DataFrame(
    [
        {
            "selector_k": 7,
            "n_estimators": chosen_params["n_estimators"],
            "max_depth": (
                "unlimited"
                if chosen_params["max_depth"] is None
                else chosen_params["max_depth"]
            ),
            "min_samples_leaf": chosen_params["min_samples_leaf"],
            "threshold": chosen_threshold,
            "chosen_from": chosen_from,
        }
    ],
    columns=[
        "selector_k",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "threshold",
        "chosen_from",
    ],
)

final_config_df.to_csv(
    OUTPUT_DIR / "part_c_final_config.csv",
    index=False,
)


# --------------------------------------------------
# 18. Print summary
# --------------------------------------------------

print("\nPart C completed successfully.")

print("\nValidation model comparison:")
print(models_df.to_string(index=False))

print("\nThreshold table:")
print(thresholds_df.to_string(index=False))

print("\nFinal test metrics:")
print(final_test_df.to_string(index=False))

print("\nFinal configuration:")
print(final_config_df.to_string(index=False))

print("\nFiles created:")
print("outputs/part_c_models.csv")
print("outputs/part_c_feature_scores.csv")
print("outputs/part_c_selectk.csv")
print("outputs/part_c_tuning.csv")
print("outputs/part_c_thresholds.csv")
print("outputs/part_c_final_test.csv")
print("outputs/part_c_final_config.csv")
