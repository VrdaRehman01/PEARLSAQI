import os
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ============================================================
# FINAL WALK-FORWARD ENSEMBLE VALIDATION
# HGB 40% + XGB 40% + RIDGE 20%
# ============================================================

print("=" * 70)
print("FINAL WALK-FORWARD ENSEMBLE VALIDATION")
print("HGB 40% + XGBoost 40% + Ridge 20%")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

TRAIN_PATH = "data/processed/v3/train.parquet"
VAL_PATH = "data/processed/v3/validation.parquet"
TEST_PATH = "data/processed/v3/test.parquet"

OUTPUT_DIR = "models/final_walk_forward_ensemble"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading datasets...")

train = pd.read_parquet(TRAIN_PATH)
validation = pd.read_parquet(VAL_PATH)
test = pd.read_parquet(TEST_PATH)

print(f"Training rows    : {len(train)}")
print(f"Validation rows : {len(validation)}")
print(f"Test rows       : {len(test)}")


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

exclude = [
    "target_aqi",
    "date",
    "city_name",
]

features = [
    c for c in train.columns
    if c not in exclude
]

print(f"Number of features: {len(features)}")


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def calculate_metrics(y_true, prediction):

    mae = mean_absolute_error(
        y_true,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            prediction
        )
    )

    r2 = r2_score(
        y_true,
        prediction
    )

    within_10 = (
        np.abs(
            y_true - prediction
        ) <= 10
    ).mean() * 100

    within_20 = (
        np.abs(
            y_true - prediction
        ) <= 20
    ).mean() * 100

    within_30 = (
        np.abs(
            y_true - prediction
        ) <= 30
    ).mean() * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "within_10": within_10,
        "within_20": within_20,
        "within_30": within_30,
    }


# ------------------------------------------------------------
# MODEL FACTORIES
# ------------------------------------------------------------

def create_hgb():

    return HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=500,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=5.0,
        random_state=42,
    )


def create_xgb():

    return XGBRegressor(
        n_estimators=800,
        learning_rate=0.025,
        max_depth=4,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=5.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


def create_ridge():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=0.01)
        )
    ])


# ------------------------------------------------------------
# FOLD FUNCTION
# ------------------------------------------------------------

def run_fold(
    fold_name,
    train_df,
    eval_df,
):

    print("\n")
    print("=" * 70)
    print(f"FOLD: {fold_name}")
    print("=" * 70)

    X_train = train_df[features]
    y_train = train_df["target_aqi"]

    X_eval = eval_df[features]
    y_eval = eval_df["target_aqi"]

    print(
        f"Training period: "
        f"{train_df['date'].min()} → "
        f"{train_df['date'].max()}"
    )

    print(
        f"Evaluation period: "
        f"{eval_df['date'].min()} → "
        f"{eval_df['date'].max()}"
    )

    print(f"Training rows: {len(train_df)}")
    print(f"Evaluation rows: {len(eval_df)}")

    # --------------------------------------------------------
    # HGB
    # --------------------------------------------------------

    print("\nTraining HistGradientBoosting...")

    start = time.time()

    hgb = create_hgb()
    hgb.fit(X_train, y_train)

    hgb_time = time.time() - start

    hgb_prediction = hgb.predict(X_eval)

    hgb_metrics = calculate_metrics(
        y_eval,
        hgb_prediction
    )

    print(
        f"HGB  MAE       : {hgb_metrics['mae']:.4f}"
    )

    print(
        f"HGB  RMSE      : {hgb_metrics['rmse']:.4f}"
    )

    print(
        f"HGB  R²        : {hgb_metrics['r2']:.4f}"
    )

    # --------------------------------------------------------
    # XGB
    # --------------------------------------------------------

    print("\nTraining XGBoost...")

    start = time.time()

    xgb = create_xgb()
    xgb.fit(X_train, y_train)

    xgb_time = time.time() - start

    xgb_prediction = xgb.predict(X_eval)

    xgb_metrics = calculate_metrics(
        y_eval,
        xgb_prediction
    )

    print(
        f"XGB  MAE       : {xgb_metrics['mae']:.4f}"
    )

    print(
        f"XGB  RMSE      : {xgb_metrics['rmse']:.4f}"
    )

    print(
        f"XGB  R²        : {xgb_metrics['r2']:.4f}"
    )

    # --------------------------------------------------------
    # RIDGE
    # --------------------------------------------------------

    print("\nTraining Ridge...")

    start = time.time()

    ridge = create_ridge()
    ridge.fit(X_train, y_train)

    ridge_time = time.time() - start

    ridge_prediction = ridge.predict(X_eval)

    ridge_metrics = calculate_metrics(
        y_eval,
        ridge_prediction
    )

    print(
        f"Ridge MAE      : {ridge_metrics['mae']:.4f}"
    )

    print(
        f"Ridge RMSE     : {ridge_metrics['rmse']:.4f}"
    )

    print(
        f"Ridge R²       : {ridge_metrics['r2']:.4f}"
    )

    # --------------------------------------------------------
    # FINAL ENSEMBLE
    # --------------------------------------------------------

    ensemble_prediction = (
        0.40 * hgb_prediction
        +
        0.40 * xgb_prediction
        +
        0.20 * ridge_prediction
    )

    ensemble_metrics = calculate_metrics(
        y_eval,
        ensemble_prediction
    )

    print("\nFINAL ENSEMBLE 40/40/20")

    print(
        f"MAE       : "
        f"{ensemble_metrics['mae']:.4f}"
    )

    print(
        f"RMSE      : "
        f"{ensemble_metrics['rmse']:.4f}"
    )

    print(
        f"R²        : "
        f"{ensemble_metrics['r2']:.4f}"
    )

    print(
        f"Within ±10: "
        f"{ensemble_metrics['within_10']:.2f}%"
    )

    print(
        f"Within ±20: "
        f"{ensemble_metrics['within_20']:.2f}%"
    )

    print(
        f"Within ±30: "
        f"{ensemble_metrics['within_30']:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    predictions = eval_df[
        [
            "city_name",
            "date",
            "aqi",
            "target_aqi",
        ]
    ].copy()

    predictions["hgb_prediction"] = hgb_prediction
    predictions["xgb_prediction"] = xgb_prediction
    predictions["ridge_prediction"] = ridge_prediction
    predictions["ensemble_prediction"] = ensemble_prediction

    predictions["absolute_error"] = np.abs(
        y_eval.values - ensemble_prediction
    )

    safe_name = fold_name.replace(
        " ",
        "_"
    )

    predictions.to_parquet(
        f"{OUTPUT_DIR}/{safe_name}_predictions.parquet",
        index=False
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    rows = []

    for model_name, model_metrics, train_time in [
        (
            "HGB",
            hgb_metrics,
            hgb_time
        ),
        (
            "XGB",
            xgb_metrics,
            xgb_time
        ),
        (
            "Ridge",
            ridge_metrics,
            ridge_time
        ),
        (
            "HGB_XGB_Ridge_40_40_20",
            ensemble_metrics,
            hgb_time + xgb_time + ridge_time
        ),
    ]:

        rows.append({
            "fold": fold_name,
            "model": model_name,
            "mae": model_metrics["mae"],
            "rmse": model_metrics["rmse"],
            "r2": model_metrics["r2"],
            "within_10": model_metrics["within_10"],
            "within_20": model_metrics["within_20"],
            "within_30": model_metrics["within_30"],
            "training_time_seconds": train_time,
        })

    return rows


# ============================================================
# WALK-FORWARD FOLDS
# ============================================================

all_results = []


# ------------------------------------------------------------
# FOLD 1
# 2023 → 2024
# ------------------------------------------------------------

fold1_train = train[
    train["date"] <= "2023-12-31"
].copy()

fold1_eval = train[
    (train["date"] >= "2024-01-01")
    &
    (train["date"] <= "2024-12-31")
].copy()

all_results.extend(
    run_fold(
        "2023_train_2024_validate",
        fold1_train,
        fold1_eval,
    )
)


# ------------------------------------------------------------
# FOLD 2
# 2023-2024 → 2025
# ------------------------------------------------------------

fold2_train = train[
    train["date"] <= "2024-12-31"
].copy()

fold2_eval = validation.copy()

all_results.extend(
    run_fold(
        "2023_2024_train_2025_validate",
        fold2_train,
        fold2_eval,
    )
)


# ------------------------------------------------------------
# FOLD 3
# 2023-2025 → 2026
# ------------------------------------------------------------

fold3_train = pd.concat(
    [
        train,
        validation,
    ],
    ignore_index=True
)

fold3_eval = test.copy()

all_results.extend(
    run_fold(
        "2023_2025_train_2026_test",
        fold3_train,
        fold3_eval,
    )
)


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    all_results
)

results.to_csv(
    f"{OUTPUT_DIR}/final_walk_forward_results.csv",
    index=False
)


print("\n")
print("=" * 70)
print("FINAL WALK-FORWARD RESULTS")
print("=" * 70)

print(
    results[
        [
            "fold",
            "model",
            "mae",
            "rmse",
            "r2",
            "within_10",
            "within_20",
            "within_30",
        ]
    ].to_string(index=False)
)


# ============================================================
# AVERAGE MODEL PERFORMANCE
# ============================================================

average = (
    results
    .groupby("model")[
        [
            "mae",
            "rmse",
            "r2",
            "within_10",
            "within_20",
            "within_30",
        ]
    ]
    .mean()
    .sort_values("mae")
)


print("\n")
print("=" * 70)
print("AVERAGE WALK-FORWARD PERFORMANCE")
print("=" * 70)

print(
    average.to_string()
)


# ============================================================
# ENSEMBLE VS XGB
# ============================================================

print("\n")
print("=" * 70)
print("ENSEMBLE VS XGBOOST")
print("=" * 70)

xgb_avg = average.loc["XGB"]
ensemble_avg = average.loc[
    "HGB_XGB_Ridge_40_40_20"
]

print(
    f"\nXGB average MAE     : "
    f"{xgb_avg['mae']:.4f}"
)

print(
    f"Ensemble average MAE: "
    f"{ensemble_avg['mae']:.4f}"
)

print(
    f"\nXGB average RMSE     : "
    f"{xgb_avg['rmse']:.4f}"
)

print(
    f"Ensemble average RMSE: "
    f"{ensemble_avg['rmse']:.4f}"
)

print(
    f"\nXGB average R²       : "
    f"{xgb_avg['r2']:.4f}"
)

print(
    f"Ensemble average R²  : "
    f"{ensemble_avg['r2']:.4f}"
)


# ============================================================
# DECISION
# ============================================================

if ensemble_avg["mae"] < xgb_avg["mae"]:

    winner = "HGB_XGB_Ridge_40_40_20"

else:

    winner = "XGB"


print("\n")
print("=" * 70)
print("FINAL MODEL DECISION")
print("=" * 70)

print(f"Winner based on average walk-forward MAE: {winner}")


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "winner": winner,

    "xgb_average": {
        "mae": float(xgb_avg["mae"]),
        "rmse": float(xgb_avg["rmse"]),
        "r2": float(xgb_avg["r2"]),
        "within_10": float(xgb_avg["within_10"]),
        "within_20": float(xgb_avg["within_20"]),
        "within_30": float(xgb_avg["within_30"]),
    },

    "ensemble_average": {
        "mae": float(ensemble_avg["mae"]),
        "rmse": float(ensemble_avg["rmse"]),
        "r2": float(ensemble_avg["r2"]),
        "within_10": float(ensemble_avg["within_10"]),
        "within_20": float(ensemble_avg["within_20"]),
        "within_30": float(ensemble_avg["within_30"]),
    },
}

import json

with open(
    f"{OUTPUT_DIR}/final_model_decision.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print("\nResults saved to:")

print(
    f"{OUTPUT_DIR}/final_walk_forward_results.csv"
)

print(
    f"{OUTPUT_DIR}/final_model_decision.json"
)

print("\nFINAL WALK-FORWARD VALIDATION COMPLETED.")