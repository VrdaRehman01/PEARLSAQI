"""
FINAL PRODUCTION XGBOOST MODEL

Training:
    2023-2025

Final holdout:
    2026

The 2026 test set remains untouched during model training.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = Path("data/processed/v4/train.parquet")
VALIDATION_PATH = Path("data/processed/v4/validation.parquet")
TEST_PATH = Path("data/processed/v4/test.parquet")

MODEL_DIR = Path("models/final_production_xgboost")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "final_xgboost_model.json"
FEATURE_PATH = MODEL_DIR / "features.json"
METRICS_PATH = MODEL_DIR / "final_metrics.json"
PREDICTIONS_PATH = MODEL_DIR / "test_predictions.parquet"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "target_aqi"

# Selected through hyperparameter tuning + walk-forward validation
PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.025,
    "max_depth": 4,
    "min_child_weight": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, prediction):

    mae = mean_absolute_error(y_true, prediction)

    rmse = np.sqrt(
        mean_squared_error(y_true, prediction)
    )

    r2 = r2_score(y_true, prediction)

    within_10 = np.mean(
        np.abs(y_true - prediction) <= 10
    ) * 100

    within_20 = np.mean(
        np.abs(y_true - prediction) <= 20
    ) * 100

    within_30 = np.mean(
        np.abs(y_true - prediction) <= 30
    ) * 100

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "within_10": float(within_10),
        "within_20": float(within_20),
        "within_30": float(within_30),
    }


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("FINAL PRODUCTION XGBOOST MODEL")
print("=" * 70)

print("\nLoading datasets...")

train = pd.read_parquet(TRAIN_PATH)
validation = pd.read_parquet(VALIDATION_PATH)
test = pd.read_parquet(TEST_PATH)

print(f"Training rows     : {len(train)}")
print(f"Validation rows   : {len(validation)}")
print(f"Test rows         : {len(test)}")

print(
    f"\nTrain period      : "
    f"{train['date'].min()} → {train['date'].max()}"
)

print(
    f"Validation period : "
    f"{validation['date'].min()} → {validation['date'].max()}"
)

print(
    f"Test period       : "
    f"{test['date'].min()} → {test['date'].max()}"
)


# ============================================================
# COMBINE 2023-2025 DATA
# ============================================================

print("\nCombining training + validation data...")

development = pd.concat(
    [train, validation],
    ignore_index=True
)

development = development.sort_values(
    ["date", "city_name"]
).reset_index(drop=True)

print(f"Final training rows: {len(development)}")

print(
    f"Final training period: "
    f"{development['date'].min()} → "
    f"{development['date'].max()}"
)


# ============================================================
# FEATURE LIST
# ============================================================

features = [
    column
    for column in development.columns
    if column not in [
        TARGET,
        "date",
        "city_name",
    ]
]

# Ensure identical feature columns
missing_test = [
    feature
    for feature in features
    if feature not in test.columns
]

if missing_test:
    raise ValueError(
        f"Missing test features: {missing_test}"
    )

print(f"\nNumber of features: {len(features)}")

for i, feature in enumerate(features, 1):
    print(f"{i:03d}. {feature}")


# ============================================================
# PREPARE MATRICES
# ============================================================

X_train = development[features]
y_train = development[TARGET]

X_test = test[features]
y_test = test[TARGET]


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL XGBOOST")
print("=" * 70)

print("\nHyperparameters:")

for key, value in PARAMS.items():
    print(f"{key:20s}: {value}")

start_time = time.time()

model = xgb.XGBRegressor(
    **PARAMS
)

model.fit(
    X_train,
    y_train,
    verbose=False
)

training_time = time.time() - start_time

print(
    f"\nTraining completed in "
    f"{training_time:.2f} seconds"
)


# ============================================================
# TRAINING PERFORMANCE
# ============================================================

train_prediction = model.predict(X_train)

train_metrics = calculate_metrics(
    y_train,
    train_prediction
)

print("\nFINAL MODEL - TRAINING")
print(f"MAE       : {train_metrics['mae']:.4f}")
print(f"RMSE      : {train_metrics['rmse']:.4f}")
print(f"R²        : {train_metrics['r2']:.4f}")
print(f"Within ±10: {train_metrics['within_10']:.2f}%")
print(f"Within ±20: {train_metrics['within_20']:.2f}%")
print(f"Within ±30: {train_metrics['within_30']:.2f}%")


# ============================================================
# FINAL 2026 HOLDOUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL 2026 HOLDOUT EVALUATION")
print("=" * 70)

test_prediction = model.predict(X_test)

test_metrics = calculate_metrics(
    y_test,
    test_prediction
)

print("\nFINAL MODEL - 2026 HOLDOUT")

print(f"MAE       : {test_metrics['mae']:.4f}")
print(f"RMSE      : {test_metrics['rmse']:.4f}")
print(f"R²        : {test_metrics['r2']:.4f}")
print(f"Within ±10: {test_metrics['within_10']:.2f}%")
print(f"Within ±20: {test_metrics['within_20']:.2f}%")
print(f"Within ±30: {test_metrics['within_30']:.2f}%")


# ============================================================
# PREDICTIONS DATAFRAME
# ============================================================

predictions = test[
    [
        "city_name",
        "date",
        "aqi",
        TARGET,
    ]
].copy()

predictions["prediction"] = test_prediction

predictions["absolute_error"] = (
    np.abs(
        predictions[TARGET]
        - predictions["prediction"]
    )
)

predictions["error"] = (
    predictions["prediction"]
    - predictions[TARGET]
)

predictions = predictions.sort_values(
    "absolute_error",
    ascending=False
)


# ============================================================
# WORST PREDICTIONS
# ============================================================

print("\nTOP 20 WORST PREDICTIONS")

print(
    predictions[
        [
            "city_name",
            "date",
            "aqi",
            TARGET,
            "prediction",
            "absolute_error",
            "error",
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save_model(MODEL_PATH)

print(
    f"\nModel saved: {MODEL_PATH}"
)


# ============================================================
# SAVE FEATURES
# ============================================================

with open(FEATURE_PATH, "w") as f:

    json.dump(
        {
            "features": features,
            "feature_count": len(features),
        },
        f,
        indent=4
    )

print(
    f"Features saved: {FEATURE_PATH}"
)


# ============================================================
# SAVE METADATA / METRICS
# ============================================================

metadata = {
    "model": "XGBoost",
    "model_type": "final_production_regressor",

    "training_start":
        str(development["date"].min()),

    "training_end":
        str(development["date"].max()),

    "test_start":
        str(test["date"].min()),

    "test_end":
        str(test["date"].max()),

    "training_rows":
        int(len(development)),

    "test_rows":
        int(len(test)),

    "feature_count":
        len(features),

    "training_time_seconds":
        float(training_time),

    "hyperparameters":
        PARAMS,

    "training_metrics":
        train_metrics,

    "test_metrics":
        test_metrics,
}

with open(METRICS_PATH, "w") as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

print(
    f"Metrics saved: {METRICS_PATH}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions.to_parquet(
    PREDICTIONS_PATH,
    index=False
)

print(
    f"Predictions saved: {PREDICTIONS_PATH}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PRODUCTION MODEL READY")
print("=" * 70)

print("\nModel       : XGBoost")
print(f"Features    : {len(features)}")
print(
    f"Train rows  : {len(development)}"
)
print(
    f"Test rows   : {len(test)}"
)

print("\n2026 HOLDOUT")
print(
    f"MAE         : "
    f"{test_metrics['mae']:.4f}"
)

print(
    f"RMSE        : "
    f"{test_metrics['rmse']:.4f}"
)

print(
    f"R²          : "
    f"{test_metrics['r2']:.4f}"
)

print(
    f"Within ±10  : "
    f"{test_metrics['within_10']:.2f}%"
)

print(
    f"Within ±20  : "
    f"{test_metrics['within_20']:.2f}%"
)

print(
    f"Within ±30  : "
    f"{test_metrics['within_30']:.2f}%"
)

print("\nProduction model:")
print(MODEL_PATH)

print("\nFINAL PRODUCTION TRAINING COMPLETED.")
print("=" * 70)