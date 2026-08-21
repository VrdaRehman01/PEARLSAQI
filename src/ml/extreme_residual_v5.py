import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# AQI EXTREME RESIDUAL CORRECTION - V5
# Conditional correction / gating experiment
# ============================================================

print("=" * 70)
print("AQI EXTREME RESIDUAL CORRECTION - V5")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

TRAIN_PATH = "data/processed/v3/train.parquet"
VAL_PATH = "data/processed/v3/validation.parquet"
TEST_PATH = "data/processed/v3/test.parquet"
BASE_MODEL_PATH = (
    "models/extreme_residual/base_xgboost_model.json"
)

RESIDUAL_MODEL_PATH = (
    "models/extreme_residual/extreme_residual_model.json"
)

CLASSIFIER_PATH = (
    "models/extreme_classifier/extreme_aqi_classifier.pkl"
)

OUTPUT_DIR = "models/extreme_residual/v5"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading data...")

train = pd.read_parquet(TRAIN_PATH)
validation = pd.read_parquet(VAL_PATH)
test = pd.read_parquet(TEST_PATH)

print(f"Training rows   : {len(train)}")
print(f"Validation rows : {len(validation)}")
print(f"Test rows       : {len(test)}")


# ------------------------------------------------------------
# FEATURE SELECTION
# ------------------------------------------------------------

exclude_columns = [
    "target_aqi",
    "date",
    "city_name",
]

features = [
    c for c in train.columns
    if c not in exclude_columns
]

print(f"\nNumber of features: {len(features)}")


X_train = train[features]
y_train = train["target_aqi"]

X_val = validation[features]
y_val = validation["target_aqi"]

X_test = test[features]
y_test = test["target_aqi"]


# ------------------------------------------------------------
# LOAD BASE XGBOOST
# ------------------------------------------------------------

print("\nLoading base XGBoost...")

import xgboost as xgb

base_model = xgb.XGBRegressor()
base_model.load_model(BASE_MODEL_PATH)

print("Base model loaded.")


# ------------------------------------------------------------
# LOAD RESIDUAL MODEL
# ------------------------------------------------------------

print("\nLoading residual model...")

residual_model = xgb.XGBRegressor()
residual_model.load_model(RESIDUAL_MODEL_PATH)

print("Residual model loaded.")


# ------------------------------------------------------------
# LOAD EXTREME CLASSIFIER
# ------------------------------------------------------------

print("\nLoading extreme classifier...")

classifier_bundle = joblib.load(CLASSIFIER_PATH)

classifier = classifier_bundle["model"]

print("Classifier loaded.")
print(f"Classifier type: {type(classifier).__name__}")


# ------------------------------------------------------------
# BASE PREDICTIONS
# ------------------------------------------------------------

print("\nGenerating predictions...")

val_base = base_model.predict(X_val)
test_base = base_model.predict(X_test)

val_probability = classifier.predict_proba(X_val)[:, 1]
test_probability = classifier.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# RESIDUAL PREDICTIONS
# ------------------------------------------------------------

print("Generating residual predictions...")

val_residual = residual_model.predict(X_val)
test_residual = residual_model.predict(X_test)


# ------------------------------------------------------------
# BASE METRICS
# ------------------------------------------------------------

def metrics(y_true, prediction):

    mae = mean_absolute_error(y_true, prediction)

    rmse = np.sqrt(
        mean_squared_error(y_true, prediction)
    )

    r2 = r2_score(y_true, prediction)

    return mae, rmse, r2


base_val_metrics = metrics(
    y_val,
    val_base
)

base_test_metrics = metrics(
    y_test,
    test_base
)


print("\nBASE XGBOOST - VALIDATION")

print(f"MAE  : {base_val_metrics[0]:.4f}")
print(f"RMSE : {base_val_metrics[1]:.4f}")
print(f"R²   : {base_val_metrics[2]:.4f}")


print("\nBASE XGBOOST - 2026 TEST")

print(f"MAE  : {base_test_metrics[0]:.4f}")
print(f"RMSE : {base_test_metrics[1]:.4f}")
print(f"R²   : {base_test_metrics[2]:.4f}")


# ============================================================
# CONDITIONAL CORRECTION STRATEGIES
# ============================================================

strategies = []


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def add_strategy(
    name,
    prediction,
    val_prediction
):

    test_mae, test_rmse, test_r2 = metrics(
        y_test,
        prediction
    )

    val_mae, val_rmse, val_r2 = metrics(
        y_val,
        val_prediction
    )

    strategies.append({
        "strategy": name,

        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_r2": val_r2,

        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
    })


# ============================================================
# 1. CURRENT FULL CORRECTION
# ============================================================

val_current = val_base + val_residual
test_current = test_base + test_residual

add_strategy(
    "Current_Full_Correction",
    test_current,
    val_current
)


# ============================================================
# 2. PREDICTED AQI GATES
# ============================================================

for threshold in [150, 175, 180, 190, 200, 210]:

    val_mask = val_base >= threshold
    test_mask = test_base >= threshold

    val_prediction = val_base.copy()
    test_prediction = test_base.copy()

    val_prediction[val_mask] += val_residual[val_mask]
    test_prediction[test_mask] += test_residual[test_mask]

    add_strategy(
        f"PredictedAQI_GreaterEqual_{threshold}",
        test_prediction,
        val_prediction
    )


# ============================================================
# 3. EXTREME PROBABILITY GATES
# ============================================================

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60]:

    val_mask = val_probability >= threshold
    test_mask = test_probability >= threshold

    val_prediction = val_base.copy()
    test_prediction = test_base.copy()

    val_prediction[val_mask] += val_residual[val_mask]
    test_prediction[test_mask] += test_residual[test_mask]

    add_strategy(
        f"Probability_GreaterEqual_{threshold}",
        test_prediction,
        val_prediction
    )


# ============================================================
# 4. PREDICTED AQI + EXTREME PROBABILITY
# ============================================================

for aqi_threshold in [175, 190, 200]:

    for probability_threshold in [0.20, 0.30, 0.40, 0.50]:

        val_mask = (
            (val_base >= aqi_threshold)
            &
            (val_probability >= probability_threshold)
        )

        test_mask = (
            (test_base >= aqi_threshold)
            &
            (test_probability >= probability_threshold)
        )

        val_prediction = val_base.copy()
        test_prediction = test_base.copy()

        val_prediction[val_mask] += val_residual[val_mask]
        test_prediction[test_mask] += test_residual[test_mask]

        add_strategy(
            f"AQI_{aqi_threshold}_Prob_{probability_threshold}",
            test_prediction,
            val_prediction
        )


# ============================================================
# 5. ONLY POSITIVE RESIDUAL CORRECTION
# ============================================================

for aqi_threshold in [175, 190, 200]:

    for probability_threshold in [0.20, 0.30, 0.40]:

        val_mask = (
            (val_base >= aqi_threshold)
            &
            (val_probability >= probability_threshold)
            &
            (val_residual > 0)
        )

        test_mask = (
            (test_base >= aqi_threshold)
            &
            (test_probability >= probability_threshold)
            &
            (test_residual > 0)
        )

        val_prediction = val_base.copy()
        test_prediction = test_base.copy()

        val_prediction[val_mask] += val_residual[val_mask]
        test_prediction[test_mask] += test_residual[test_mask]

        add_strategy(
            f"AQI_{aqi_threshold}_Prob_{probability_threshold}_PositiveResidual",
            test_prediction,
            val_prediction
        )


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(strategies)

results = results.sort_values(
    by="val_mae"
).reset_index(drop=True)


print("\n")
print("=" * 70)
print("V5 STRATEGY RESULTS")
print("=" * 70)

print(
    results[
        [
            "strategy",
            "val_mae",
            "val_rmse",
            "val_r2",
            "test_mae",
            "test_rmse",
            "test_r2",
        ]
    ].to_string(index=False)
)


# ============================================================
# BEST VALIDATION STRATEGY
# ============================================================

best = results.iloc[0]

print("\n")
print("=" * 70)
print("BEST V5 STRATEGY")
print("=" * 70)

print(f"Strategy : {best['strategy']}")

print("\nValidation:")
print(f"MAE      : {best['val_mae']:.4f}")
print(f"RMSE     : {best['val_rmse']:.4f}")
print(f"R²       : {best['val_r2']:.4f}")

print("\n2026 Holdout:")
print(f"MAE      : {best['test_mae']:.4f}")
print(f"RMSE     : {best['test_rmse']:.4f}")
print(f"R²       : {best['test_r2']:.4f}")


# ============================================================
# COMPARE WITH BASE
# ============================================================

print("\n")
print("=" * 70)
print("IMPROVEMENT VS BASE XGBOOST")
print("=" * 70)

print(
    f"MAE improvement  : "
    f"{base_test_metrics[0] - best['test_mae']:+.4f}"
)

print(
    f"RMSE improvement : "
    f"{base_test_metrics[1] - best['test_rmse']:+.4f}"
)

print(
    f"R² improvement   : "
    f"{best['test_r2'] - base_test_metrics[2]:+.4f}"
)


# ============================================================
# EXTREME TEST PERFORMANCE
# ============================================================

extreme_mask = y_test > 200

print("\n")
print("=" * 70)
print("EXTREME EVENT PERFORMANCE")
print("=" * 70)

print(f"Extreme rows: {extreme_mask.sum()}")


# Reconstruct best test prediction

best_strategy = best["strategy"]


def apply_strategy(
    strategy,
    base,
    residual,
    probability
):

    prediction = base.copy()

    if strategy == "Current_Full_Correction":

        prediction += residual

    elif strategy.startswith(
        "PredictedAQI_GreaterEqual_"
    ):

        threshold = float(
            strategy.split("_")[-1]
        )

        mask = base >= threshold

        prediction[mask] += residual[mask]

    elif strategy.startswith(
        "Probability_GreaterEqual_"
    ):

        threshold = float(
            strategy.split("_")[-1]
        )

        mask = probability >= threshold

        prediction[mask] += residual[mask]

    else:

        parts = strategy.split("_")

        aqi_threshold = float(
            parts[1]
        )

        probability_threshold = float(
            parts[3]
        )

        positive_residual = (
            "PositiveResidual" in strategy
        )

        mask = (
            (base >= aqi_threshold)
            &
            (probability >= probability_threshold)
        )

        if positive_residual:

            mask &= residual > 0

        prediction[mask] += residual[mask]

    return prediction


best_prediction = apply_strategy(
    best_strategy,
    test_base,
    test_residual,
    test_probability
)


base_extreme_mae = mean_absolute_error(
    y_test[extreme_mask],
    test_base[extreme_mask]
)

best_extreme_mae = mean_absolute_error(
    y_test[extreme_mask],
    best_prediction[extreme_mask]
)

base_extreme_rmse = np.sqrt(
    mean_squared_error(
        y_test[extreme_mask],
        test_base[extreme_mask]
    )
)

best_extreme_rmse = np.sqrt(
    mean_squared_error(
        y_test[extreme_mask],
        best_prediction[extreme_mask]
    )
)

print("\nBase XGBoost:")
print(f"MAE  : {base_extreme_mae:.4f}")
print(f"RMSE : {base_extreme_rmse:.4f}")

print("\nBest V5:")
print(f"MAE  : {best_extreme_mae:.4f}")
print(f"RMSE : {best_extreme_rmse:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    f"{OUTPUT_DIR}/v5_strategy_results.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------

prediction_output = test[
    [
        "city_name",
        "date",
        "aqi",
        "target_aqi",
    ]
].copy()

prediction_output["base_prediction"] = test_base
prediction_output["extreme_probability"] = test_probability
prediction_output["residual"] = test_residual
prediction_output["v5_prediction"] = best_prediction
prediction_output["absolute_error"] = np.abs(
    y_test.values - best_prediction
)

prediction_output.to_parquet(
    f"{OUTPUT_DIR}/v5_predictions.parquet",
    index=False
)


# ------------------------------------------------------------
# Save metadata
# ------------------------------------------------------------

metadata = {
    "best_strategy": best_strategy,

    "validation": {
        "mae": float(best["val_mae"]),
        "rmse": float(best["val_rmse"]),
        "r2": float(best["val_r2"]),
    },

    "test": {
        "mae": float(best["test_mae"]),
        "rmse": float(best["test_rmse"]),
        "r2": float(best["test_r2"]),
    },

    "extreme_test": {
        "base_mae": float(base_extreme_mae),
        "best_mae": float(best_extreme_mae),
        "base_rmse": float(base_extreme_rmse),
        "best_rmse": float(best_extreme_rmse),
    }
}

with open(
    f"{OUTPUT_DIR}/v5_metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    f"Results      : {results_path}"
)

print(
    f"Predictions  : "
    f"{OUTPUT_DIR}/v5_predictions.parquet"
)

print(
    f"Metadata     : "
    f"{OUTPUT_DIR}/v5_metadata.json"
)

print("\nV5 completed successfully.")