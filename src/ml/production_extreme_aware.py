"""
Production Extreme-Aware AQI Pipeline

Uses:
1. Base XGBoost regression model
2. Extreme AQI classifier
3. Extreme residual correction model

IMPORTANT:
The production correction decision is based ONLY on the
classifier probability, never on target_aqi.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = Path("data/processed/v3/train.parquet")
VALIDATION_PATH = Path("data/processed/v3/validation.parquet")
TEST_PATH = Path("data/processed/v3/test.parquet")

BASE_MODEL_PATH = Path(
    "models/extreme_residual/base_xgboost_model.json"
)

CLASSIFIER_PATH = Path(
    "models/extreme_classifier/extreme_aqi_classifier.pkl"
)

RESIDUAL_MODEL_PATH = Path(
    "models/extreme_residual/extreme_residual_model.json"
)

OUTPUT_DIR = Path("models/production_extreme_aware")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIER_THRESHOLD = 0.40
CORRECTION_STRENGTH = 1.0

TARGET = "target_aqi"


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
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "within_10": within_10,
        "within_20": within_20,
        "within_30": within_30,
    }


def print_metrics(name, y_true, prediction):
    metrics = calculate_metrics(y_true, prediction)

    print(f"\n{name}")

    print(f"MAE       : {metrics['mae']:.4f}")
    print(f"RMSE      : {metrics['rmse']:.4f}")
    print(f"R²        : {metrics['r2']:.4f}")
    print(f"Within ±10: {metrics['within_10']:.2f}%")
    print(f"Within ±20: {metrics['within_20']:.2f}%")
    print(f"Within ±30: {metrics['within_30']:.2f}%")

    return metrics


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("PRODUCTION EXTREME-AWARE AQI PIPELINE")
print("=" * 60)

print("\nLoading data...")

train = pd.read_parquet(TRAIN_PATH)
validation = pd.read_parquet(VALIDATION_PATH)
test = pd.read_parquet(TEST_PATH)

print(f"Training rows   : {len(train)}")
print(f"Validation rows : {len(validation)}")
print(f"Test rows       : {len(test)}")


# ============================================================
# BUILD FEATURE LIST
# ============================================================

excluded_columns = {
    TARGET,
    "city_name",
    "date",
}

feature_columns = [
    c for c in train.columns
    if c not in excluded_columns
]

print(f"\nNumber of features: {len(feature_columns)}")


X_train = train[feature_columns]
y_train = train[TARGET]

X_validation = validation[feature_columns]
y_validation = validation[TARGET]

X_test = test[feature_columns]
y_test = test[TARGET]


# ============================================================
# LOAD BASE XGBOOST
# ============================================================

print("\nLoading base XGBoost...")

base_model = xgb.XGBRegressor()

base_model.load_model(str(BASE_MODEL_PATH))

print("Base model loaded.")


# ============================================================
# LOAD EXTREME CLASSIFIER
# ============================================================

print("\nLoading extreme-event classifier...")

classifier_bundle = joblib.load(CLASSIFIER_PATH)

classifier = classifier_bundle["model"]
classifier_features = classifier_bundle["features"]
saved_classifier_threshold = classifier_bundle["threshold"]

print("Classifier loaded.")
print(f"Classifier type: {type(classifier).__name__}")
print(f"Classifier features: {len(classifier_features)}")
print(f"Saved classifier threshold: {saved_classifier_threshold}")
print(f"Production threshold: {CLASSIFIER_THRESHOLD}")


# ============================================================
# LOAD RESIDUAL MODEL
# ============================================================

print("\nLoading extreme residual model...")

residual_model = xgb.XGBRegressor()

residual_model.load_model(str(RESIDUAL_MODEL_PATH))

print("Residual model loaded.")
print(f"Correction strength: {CORRECTION_STRENGTH}")


# ============================================================
# GENERATE BASE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

base_validation = base_model.predict(X_validation)
base_test = base_model.predict(X_test)


# ============================================================
# EXTREME PROBABILITIES
# ============================================================

print("Generating extreme probabilities...")

# Use exactly the features the classifier was trained on.
X_validation_classifier = X_validation[classifier_features]
X_test_classifier = X_test[classifier_features]

validation_probability = classifier.predict_proba(
    X_validation_classifier
)[:, 1]

test_probability = classifier.predict_proba(
    X_test_classifier
)[:, 1]


# ============================================================
# PRODUCTION CORRECTION
# ============================================================

print("\nApplying production correction...")

# IMPORTANT:
# We ONLY use classifier probability here.
#
# We DO NOT use:
# target_aqi
# actual future AQI
# future information
#
# This makes the pipeline production-safe.

validation_extreme = (
    validation_probability >= CLASSIFIER_THRESHOLD
)

test_extreme = (
    test_probability >= CLASSIFIER_THRESHOLD
)


# Residual model predicts correction amount.
validation_residual = residual_model.predict(
    X_validation
)

test_residual = residual_model.predict(
    X_test
)


validation_corrected = base_validation.copy()

test_corrected = base_test.copy()


validation_corrected[validation_extreme] += (
    CORRECTION_STRENGTH
    * validation_residual[validation_extreme]
)

test_corrected[test_extreme] += (
    CORRECTION_STRENGTH
    * test_residual[test_extreme]
)


# ============================================================
# BASE RESULTS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print_metrics(
    "BASE XGBOOST",
    y_validation,
    base_validation,
)

print_metrics(
    "EXTREME-AWARE PRODUCTION",
    y_validation,
    validation_corrected,
)


# ============================================================
# 2026 TEST RESULTS
# ============================================================

print("\n" + "=" * 60)
print("2026 HOLDOUT TEST RESULTS")
print("=" * 60)

base_metrics = print_metrics(
    "BASE XGBOOST",
    y_test,
    base_test,
)

corrected_metrics = print_metrics(
    "EXTREME-AWARE PRODUCTION",
    y_test,
    test_corrected,
)


# ============================================================
# IMPROVEMENT
# ============================================================

print("\n" + "=" * 60)
print("IMPROVEMENT")
print("=" * 60)

print(
    f"MAE improvement : "
    f"{base_metrics['mae'] - corrected_metrics['mae']:+.4f}"
)

print(
    f"RMSE improvement: "
    f"{base_metrics['rmse'] - corrected_metrics['rmse']:+.4f}"
)

print(
    f"R² improvement  : "
    f"{corrected_metrics['r2'] - base_metrics['r2']:+.4f}"
)


# ============================================================
# EXTREME TEST PERFORMANCE
# ============================================================

actual_extreme = y_test.values > 200

print("\n" + "=" * 60)
print("EXTREME-EVENT TEST PERFORMANCE")
print("=" * 60)

print(
    f"Actual extreme rows (>200): "
    f"{actual_extreme.sum()}"
)

if actual_extreme.sum() > 0:

    base_extreme_mae = mean_absolute_error(
        y_test.values[actual_extreme],
        base_test[actual_extreme],
    )

    corrected_extreme_mae = mean_absolute_error(
        y_test.values[actual_extreme],
        test_corrected[actual_extreme],
    )

    base_extreme_rmse = np.sqrt(
        mean_squared_error(
            y_test.values[actual_extreme],
            base_test[actual_extreme],
        )
    )

    corrected_extreme_rmse = np.sqrt(
        mean_squared_error(
            y_test.values[actual_extreme],
            test_corrected[actual_extreme],
        )
    )

    print(
        f"\nBASE XGBOOST"
    )

    print(
        f"Extreme MAE : {base_extreme_mae:.4f}"
    )

    print(
        f"Extreme RMSE: {base_extreme_rmse:.4f}"
    )

    print(
        f"\nEXTREME-AWARE"
    )

    print(
        f"Extreme MAE : {corrected_extreme_mae:.4f}"
    )

    print(
        f"Extreme RMSE: {corrected_extreme_rmse:.4f}"
    )


# ============================================================
# CLASSIFIER PERFORMANCE
# ============================================================

predicted_extreme = test_extreme

true_extreme = y_test.values > 200

tp = np.sum(
    predicted_extreme & true_extreme
)

fp = np.sum(
    predicted_extreme & ~true_extreme
)

fn = np.sum(
    ~predicted_extreme & true_extreme
)

tn = np.sum(
    ~predicted_extreme & ~true_extreme
)

print("\n" + "=" * 60)
print("EXTREME CLASSIFIER ON 2026 TEST")
print("=" * 60)

print(f"Threshold : {CLASSIFIER_THRESHOLD}")
print(f"TP        : {tp}")
print(f"FP        : {fp}")
print(f"FN        : {fn}")
print(f"TN        : {tn}")

if tp + fp > 0:
    precision = tp / (tp + fp)
else:
    precision = 0

if tp + fn > 0:
    recall = tp / (tp + fn)
else:
    recall = 0

if precision + recall > 0:
    f1 = (
        2 * precision * recall
        / (precision + recall)
    )
else:
    f1 = 0

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1        : {f1:.4f}")


# ============================================================
# PREDICTION TABLE
# ============================================================

results = test[
    [
        "city_name",
        "date",
        "aqi",
    ]
].copy()

results["target_aqi"] = y_test.values

results["base_prediction"] = base_test

results["extreme_probability"] = test_probability

results["extreme_predicted"] = test_extreme

results["residual_correction"] = (
    np.where(
        test_extreme,
        test_residual * CORRECTION_STRENGTH,
        0,
    )
)

results["corrected_prediction"] = test_corrected

results["base_absolute_error"] = (
    np.abs(
        results["target_aqi"]
        - results["base_prediction"]
    )
)

results["corrected_absolute_error"] = (
    np.abs(
        results["target_aqi"]
        - results["corrected_prediction"]
    )
)

results["improvement"] = (
    results["base_absolute_error"]
    - results["corrected_absolute_error"]
)


# ============================================================
# WORST PREDICTIONS
# ============================================================

worst = results.sort_values(
    "corrected_absolute_error",
    ascending=False,
).head(30)

print("\n" + "=" * 60)
print("TOP 30 WORST CORRECTED PREDICTIONS")
print("=" * 60)

print(
    worst[
        [
            "city_name",
            "date",
            "aqi",
            "target_aqi",
            "base_prediction",
            "extreme_probability",
            "corrected_prediction",
            "corrected_absolute_error",
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

results_path = (
    OUTPUT_DIR
    / "production_predictions.parquet"
)

metrics_path = (
    OUTPUT_DIR
    / "production_metrics.csv"
)

results.to_parquet(
    results_path,
    index=False,
)

comparison = pd.DataFrame(
    [
        {
            "model": "Base XGBoost",
            **base_metrics,
        },
        {
            "model": "Extreme-Aware Production",
            **corrected_metrics,
        },
    ]
)

comparison.to_csv(
    metrics_path,
    index=False,
)


print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)

print(
    f"Predictions: {results_path}"
)

print(
    f"Metrics     : {metrics_path}"
)

print("\nPipeline completed successfully.")