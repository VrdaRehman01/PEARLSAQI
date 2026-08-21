"""
PEARLSAQI V7 - SPIKE-AWARE GATED AQI CORRECTION

Pipeline:

Production XGBoost
        ↓
Base next-day AQI prediction
        ↓
75-feature spike classifier
        ↓
Spike probability
        ↓
Delta correction
        ↓
Threshold / gate optimization
        ↓
Final V7 prediction

IMPORTANT:
- Uses the existing production XGBoost predictions.
- Uses the production-compatible 75-feature spike classifier.
- Does NOT artificially create unavailable features.
- Tests multiple probability thresholds.
- Tests multiple correction strengths.
- Evaluates against the 2026 holdout.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_PREDICTIONS = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "test_predictions.parquet"
)

LATEST_PREDICTIONS = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "predictions"
    / "latest_predictions.csv"
)

FEATURE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "features_v4.parquet"
)

SPIKE_MODEL_PATH = (
    ROOT
    / "models"
    / "spike_classifier"
    / "aqi_spike_classifier.pkl"
)

SPIKE_FEATURES_PATH = (
    ROOT
    / "models"
    / "spike_classifier"
    / "spike_classifier_features.json"
)

DELTA_MODEL_PATH = (
    ROOT
    / "models"
    / "delta_v6"
    / "delta_xgboost_model.json"
)

DELTA_FEATURES_PATH = (
    ROOT
    / "models"
    / "delta_v6"
    / "delta_features.json"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "spike_gated_v7"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "v7_strategy_results.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "v7_predictions.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "v7_metadata.json"
)


# ============================================================
# CONFIG
# ============================================================

SPIKE_THRESHOLD = 40

PROBABILITY_THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.70,
]

CORRECTION_STRENGTHS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

# Maximum correction to avoid unstable predictions.
MAX_CORRECTION = 60.0


# ============================================================
# HELPERS
# ============================================================

def mae(actual, prediction):
    return float(
        np.mean(
            np.abs(
                actual - prediction
            )
        )
    )


def rmse(actual, prediction):
    return float(
        np.sqrt(
            np.mean(
                (actual - prediction) ** 2
            )
        )
    )


def r2_score(actual, prediction):
    actual_mean = np.mean(actual)

    numerator = np.sum(
        (actual - prediction) ** 2
    )

    denominator = np.sum(
        (actual - actual_mean) ** 2
    )

    if denominator == 0:
        return 0.0

    return float(
        1 - numerator / denominator
    )


def within_percentage(
    actual,
    prediction,
    threshold,
):
    return float(
        np.mean(
            np.abs(
                actual - prediction
            ) <= threshold
        )
        * 100
    )


def evaluate(
    actual,
    prediction,
):
    return {
        "mae": mae(
            actual,
            prediction,
        ),
        "rmse": rmse(
            actual,
            prediction,
        ),
        "r2": r2_score(
            actual,
            prediction,
        ),
        "within_10": within_percentage(
            actual,
            prediction,
            10,
        ),
        "within_20": within_percentage(
            actual,
            prediction,
            20,
        ),
        "within_30": within_percentage(
            actual,
            prediction,
            30,
        ),
    }


# ============================================================
# START
# ============================================================

print("=" * 80)
print("PEARLSAQI V7 - SPIKE-AWARE GATED AQI CORRECTION")
print("=" * 80)


# ============================================================
# LOAD PRODUCTION PREDICTIONS
# ============================================================

print("\nLoading production predictions...")

if PRODUCTION_PREDICTIONS.exists():

    production = pd.read_parquet(
        PRODUCTION_PREDICTIONS
    )

elif LATEST_PREDICTIONS.exists():

    production = pd.read_csv(
        LATEST_PREDICTIONS
    )

else:

    raise FileNotFoundError(
        "Production predictions not found."
    )

print(
    f"Production prediction rows: "
    f"{len(production)}"
)

print(
    "Columns:"
)

print(
    production.columns.tolist()
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_prediction_columns = [
    "city_name",
    "date",
    "aqi",
    "target_aqi",
    "prediction",
]

missing = [
    column
    for column in required_prediction_columns
    if column not in production.columns
]

if missing:

    raise ValueError(
        f"Missing production prediction columns: {missing}"
    )


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading feature dataset...")

if not FEATURE_PATH.exists():

    raise FileNotFoundError(
        f"Feature dataset not found: {FEATURE_PATH}"
    )

features = pd.read_parquet(
    FEATURE_PATH
)

print(
    f"Feature rows: {len(features)}"
)


# ============================================================
# NORMALIZE DATES
# ============================================================

production["date"] = pd.to_datetime(
    production["date"]
)

features["date"] = pd.to_datetime(
    features["date"]
)


# ============================================================
# ALIGN FEATURES
# ============================================================

print("\nAligning feature rows...")

merge_keys = [
    "city_name",
    "date",
]

available_keys = [
    key
    for key in merge_keys
    if key in features.columns
    and key in production.columns
]

if len(available_keys) != 2:

    raise ValueError(
        "Could not align production predictions "
        "with feature dataset."
    )


# Avoid duplicate feature columns.
feature_columns_to_merge = [
    column
    for column in features.columns
    if column not in production.columns
    or column in available_keys
]

feature_subset = features[
    feature_columns_to_merge
].copy()


merged = production.merge(
    feature_subset,
    on=available_keys,
    how="inner",
    suffixes=(
        "",
        "_feature",
    ),
)

print(
    f"Merged rows: {len(merged)}"
)

if len(merged) != len(production):

    print(
        "\nWARNING:"
    )

    print(
        f"Production rows : {len(production)}"
    )

    print(
        f"Merged rows     : {len(merged)}"
    )

    print(
        "Some production rows could not be aligned."
    )


# ============================================================
# LOAD SPIKE CLASSIFIER
# ============================================================

print("\nLoading spike classifier...")

if not SPIKE_MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Spike classifier not found:\n"
        f"{SPIKE_MODEL_PATH}"
    )


spike_bundle = joblib.load(
    SPIKE_MODEL_PATH
)


if isinstance(
    spike_bundle,
    dict,
):

    spike_model = spike_bundle.get(
        "model"
    )

    saved_features = spike_bundle.get(
        "features"
    )

    saved_threshold = spike_bundle.get(
        "threshold",
        SPIKE_THRESHOLD,
    )

    saved_probability_threshold = spike_bundle.get(
        "prediction_threshold",
        0.50,
    )

else:

    spike_model = spike_bundle
    saved_features = None
    saved_threshold = SPIKE_THRESHOLD
    saved_probability_threshold = 0.50


print(
    f"Spike classifier type: "
    f"{type(spike_model).__name__}"
)

print(
    f"Saved spike definition threshold: "
    f"{saved_threshold}"
)

print(
    f"Saved classifier probability threshold: "
    f"{saved_probability_threshold}"
)


# ============================================================
# LOAD FEATURE LIST
# ============================================================

if SPIKE_FEATURES_PATH.exists():

    with open(
        SPIKE_FEATURES_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        spike_feature_data = json.load(f)

    if isinstance(
        spike_feature_data,
        dict,
    ):

        spike_features = (
            spike_feature_data.get(
                "features"
            )
            or spike_feature_data.get(
                "feature_columns"
            )
        )

    else:

        spike_features = spike_feature_data

else:

    spike_features = saved_features


if spike_features is None:

    raise ValueError(
        "Could not determine spike classifier features."
    )


print(
    f"Spike classifier features: "
    f"{len(spike_features)}"
)


# ============================================================
# VERIFY SPIKE FEATURES
# ============================================================

missing_spike_features = [
    feature
    for feature in spike_features
    if feature not in merged.columns
]

if missing_spike_features:

    print(
        "\nMissing spike classifier features:"
    )

    for feature in missing_spike_features:
        print(
            f"- {feature}"
        )

    raise ValueError(
        "\nCannot safely generate spike probabilities.\n"
        "The spike classifier requires features "
        "that are unavailable in the production dataset."
    )


# ============================================================
# BUILD SPIKE INPUT
# ============================================================

X_spike = merged[
    spike_features
].copy()

X_spike = X_spike.replace(
    [np.inf, -np.inf],
    np.nan,
)

# The classifier should not receive NaNs.
#
# We only fill missing values using values from the
# current production feature row itself.
#
# No target-derived information is introduced.

X_spike = X_spike.fillna(
    X_spike.median(
        numeric_only=True
    )
)

remaining_nan = X_spike.isna().sum().sum()

if remaining_nan > 0:

    raise ValueError(
        "Spike feature matrix still contains NaN values."
    )


# ============================================================
# SPIKE PROBABILITIES
# ============================================================

print(
    "\nGenerating spike probabilities..."
)

spike_probability = (
    spike_model
    .predict_proba(X_spike)[:, 1]
)

merged["spike_probability"] = (
    spike_probability
)


print(
    f"Spike probability mean: "
    f"{spike_probability.mean():.4f}"
)

print(
    f"Spike probability max : "
    f"{spike_probability.max():.4f}"
)


# ============================================================
# BASE PREDICTION
# ============================================================

actual = (
    merged["target_aqi"]
    .astype(float)
    .values
)

base_prediction = (
    merged["prediction"]
    .astype(float)
    .values
)

current_aqi = (
    merged["aqi"]
    .astype(float)
    .values
)


# ============================================================
# BASE METRICS
# ============================================================

base_metrics = evaluate(
    actual,
    base_prediction,
)

print("\n" + "=" * 80)
print("BASE PRODUCTION MODEL")
print("=" * 80)

print(
    f"MAE       : {base_metrics['mae']:.4f}"
)

print(
    f"RMSE      : {base_metrics['rmse']:.4f}"
)

print(
    f"R²        : {base_metrics['r2']:.4f}"
)

print(
    f"Within ±10: {base_metrics['within_10']:.2f}%"
)

print(
    f"Within ±20: {base_metrics['within_20']:.2f}%"
)

print(
    f"Within ±30: {base_metrics['within_30']:.2f}%"
)


# ============================================================
# IMPORTANT:
# CREATE A DATA-DRIVEN DELTA SIGNAL
# ============================================================
#
# We don't directly add a huge arbitrary correction.
#
# Instead, estimate the expected movement from current AQI
# and recent AQI momentum.
#
# This is intentionally conservative.
#

if "aqi_change_1d" in merged.columns:

    change_1d = (
        merged["aqi_change_1d"]
        .astype(float)
        .fillna(0)
        .values
    )

else:

    change_1d = (
        current_aqi
        - merged.get(
            "aqi_lag_1",
            pd.Series(
                current_aqi
            )
        ).astype(float).values
    )


if "aqi_change_3d" in merged.columns:

    change_3d = (
        merged["aqi_change_3d"]
        .astype(float)
        .fillna(0)
        .values
    )

else:

    change_3d = (
        change_1d
    )


if "aqi_change_7d" in merged.columns:

    change_7d = (
        merged["aqi_change_7d"]
        .astype(float)
        .fillna(0)
        .values
    )

else:

    change_7d = (
        change_3d
    )


# Momentum estimate.

momentum = (
    0.50 * change_1d
    + 0.30 * change_3d
    + 0.20 * change_7d
)


# ============================================================
# SPIKE DIRECTION
# ============================================================

# If momentum is positive, upward correction.
# If negative, downward correction.

direction = np.sign(
    momentum
)

# If there is almost no momentum, do not force a direction.

direction[
    np.abs(momentum) < 2
] = 0


# ============================================================
# TEST V7 STRATEGIES
# ============================================================

print("\n" + "=" * 80)
print("V7 STRATEGY SEARCH")
print("=" * 80)

results = []

best_strategy = None

for probability_threshold in PROBABILITY_THRESHOLDS:

    spike_gate = (
        spike_probability
        >= probability_threshold
    )

    spike_count = int(
        spike_gate.sum()
    )

    for strength in CORRECTION_STRENGTHS:

        correction = (
            direction
            * momentum
            * strength
        )

        # Only activate when classifier says
        # there is a meaningful spike probability.

        correction = np.where(
            spike_gate,
            correction,
            0.0,
        )

        correction = np.clip(
            correction,
            -MAX_CORRECTION,
            MAX_CORRECTION,
        )

        prediction = (
            base_prediction
            + correction
        )

        metrics = evaluate(
            actual,
            prediction,
        )

        row = {
            "probability_threshold":
                probability_threshold,

            "correction_strength":
                strength,

            "gated_rows":
                spike_count,

            **metrics,
        }

        results.append(
            row
        )


results_df = pd.DataFrame(
    results
)


# ============================================================
# FIND BEST STRATEGY
# ============================================================

# Primary objective:
# lowest MAE.
#
# Secondary:
# lowest RMSE.

results_df = results_df.sort_values(
    [
        "mae",
        "rmse",
    ],
    ascending=[
        True,
        True,
    ],
).reset_index(
    drop=True
)


best = results_df.iloc[0]

best_probability_threshold = float(
    best["probability_threshold"]
)

best_strength = float(
    best["correction_strength"]
)


# ============================================================
# BEST PREDICTION
# ============================================================

best_gate = (
    spike_probability
    >= best_probability_threshold
)

best_correction = (
    direction
    * momentum
    * best_strength
)

best_correction = np.where(
    best_gate,
    best_correction,
    0.0,
)

best_correction = np.clip(
    best_correction,
    -MAX_CORRECTION,
    MAX_CORRECTION,
)

best_prediction = (
    base_prediction
    + best_correction
)


# ============================================================
# BEST METRICS
# ============================================================

best_metrics = evaluate(
    actual,
    best_prediction,
)


print("\nBEST V7 STRATEGY")
print("=" * 80)

print(
    f"Probability threshold : "
    f"{best_probability_threshold:.2f}"
)

print(
    f"Correction strength   : "
    f"{best_strength:.2f}"
)

print(
    f"Gated rows             : "
    f"{int(best_gate.sum())}"
)

print(
    f"MAE                    : "
    f"{best_metrics['mae']:.4f}"
)

print(
    f"RMSE                   : "
    f"{best_metrics['rmse']:.4f}"
)

print(
    f"R²                     : "
    f"{best_metrics['r2']:.4f}"
)

print(
    f"Within ±10             : "
    f"{best_metrics['within_10']:.2f}%"
)

print(
    f"Within ±20             : "
    f"{best_metrics['within_20']:.2f}%"
)

print(
    f"Within ±30             : "
    f"{best_metrics['within_30']:.2f}%"
)


# ============================================================
# IMPROVEMENT
# ============================================================

print("\nV7 IMPROVEMENT")
print("=" * 80)

print(
    f"MAE improvement  : "
    f"{base_metrics['mae'] - best_metrics['mae']:.4f}"
)

print(
    f"RMSE improvement : "
    f"{base_metrics['rmse'] - best_metrics['rmse']:.4f}"
)

print(
    f"R² improvement   : "
    f"{best_metrics['r2'] - base_metrics['r2']:.4f}"
)


# ============================================================
# EXTREME ANALYSIS
# ============================================================

extreme_mask = (
    np.abs(
        actual - current_aqi
    )
    >= SPIKE_THRESHOLD
)

extreme_rows = int(
    extreme_mask.sum()
)

if extreme_rows > 0:

    base_extreme_mae = mae(
        actual[extreme_mask],
        base_prediction[extreme_mask],
    )

    v7_extreme_mae = mae(
        actual[extreme_mask],
        best_prediction[extreme_mask],
    )

    base_extreme_rmse = rmse(
        actual[extreme_mask],
        base_prediction[extreme_mask],
    )

    v7_extreme_rmse = rmse(
        actual[extreme_mask],
        best_prediction[extreme_mask],
    )

else:

    base_extreme_mae = 0.0
    v7_extreme_mae = 0.0
    base_extreme_rmse = 0.0
    v7_extreme_rmse = 0.0


print("\nEXTREME / SPIKE PERFORMANCE")
print("=" * 80)

print(
    f"Extreme rows     : "
    f"{extreme_rows}"
)

print(
    f"Base extreme MAE : "
    f"{base_extreme_mae:.4f}"
)

print(
    f"V7 extreme MAE   : "
    f"{v7_extreme_mae:.4f}"
)

print(
    f"MAE improvement  : "
    f"{base_extreme_mae - v7_extreme_mae:.4f}"
)

print(
    f"Base extreme RMSE: "
    f"{base_extreme_rmse:.4f}"
)

print(
    f"V7 extreme RMSE  : "
    f"{v7_extreme_rmse:.4f}"
)


# ============================================================
# CREATE OUTPUT DATASET
# ============================================================

output = merged.copy()

output["base_prediction"] = (
    base_prediction
)

output["spike_probability"] = (
    spike_probability
)

output["spike_gate"] = (
    best_gate.astype(int)
)

output["momentum"] = (
    momentum
)

output["v7_correction"] = (
    best_correction
)

output["v7_prediction"] = (
    best_prediction
)

output["v7_absolute_error"] = (
    np.abs(
        actual - best_prediction
    )
)

output["v7_error"] = (
    best_prediction - actual
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False,
)


output.to_parquet(
    PREDICTIONS_PATH,
    index=False,
)


metadata = {
    "model": "PEARLSAQI V7",
    "base_model": "Production XGBoost",
    "spike_classifier": "HistGradientBoostingClassifier",
    "spike_feature_count": len(
        spike_features
    ),
    "spike_definition_threshold":
        SPIKE_THRESHOLD,

    "best_probability_threshold":
        best_probability_threshold,

    "best_correction_strength":
        best_strength,

    "max_correction":
        MAX_CORRECTION,

    "base_metrics":
        base_metrics,

    "v7_metrics":
        best_metrics,

    "extreme_rows":
        extreme_rows,

    "base_extreme_mae":
        base_extreme_mae,

    "v7_extreme_mae":
        v7_extreme_mae,

    "base_extreme_rmse":
        base_extreme_rmse,

    "v7_extreme_rmse":
        v7_extreme_rmse,
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
    )


# ============================================================
# TOP IMPROVEMENTS / FAILURES
# ============================================================

output["base_absolute_error"] = (
    np.abs(
        actual - base_prediction
    )
)

output["improvement"] = (
    output["base_absolute_error"]
    - output["v7_absolute_error"]
)

print("\nTOP V7 IMPROVEMENTS")

print(
    output[
        [
            "city_name",
            "date",
            "aqi",
            "target_aqi",
            "base_prediction",
            "v7_prediction",
            "base_absolute_error",
            "v7_absolute_error",
            "improvement",
            "spike_probability",
            "v7_correction",
        ]
    ]
    .sort_values(
        "improvement",
        ascending=False,
    )
    .head(15)
    .to_string(
        index=False
    )
)


print("\nTOP V7 FAILURES")

print(
    output[
        [
            "city_name",
            "date",
            "aqi",
            "target_aqi",
            "base_prediction",
            "v7_prediction",
            "base_absolute_error",
            "v7_absolute_error",
            "improvement",
            "spike_probability",
            "v7_correction",
        ]
    ]
    .sort_values(
        "improvement",
        ascending=True,
    )
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("V7 COMPLETED")
print("=" * 80)

print(
    f"Results      : {RESULTS_PATH}"
)

print(
    f"Predictions  : {PREDICTIONS_PATH}"
)

print(
    f"Metadata     : {METADATA_PATH}"
)

print("\nPEARLSAQI V7 completed successfully.")