"""
PEARLSAQI V8 - CITY-AWARE MOMENTUM CORRECTION

Purpose
-------
Improve AQI predictions where diagnostics show systematic errors:

1. Sudden upward AQI spikes
2. Sudden downward AQI spikes
3. Extreme / hazardous AQI underprediction
4. City-specific behavior
5. Current-AQI regime behavior
6. Momentum behavior

IMPORTANT
---------
This module does NOT retrain the production XGBoost model.

The correction parameters are learned only from validation data
and evaluated on the untouched test set.

If validation does not contain production predictions, V8 creates
a validation-compatible baseline model using the training dataset.
This avoids requiring unavailable production-only features.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "train.parquet"
)

VALIDATION_PATH = (
    ROOT
    / "data"
    / "processed"
    / "validation.parquet"
)

TEST_PATH = (
    ROOT
    / "data"
    / "analysis"
    / "test_error_analysis.parquet"
)

PRODUCTION_MODEL = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "final_xgboost_model.json"
)

PRODUCTION_FEATURES = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "features.json"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "v8"
    / "momentum_correction"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# AQI CATEGORIES
# ============================================================

def get_regime(aqi):

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "USG"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


# ============================================================
# LOAD PRODUCTION FEATURE LIST
# ============================================================

def load_production_features():

    if not PRODUCTION_FEATURES.exists():

        raise FileNotFoundError(
            f"Production feature file not found:\n"
            f"{PRODUCTION_FEATURES}"
        )

    with open(
        PRODUCTION_FEATURES,
        "r",
        encoding="utf-8",
    ) as f:

        feature_data = json.load(f)

    if isinstance(feature_data, dict):

        if "features" in feature_data:

            features = feature_data["features"]

        elif "feature_names" in feature_data:

            features = feature_data["feature_names"]

        else:

            raise ValueError(
                "features.json does not contain "
                "'features' or 'feature_names'."
            )

    elif isinstance(feature_data, list):

        features = feature_data

    else:

        raise ValueError(
            "Unsupported features.json format."
        )

    return [
        str(feature)
        for feature in features
    ]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_momentum_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in [
        "aqi",
        "prediction",
        "target_aqi",
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    if "aqi_change_1d" in df.columns:

        df["momentum_1d"] = (
            pd.to_numeric(
                df["aqi_change_1d"],
                errors="coerce",
            )
            .fillna(0)
        )

    else:

        df["momentum_1d"] = 0.0

    if "aqi_change_2d" in df.columns:

        df["momentum_2d"] = (
            pd.to_numeric(
                df["aqi_change_2d"],
                errors="coerce",
            )
            .fillna(0)
        )

    else:

        df["momentum_2d"] = 0.0

    if "aqi_change_3d" in df.columns:

        df["momentum_3d"] = (
            pd.to_numeric(
                df["aqi_change_3d"],
                errors="coerce",
            )
            .fillna(0)
        )

    else:

        df["momentum_3d"] = 0.0

    if "aqi_change_7d" in df.columns:

        df["momentum_7d"] = (
            pd.to_numeric(
                df["aqi_change_7d"],
                errors="coerce",
            )
            .fillna(0)
        )

    else:

        df["momentum_7d"] = 0.0

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    df["momentum_acceleration"] = (
        df["momentum_1d"]
        - df["momentum_2d"]
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    if "aqi_std_3" in df.columns:

        df["volatility_3"] = (
            pd.to_numeric(
                df["aqi_std_3"],
                errors="coerce",
            )
            .fillna(0)
        )

    else:

        df["volatility_3"] = 0.0

    if "aqi_std_7" in df.columns:

        df["volatility_7"] = (
            pd.to_numeric(
                df["aqi_std_7"],
                errors="coerce",
            )
            .fillna(0)
        )

    else:

        df["volatility_7"] = 0.0

    # --------------------------------------------------------
    # Range
    # --------------------------------------------------------

    if "aqi_range_7" in df.columns:

        df["range_7"] = (
            pd.to_numeric(
                df["aqi_range_7"],
                errors="coerce",
            )
            .fillna(0)
        )

    else:

        df["range_7"] = 0.0

    if "aqi_range_14" in df.columns:

        df["range_14"] = (
            pd.to_numeric(
                df["aqi_range_14"],
                errors="coerce",
            )
            .fillna(0)
        )

    else:

        df["range_14"] = 0.0

    # --------------------------------------------------------
    # Distance from recent maximum
    # --------------------------------------------------------

    if (
        "aqi_max_7" in df.columns
        and "aqi" in df.columns
    ):

        df["distance_from_max_7"] = (
            pd.to_numeric(
                df["aqi_max_7"],
                errors="coerce",
            ).fillna(df["aqi"])
            - df["aqi"]
        )

    else:

        df["distance_from_max_7"] = 0.0

    if (
        "aqi_max_14" in df.columns
        and "aqi" in df.columns
    ):

        df["distance_from_max_14"] = (
            pd.to_numeric(
                df["aqi_max_14"],
                errors="coerce",
            ).fillna(df["aqi"])
            - df["aqi"]
        )

    else:

        df["distance_from_max_14"] = 0.0

    # --------------------------------------------------------
    # AQI regime
    # --------------------------------------------------------

    df["regime"] = df["aqi"].apply(
        get_regime
    )

    # --------------------------------------------------------
    # AQI flags
    # --------------------------------------------------------

    df["high_aqi_flag"] = (
        df["aqi"] >= 200
    ).astype(int)

    df["very_high_aqi_flag"] = (
        df["aqi"] >= 300
    ).astype(int)

    df["upward_momentum_flag"] = (
        df["momentum_1d"] >= 20
    ).astype(int)

    df["strong_upward_momentum_flag"] = (
        df["momentum_1d"] >= 40
    ).astype(int)

    df["downward_momentum_flag"] = (
        df["momentum_1d"] <= -20
    ).astype(int)

    return df


# ============================================================
# METRICS
# ============================================================

def mae(df, prediction_column):

    return float(
        np.mean(
            np.abs(
                df["target_aqi"]
                - df[prediction_column]
            )
        )
    )


def rmse(df, prediction_column):

    return float(
        np.sqrt(
            np.mean(
                (
                    df["target_aqi"]
                    - df[prediction_column]
                ) ** 2
            )
        )
    )


def r2(df, prediction_column):

    y = df["target_aqi"].to_numpy()
    p = df[prediction_column].to_numpy()

    ss_res = np.sum(
        (y - p) ** 2
    )

    ss_tot = np.sum(
        (y - y.mean()) ** 2
    )

    if ss_tot == 0:

        return 0.0

    return float(
        1 - ss_res / ss_tot
    )


# ============================================================
# VALIDATION BASELINE MODEL
# ============================================================

def generate_validation_predictions(
    train,
    validation,
):

    print("\n" + "=" * 80)
    print("GENERATING VALIDATION PREDICTIONS")
    print("=" * 80)

    print(
        "Validation does not contain a "
        "'prediction' column."
    )

    print(
        "A validation-compatible baseline "
        "will be trained using the training dataset."
    )

    # --------------------------------------------------------
    # Determine common usable features
    # --------------------------------------------------------

    excluded = {
        "target_aqi",
        "city_name",
        "date",
        "prediction",
        "absolute_error",
        "error",
    }

    common_features = [
        column
        for column in train.columns
        if (
            column in validation.columns
            and column not in excluded
        )
    ]

    if not common_features:

        raise ValueError(
            "No common training/validation features "
            "are available."
        )

    print(
        f"Common baseline features: "
        f"{len(common_features)}"
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    X_train = train[
        common_features
    ].copy()

    X_validation = validation[
        common_features
    ].copy()

    for column in common_features:

        X_train[column] = pd.to_numeric(
            X_train[column],
            errors="coerce",
        )

        X_validation[column] = pd.to_numeric(
            X_validation[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Fill missing values using training medians
    # --------------------------------------------------------

    train_medians = (
        X_train
        .median(numeric_only=True)
        .fillna(0)
    )

    X_train = X_train.fillna(
        train_medians
    )

    X_validation = X_validation.fillna(
        train_medians
    )

    y_train = pd.to_numeric(
        train["target_aqi"],
        errors="coerce",
    )

    valid_mask = y_train.notna()

    X_train = X_train.loc[
        valid_mask
    ]

    y_train = y_train.loc[
        valid_mask
    ]

    print(
        f"Training rows used: "
        f"{len(X_train)}"
    )

    # --------------------------------------------------------
    # Compatible XGBoost baseline
    # --------------------------------------------------------

    baseline_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=-1,
    )

    print(
        "\nTraining validation-compatible "
        "XGBoost baseline..."
    )

    baseline_model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    print(
        "Baseline training completed."
    )

    # --------------------------------------------------------
    # Predict validation
    # --------------------------------------------------------

    validation["prediction"] = (
        baseline_model.predict(
            X_validation
        )
    )

    validation["prediction"] = (
        pd.to_numeric(
            validation["prediction"],
            errors="coerce",
        )
    )

    if validation["prediction"].isna().any():

        raise ValueError(
            "Generated validation predictions "
            "contain NaN values."
        )

    print(
        f"Validation prediction mean: "
        f"{validation['prediction'].mean():.2f}"
    )

    print(
        f"Validation prediction min : "
        f"{validation['prediction'].min():.2f}"
    )

    print(
        f"Validation prediction max : "
        f"{validation['prediction'].max():.2f}"
    )

    return validation


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("PEARLSAQI V8 - CITY-AWARE MOMENTUM CORRECTION")
print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training data...")

train = pd.read_parquet(
    TRAIN_PATH
)

print(
    f"Training rows   : {len(train)}"
)

print("\nLoading validation data...")

validation = pd.read_parquet(
    VALIDATION_PATH
)

print(
    f"Validation rows : {len(validation)}"
)

print("\nLoading test data...")

test = pd.read_parquet(
    TEST_PATH
)

print(
    f"Test rows       : {len(test)}"
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_validation = {
    "city_name",
    "date",
    "aqi",
    "target_aqi",
}

required_test = {
    "city_name",
    "date",
    "aqi",
    "target_aqi",
    "prediction",
}

for column in required_validation:

    if column not in validation.columns:

        raise ValueError(
            f"Validation missing column: "
            f"{column}"
        )


for column in required_test:

    if column not in test.columns:

        raise ValueError(
            f"Test missing column: "
            f"{column}"
        )


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

if "prediction" not in validation.columns:

    validation = generate_validation_predictions(
        train,
        validation,
    )

else:

    print(
        "\nValidation already contains "
        "predictions."
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print(
    "\nBuilding momentum features..."
)

validation = build_momentum_features(
    validation
)

test = build_momentum_features(
    test
)


# ============================================================
# BASELINE ERRORS
# ============================================================

validation["base_error"] = (
    validation["target_aqi"]
    - validation["prediction"]
)

test["base_error"] = (
    test["target_aqi"]
    - test["prediction"]
)

validation["base_abs_error"] = (
    validation["base_error"].abs()
)

test["base_abs_error"] = (
    test["base_error"].abs()
)


# ============================================================
# BASELINE METRICS
# ============================================================

BASE_VALIDATION_MAE = mae(
    validation,
    "prediction",
)

BASE_TEST_MAE = mae(
    test,
    "prediction",
)

BASE_TEST_RMSE = rmse(
    test,
    "prediction",
)

BASE_TEST_R2 = r2(
    test,
    "prediction",
)


print("\n" + "=" * 80)
print("BASELINE")
print("=" * 80)

print(
    f"Validation MAE : "
    f"{BASE_VALIDATION_MAE:.4f}"
)

print(
    f"Test MAE       : "
    f"{BASE_TEST_MAE:.4f}"
)

print(
    f"Test RMSE      : "
    f"{BASE_TEST_RMSE:.4f}"
)

print(
    f"Test R²        : "
    f"{BASE_TEST_R2:.4f}"
)


# ============================================================
# CITY BIAS
# ============================================================

city_bias = (
    validation
    .groupby("city_name")["base_error"]
    .mean()
    .to_dict()
)


# ============================================================
# REGIME BIAS
# ============================================================

regime_bias = (
    validation
    .groupby("regime")["base_error"]
    .mean()
    .to_dict()
)


# ============================================================
# CITY × REGIME BIAS
# ============================================================

city_regime_bias = (
    validation
    .groupby(
        ["city_name", "regime"]
    )["base_error"]
    .mean()
    .to_dict()
)


# ============================================================
# MOMENTUM ANALYSIS
# ============================================================

validation["actual_change"] = (
    validation["target_aqi"]
    - validation["aqi"]
)

validation["predicted_change"] = (
    validation["prediction"]
    - validation["aqi"]
)

validation["change_error"] = (
    validation["actual_change"]
    - validation["predicted_change"]
)


# ============================================================
# MOMENTUM BUCKET
# ============================================================

def momentum_bucket(value):

    if value >= 60:
        return "strong_up"

    if value >= 30:
        return "up"

    if value <= -60:
        return "strong_down"

    if value <= -30:
        return "down"

    return "stable"


validation["momentum_bucket"] = (
    validation["momentum_1d"]
    .apply(momentum_bucket)
)

test["momentum_bucket"] = (
    test["momentum_1d"]
    .apply(momentum_bucket)
)


momentum_bias = (
    validation
    .groupby(
        "momentum_bucket"
    )["change_error"]
    .mean()
    .to_dict()
)


# ============================================================
# SEARCH CORRECTION STRATEGIES
# ============================================================

print("\n" + "=" * 80)
print("SEARCHING V8 CORRECTION STRATEGIES")
print("=" * 80)

results = []


# ============================================================
# GLOBAL
# ============================================================

for strength in np.arange(
    0.0,
    1.01,
    0.05,
):

    correction = (
        validation["base_error"].mean()
    )

    corrected = (
        validation["prediction"]
        + strength * correction
    )

    error = np.abs(
        validation["target_aqi"]
        - corrected
    ).mean()

    results.append({
        "strategy": "GLOBAL",
        "strength": strength,
        "validation_mae": error,
    })


# ============================================================
# CITY
# ============================================================

for strength in np.arange(
    0.0,
    1.01,
    0.05,
):

    correction = (
        validation["city_name"]
        .map(city_bias)
        .fillna(0)
    )

    corrected = (
        validation["prediction"]
        + strength * correction
    )

    error = np.abs(
        validation["target_aqi"]
        - corrected
    ).mean()

    results.append({
        "strategy": "CITY",
        "strength": strength,
        "validation_mae": error,
    })


# ============================================================
# REGIME
# ============================================================

for strength in np.arange(
    0.0,
    1.01,
    0.05,
):

    correction = (
        validation["regime"]
        .map(regime_bias)
        .fillna(0)
    )

    corrected = (
        validation["prediction"]
        + strength * correction
    )

    error = np.abs(
        validation["target_aqi"]
        - corrected
    ).mean()

    results.append({
        "strategy": "REGIME",
        "strength": strength,
        "validation_mae": error,
    })


# ============================================================
# CITY × REGIME
# ============================================================

for strength in np.arange(
    0.0,
    1.01,
    0.05,
):

    correction = pd.Series(
        [
            city_regime_bias.get(
                (
                    city,
                    regime,
                ),
                0.0,
            )
            for city, regime
            in zip(
                validation["city_name"],
                validation["regime"],
            )
        ],
        index=validation.index,
    )

    corrected = (
        validation["prediction"]
        + strength * correction
    )

    error = np.abs(
        validation["target_aqi"]
        - corrected
    ).mean()

    results.append({
        "strategy": "CITY_REGIME",
        "strength": strength,
        "validation_mae": error,
    })


# ============================================================
# MOMENTUM
# ============================================================

for strength in np.arange(
    0.0,
    1.01,
    0.05,
):

    correction = (
        validation["momentum_bucket"]
        .map(momentum_bias)
        .fillna(0)
    )

    corrected = (
        validation["prediction"]
        + strength * correction
    )

    error = np.abs(
        validation["target_aqi"]
        - corrected
    ).mean()

    results.append({
        "strategy": "MOMENTUM",
        "strength": strength,
        "validation_mae": error,
    })


# ============================================================
# SELECT BEST
# ============================================================

results_df = pd.DataFrame(
    results
)

best = results_df.loc[
    results_df["validation_mae"].idxmin()
]

best_strategy = (
    best["strategy"]
)

best_strength = float(
    best["strength"]
)


print("\n" + "=" * 80)
print("BEST VALIDATION STRATEGY")
print("=" * 80)

print(
    f"Strategy : {best_strategy}"
)

print(
    f"Strength : {best_strength:.2f}"
)

print(
    f"MAE      : "
    f"{best['validation_mae']:.4f}"
)


# ============================================================
# APPLY CORRECTION
# ============================================================

def apply_correction(df):

    df = df.copy()

    prediction = (
        df["prediction"]
        .astype(float)
    )

    if best_strategy == "GLOBAL":

        correction = pd.Series(
            validation["base_error"].mean(),
            index=df.index,
        )

    elif best_strategy == "CITY":

        correction = (
            df["city_name"]
            .map(city_bias)
            .fillna(0)
        )

    elif best_strategy == "REGIME":

        correction = (
            df["regime"]
            .map(regime_bias)
            .fillna(0)
        )

    elif best_strategy == "CITY_REGIME":

        correction = pd.Series(
            [
                city_regime_bias.get(
                    (
                        city,
                        regime,
                    ),
                    0.0,
                )
                for city, regime
                in zip(
                    df["city_name"],
                    df["regime"],
                )
            ],
            index=df.index,
        )

    elif best_strategy == "MOMENTUM":

        correction = (
            df["momentum_bucket"]
            .map(momentum_bias)
            .fillna(0)
        )

    else:

        correction = pd.Series(
            0.0,
            index=df.index,
        )

    return (
        prediction
        + best_strength * correction
    )


# ============================================================
# APPLY TO TEST
# ============================================================

test["v8_prediction"] = (
    apply_correction(test)
)


# ============================================================
# SAFETY CLAMP
# ============================================================

test["v8_prediction"] = (
    test["v8_prediction"]
    .clip(
        lower=0,
        upper=500,
    )
)


# ============================================================
# TEST METRICS
# ============================================================

V8_TEST_MAE = mae(
    test,
    "v8_prediction",
)

V8_TEST_RMSE = rmse(
    test,
    "v8_prediction",
)

V8_TEST_R2 = r2(
    test,
    "v8_prediction",
)


print("\n" + "=" * 80)
print("V8 TEST RESULTS")
print("=" * 80)

print(
    f"Base MAE     : "
    f"{BASE_TEST_MAE:.4f}"
)

print(
    f"V8 MAE       : "
    f"{V8_TEST_MAE:.4f}"
)

print(
    f"MAE improvement : "
    f"{BASE_TEST_MAE - V8_TEST_MAE:.4f}"
)

print(
    f"\nBase RMSE    : "
    f"{BASE_TEST_RMSE:.4f}"
)

print(
    f"V8 RMSE      : "
    f"{V8_TEST_RMSE:.4f}"
)

print(
    f"RMSE improvement : "
    f"{BASE_TEST_RMSE - V8_TEST_RMSE:.4f}"
)

print(
    f"\nBase R²      : "
    f"{BASE_TEST_R2:.4f}"
)

print(
    f"V8 R²        : "
    f"{V8_TEST_R2:.4f}"
)


# ============================================================
# EXTREME AQI
# ============================================================

test["extreme"] = (
    test["target_aqi"] >= 200
)

extreme = test[
    test["extreme"]
].copy()


if len(extreme) > 0:

    base_extreme_mae = np.mean(
        np.abs(
            extreme["target_aqi"]
            - extreme["prediction"]
        )
    )

    v8_extreme_mae = np.mean(
        np.abs(
            extreme["target_aqi"]
            - extreme["v8_prediction"]
        )
    )

else:

    base_extreme_mae = 0.0
    v8_extreme_mae = 0.0


print("\n" + "=" * 80)
print("EXTREME AQI")
print("=" * 80)

print(
    f"Extreme rows : "
    f"{len(extreme)}"
)

print(
    f"Base MAE     : "
    f"{base_extreme_mae:.4f}"
)

print(
    f"V8 MAE       : "
    f"{v8_extreme_mae:.4f}"
)

print(
    f"Improvement  : "
    f"{base_extreme_mae - v8_extreme_mae:.4f}"
)


# ============================================================
# UPWARD SPIKES
# ============================================================

upward = (
    test["target_aqi"]
    - test["aqi"]
) >= 40


if upward.sum() > 0:

    base_spike_mae = np.mean(
        np.abs(
            test.loc[
                upward,
                "target_aqi",
            ]
            -
            test.loc[
                upward,
                "prediction",
            ]
        )
    )

    v8_spike_mae = np.mean(
        np.abs(
            test.loc[
                upward,
                "target_aqi",
            ]
            -
            test.loc[
                upward,
                "v8_prediction",
            ]
        )
    )

else:

    base_spike_mae = 0.0
    v8_spike_mae = 0.0


print("\n" + "=" * 80)
print("UPWARD SPIKES")
print("=" * 80)

print(
    f"Rows       : "
    f"{upward.sum()}"
)

print(
    f"Base MAE   : "
    f"{base_spike_mae:.4f}"
)

print(
    f"V8 MAE     : "
    f"{v8_spike_mae:.4f}"
)

print(
    f"Improvement: "
    f"{base_spike_mae - v8_spike_mae:.4f}"
)


# ============================================================
# DOWNWARD SPIKES
# ============================================================

downward = (
    test["target_aqi"]
    - test["aqi"]
) <= -40


if downward.sum() > 0:

    base_downward_mae = np.mean(
        np.abs(
            test.loc[
                downward,
                "target_aqi",
            ]
            -
            test.loc[
                downward,
                "prediction",
            ]
        )
    )

    v8_downward_mae = np.mean(
        np.abs(
            test.loc[
                downward,
                "target_aqi",
            ]
            -
            test.loc[
                downward,
                "v8_prediction",
            ]
        )
    )

else:

    base_downward_mae = 0.0
    v8_downward_mae = 0.0


print("\n" + "=" * 80)
print("DOWNWARD SPIKES")
print("=" * 80)

print(
    f"Rows       : "
    f"{downward.sum()}"
)

print(
    f"Base MAE   : "
    f"{base_downward_mae:.4f}"
)

print(
    f"V8 MAE     : "
    f"{v8_downward_mae:.4f}"
)

print(
    f"Improvement: "
    f"{base_downward_mae - v8_downward_mae:.4f}"
)


# ============================================================
# CITY PERFORMANCE
# ============================================================

city_results = []


for city, group in test.groupby(
    "city_name"
):

    city_base = np.mean(
        np.abs(
            group["target_aqi"]
            - group["prediction"]
        )
    )

    city_v8 = np.mean(
        np.abs(
            group["target_aqi"]
            - group["v8_prediction"]
        )
    )

    city_results.append({
        "city": city,
        "rows": len(group),
        "base_mae": city_base,
        "v8_mae": city_v8,
        "improvement": (
            city_base - city_v8
        ),
    })


city_results_df = (
    pd.DataFrame(city_results)
    .sort_values(
        "improvement",
        ascending=False,
    )
)


print("\n" + "=" * 80)
print("CITY PERFORMANCE")
print("=" * 80)

print(
    city_results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    OUTPUT_DIR
    / "v8_strategy_results.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)


city_path = (
    OUTPUT_DIR
    / "v8_city_results.csv"
)

city_results_df.to_csv(
    city_path,
    index=False,
)


predictions_path = (
    OUTPUT_DIR
    / "v8_predictions.parquet"
)

test[
    [
        "city_name",
        "date",
        "aqi",
        "target_aqi",
        "prediction",
        "v8_prediction",
        "base_abs_error",
        "extreme",
    ]
].to_parquet(
    predictions_path,
    index=False,
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model":
        "PEARLSAQI V8",

    "type":
        "city-aware momentum correction",

    "validation_prediction_source":
        (
            "existing validation prediction"
            if "prediction" in validation.columns
            else "validation-compatible XGBoost"
        ),

    "validation_rows":
        int(len(validation)),

    "test_rows":
        int(len(test)),

    "base_validation_mae":
        BASE_VALIDATION_MAE,

    "base_test_mae":
        BASE_TEST_MAE,

    "base_test_rmse":
        BASE_TEST_RMSE,

    "base_test_r2":
        BASE_TEST_R2,

    "v8_test_mae":
        V8_TEST_MAE,

    "v8_test_rmse":
        V8_TEST_RMSE,

    "v8_test_r2":
        V8_TEST_R2,

    "mae_improvement":
        BASE_TEST_MAE - V8_TEST_MAE,

    "rmse_improvement":
        BASE_TEST_RMSE - V8_TEST_RMSE,

    "best_strategy":
        str(best_strategy),

    "best_strength":
        best_strength,

    "city_bias":
        {
            str(k): float(v)
            for k, v in city_bias.items()
        },

    "regime_bias":
        {
            str(k): float(v)
            for k, v in regime_bias.items()
        },

    "momentum_bias":
        {
            str(k): float(v)
            for k, v in momentum_bias.items()
        },

    "extreme_base_mae":
        float(base_extreme_mae),

    "extreme_v8_mae":
        float(v8_extreme_mae),

    "upward_spike_base_mae":
        float(base_spike_mae),

    "upward_spike_v8_mae":
        float(v8_spike_mae),

    "downward_spike_base_mae":
        float(base_downward_mae),

    "downward_spike_v8_mae":
        float(v8_downward_mae),
}


metadata_path = (
    OUTPUT_DIR
    / "v8_metadata.json"
)


with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(
    f"Strategy results : "
    f"{results_path}"
)

print(
    f"City results     : "
    f"{city_path}"
)

print(
    f"Predictions      : "
    f"{predictions_path}"
)

print(
    f"Metadata         : "
    f"{metadata_path}"
)

print(
    "\nPEARLSAQI V8 completed successfully."
)