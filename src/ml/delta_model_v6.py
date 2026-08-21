from pathlib import Path

import json
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PEARLSAQI V6
# AQI DELTA / MOMENTUM MODEL
#
# Purpose:
#   Learn next-day AQI movement separately from the main AQI
#   prediction model.
#
# IMPORTANT:
#   The production XGBoost uses 107 engineered features.
#   The older train/validation/test parquet files available
#   for this experiment contain only 59 of those features.
#
#   Therefore:
#       - V6 Delta Model -> uses common 59 features
#       - Existing production predictions -> used as baseline
#
#   We DO NOT feed 59 features into the 107-feature production
#   model.
# ============================================================


print()
print("=" * 80)
print("PEARLSAQI V6 - AQI DELTA / MOMENTUM MODEL")
print("=" * 80)
print()


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    ROOT
    / "data"
    / "processed"
    / "train.parquet"
)

VALIDATION_FILE = (
    ROOT
    / "data"
    / "processed"
    / "validation.parquet"
)

TEST_FILE = (
    ROOT
    / "data"
    / "analysis"
    / "test_error_analysis.parquet"
)

FEATURE_FILE = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "features.json"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "delta_v6"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE FINDER
# ============================================================

def find_file(primary, keywords):

    if primary.exists():
        return primary

    candidates = []

    search_folders = [
        ROOT / "data",
        ROOT / "data" / "processed",
        ROOT / "data" / "analysis",
        ROOT / "models",
    ]

    for folder in search_folders:

        if not folder.exists():
            continue

        for file in folder.rglob("*"):

            if file.suffix.lower() not in [
                ".parquet",
                ".csv",
            ]:
                continue

            name = file.name.lower()

            if all(
                keyword.lower() in name
                for keyword in keywords
            ):
                candidates.append(file)

    if candidates:
        return candidates[0]

    return None


TRAIN_FILE = find_file(
    TRAIN_FILE,
    ["train"],
)

VALIDATION_FILE = find_file(
    VALIDATION_FILE,
    ["validation"],
)

TEST_FILE = find_file(
    TEST_FILE,
    ["test"],
)


print("Training file   :", TRAIN_FILE)
print("Validation file :", VALIDATION_FILE)
print("Test file       :", TEST_FILE)
print()


if TRAIN_FILE is None:
    raise FileNotFoundError(
        "Training dataset could not be found."
    )

if VALIDATION_FILE is None:
    raise FileNotFoundError(
        "Validation dataset could not be found."
    )

if TEST_FILE is None:
    raise FileNotFoundError(
        "Test dataset could not be found."
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    return pd.read_csv(path)


train = load_data(TRAIN_FILE)
validation = load_data(VALIDATION_FILE)
test = load_data(TEST_FILE)


print("Training rows   :", len(train))
print("Validation rows :", len(validation))
print("Test rows       :", len(test))
print()


# ============================================================
# LOAD PRODUCTION FEATURE LIST
# ============================================================

if FEATURE_FILE.exists():

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        feature_data = json.load(f)

    if isinstance(feature_data, dict):

        production_features = feature_data.get(
            "features",
            [],
        )

    else:

        production_features = feature_data

else:

    production_features = []


production_features = [
    feature
    for feature in production_features
    if "target" not in feature.lower()
]


print(
    "Production feature count:",
    len(production_features),
)
print()


# ============================================================
# COMMON FEATURES
# ============================================================

common_features = [
    feature
    for feature in production_features
    if (
        feature in train.columns
        and feature in validation.columns
        and feature in test.columns
    )
]


# Fallback if features.json is unavailable
if not common_features:

    excluded = {
        "target_aqi",
        "date",
        "city_name",
        "prediction",
        "absolute_error",
        "error",
    }

    common_features = [
        column
        for column in train.columns
        if (
            column not in excluded
            and column in validation.columns
            and column in test.columns
            and pd.api.types.is_numeric_dtype(
                train[column]
            )
        )
    ]


missing_features = [
    feature
    for feature in production_features
    if feature not in common_features
]


print("=" * 80)
print("V6 FEATURE ALIGNMENT")
print("=" * 80)
print()

print(
    "Production features :",
    len(production_features),
)

print(
    "Common V6 features  :",
    len(common_features),
)

print(
    "Unavailable features:",
    len(missing_features),
)

print()

print(
    "The unavailable production-only features will NOT be "
    "artificially filled."
)

print()


# ============================================================
# SHOW FEATURES
# ============================================================

print("V6 DELTA FEATURES:")
print()

for index, feature in enumerate(
    common_features,
    start=1,
):

    print(
        f"{index:03d}. {feature}"
    )

print()


# ============================================================
# PREPARE DELTA MATRICES
# ============================================================

X_train = train[
    common_features
].copy()

X_validation = validation[
    common_features
].copy()

X_test = test[
    common_features
].copy()


def clean_features(df):

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df.fillna(0)


X_train = clean_features(
    X_train
)

X_validation = clean_features(
    X_validation
)

X_test = clean_features(
    X_test
)


# ============================================================
# TARGETS
# ============================================================

required_columns = [
    "aqi",
    "target_aqi",
]


for column in required_columns:

    if column not in train.columns:
        raise ValueError(
            f"Training dataset missing '{column}'."
        )

    if column not in validation.columns:
        raise ValueError(
            f"Validation dataset missing '{column}'."
        )

    if column not in test.columns:
        raise ValueError(
            f"Test dataset missing '{column}'."
        )


y_train = train[
    "target_aqi"
].astype(float)

y_validation = validation[
    "target_aqi"
].astype(float)

y_test = test[
    "target_aqi"
].astype(float)


current_train = train[
    "aqi"
].astype(float)

current_validation = validation[
    "aqi"
].astype(float)

current_test = test[
    "aqi"
].astype(float)


# ============================================================
# DELTA TARGET
# ============================================================

delta_train = (
    y_train - current_train
)

delta_validation = (
    y_validation - current_validation
)

delta_test = (
    y_test - current_test
)


print("=" * 80)
print("DELTA TARGET")
print("=" * 80)
print()

print(
    "Training delta mean   :",
    round(
        delta_train.mean(),
        2,
    ),
)

print(
    "Validation delta mean :",
    round(
        delta_validation.mean(),
        2,
    ),
)

print(
    "Test delta mean       :",
    round(
        delta_test.mean(),
        2,
    ),
)

print()

print(
    "Training delta std    :",
    round(
        delta_train.std(),
        2,
    ),
)

print(
    "Validation delta std  :",
    round(
        delta_validation.std(),
        2,
    ),
)

print(
    "Test delta std        :",
    round(
        delta_test.std(),
        2,
    ),
)

print()


# ============================================================
# TRAIN DELTA XGBOOST
# ============================================================

print("=" * 80)
print("TRAINING DELTA XGBOOST")
print("=" * 80)
print()


delta_model = XGBRegressor(
    n_estimators=900,
    learning_rate=0.025,
    max_depth=4,
    min_child_weight=8,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.5,
    reg_lambda=5.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)


delta_model.fit(
    X_train,
    delta_train,
)


print("Delta model trained.")
print()


# ============================================================
# DELTA PREDICTIONS
# ============================================================

delta_prediction_validation = (
    delta_model.predict(
        X_validation
    )
)

delta_prediction_test = (
    delta_model.predict(
        X_test
    )
)


# ============================================================
# DELTA METRICS
# ============================================================

print("=" * 80)
print("DELTA MODEL PERFORMANCE")
print("=" * 80)
print()


delta_mae = mean_absolute_error(
    delta_validation,
    delta_prediction_validation,
)

delta_rmse = np.sqrt(
    mean_squared_error(
        delta_validation,
        delta_prediction_validation,
    )
)

delta_r2 = r2_score(
    delta_validation,
    delta_prediction_validation,
)


print(
    f"Delta MAE  : {delta_mae:.4f}"
)

print(
    f"Delta RMSE : {delta_rmse:.4f}"
)

print(
    f"Delta R²   : {delta_r2:.4f}"
)

print()


# ============================================================
# BASELINE PREDICTIONS
#
# IMPORTANT:
# We first look for an already generated production prediction.
#
# The test_error_analysis file definitely contains:
#     prediction
#
# This prevents us from accidentally feeding 59 features into
# the production model that expects 107.
# ============================================================

def get_existing_prediction(
    dataframe,
    name,
):

    if "prediction" in dataframe.columns:

        print(
            f"{name}: using existing production "
            "prediction column."
        )

        return dataframe[
            "prediction"
        ].astype(float).values

    return None


base_validation = get_existing_prediction(
    validation,
    "Validation",
)

base_test = get_existing_prediction(
    test,
    "Test",
)


# ============================================================
# FALLBACK BASELINE
# ============================================================
#
# If validation does not contain an existing production
# prediction, train a comparable 59-feature baseline.
#
# This is ONLY a fallback for experimentation.
# ============================================================

baseline_model = None


if base_validation is None:

    print()
    print("=" * 80)
    print("VALIDATION BASELINE")
    print("=" * 80)
    print()

    print(
        "No existing validation production predictions found."
    )

    print(
        "Training a 59-feature baseline for V6 comparison."
    )

    print()

    baseline_model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.025,
        max_depth=4,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=5.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    baseline_model.fit(
        X_train,
        y_train,
    )

    base_validation = baseline_model.predict(
        X_validation
    )


if base_test is None:

    if baseline_model is None:

        baseline_model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=5.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

        baseline_model.fit(
            X_train,
            y_train,
        )

    base_test = baseline_model.predict(
        X_test
    )


print()


# ============================================================
# BASELINE METRICS
# ============================================================

print("=" * 80)
print("BASELINE PERFORMANCE")
print("=" * 80)
print()


base_val_mae = mean_absolute_error(
    y_validation,
    base_validation,
)

base_val_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        base_validation,
    )
)

base_val_r2 = r2_score(
    y_validation,
    base_validation,
)


base_test_mae = mean_absolute_error(
    y_test,
    base_test,
)

base_test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        base_test,
    )
)

base_test_r2 = r2_score(
    y_test,
    base_test,
)


print(
    f"Validation MAE  : {base_val_mae:.4f}"
)

print(
    f"Validation RMSE : {base_val_rmse:.4f}"
)

print(
    f"Validation R²   : {base_val_r2:.4f}"
)

print()

print(
    f"2026 Test MAE   : {base_test_mae:.4f}"
)

print(
    f"2026 Test RMSE  : {base_test_rmse:.4f}"
)

print(
    f"2026 Test R²    : {base_test_r2:.4f}"
)

print()


# ============================================================
# DELTA-ONLY PREDICTION
# ============================================================

delta_only_validation = (
    current_validation.values
    + delta_prediction_validation
)

delta_only_test = (
    current_test.values
    + delta_prediction_test
)


# ============================================================
# STRATEGY TESTING
# ============================================================

strategies = []


# ------------------------------------------------------------
# BASE
# ------------------------------------------------------------

strategies.append(
    (
        "BASE_XGBOOST",
        base_validation,
        base_test,
    )
)


# ------------------------------------------------------------
# DELTA ONLY
# ------------------------------------------------------------

strategies.append(
    (
        "DELTA_ONLY",
        delta_only_validation,
        delta_only_test,
    )
)


# ------------------------------------------------------------
# BLENDED STRATEGIES
# ------------------------------------------------------------

for weight in [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]:

    corrected_validation = (
        (1.0 - weight)
        * base_validation
        +
        weight
        * delta_only_validation
    )

    corrected_test = (
        (1.0 - weight)
        * base_test
        +
        weight
        * delta_only_test
    )

    strategies.append(
        (
            f"BASE_DELTA_{weight:.2f}",
            corrected_validation,
            corrected_test,
        )
    )


# ============================================================
# EVALUATE STRATEGIES
# ============================================================

results = []


for (
    name,
    pred_val,
    pred_test,
) in strategies:

    val_mae = mean_absolute_error(
        y_validation,
        pred_val,
    )

    val_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            pred_val,
        )
    )

    val_r2 = r2_score(
        y_validation,
        pred_val,
    )

    test_mae = mean_absolute_error(
        y_test,
        pred_test,
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred_test,
        )
    )

    test_r2 = r2_score(
        y_test,
        pred_test,
    )

    results.append(
        {
            "strategy": name,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 80)
print("V6 STRATEGY RESULTS")
print("=" * 80)
print()

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print()


# ============================================================
# BEST STRATEGY
# ============================================================

best = results_df.loc[
    results_df["val_mae"].idxmin()
]

best_strategy = (
    best["strategy"]
)


print("=" * 80)
print("BEST V6 STRATEGY")
print("=" * 80)
print()

print(
    "Strategy:",
    best_strategy,
)

print(
    f"Validation MAE : "
    f"{best['val_mae']:.4f}"
)

print(
    f"Validation RMSE: "
    f"{best['val_rmse']:.4f}"
)

print(
    f"Validation R²  : "
    f"{best['val_r2']:.4f}"
)

print(
    f"2026 MAE       : "
    f"{best['test_mae']:.4f}"
)

print(
    f"2026 RMSE      : "
    f"{best['test_rmse']:.4f}"
)

print(
    f"2026 R²        : "
    f"{best['test_r2']:.4f}"
)

print()


# ============================================================
# GET BEST PREDICTIONS
# ============================================================

strategy_names = [
    strategy[0]
    for strategy in strategies
]

best_index = (
    strategy_names.index(
        best_strategy
    )
)

best_test_predictions = (
    strategies[
        best_index
    ][2]
)


# ============================================================
# EXTREME AQI PERFORMANCE
# ============================================================

extreme_mask = (
    y_test.values >= 200
)


if extreme_mask.sum() > 0:

    extreme_actual = (
        y_test.values[
            extreme_mask
        ]
    )

    extreme_base = (
        base_test[
            extreme_mask
        ]
    )

    extreme_prediction = (
        best_test_predictions[
            extreme_mask
        ]
    )

    extreme_base_mae = (
        mean_absolute_error(
            extreme_actual,
            extreme_base,
        )
    )

    extreme_base_rmse = np.sqrt(
        mean_squared_error(
            extreme_actual,
            extreme_base,
        )
    )

    extreme_mae = (
        mean_absolute_error(
            extreme_actual,
            extreme_prediction,
        )
    )

    extreme_rmse = np.sqrt(
        mean_squared_error(
            extreme_actual,
            extreme_prediction,
        )
    )

    print("=" * 80)
    print("EXTREME AQI PERFORMANCE")
    print("=" * 80)
    print()

    print(
        "Extreme rows:",
        int(
            extreme_mask.sum()
        ),
    )

    print()

    print(
        "BASE XGBOOST"
    )

    print(
        f"Extreme MAE : "
        f"{extreme_base_mae:.4f}"
    )

    print(
        f"Extreme RMSE: "
        f"{extreme_base_rmse:.4f}"
    )

    print()

    print(
        "BEST V6"
    )

    print(
        f"Extreme MAE : "
        f"{extreme_mae:.4f}"
    )

    print(
        f"Extreme RMSE: "
        f"{extreme_rmse:.4f}"
    )

    print()

    print(
        "Extreme MAE improvement:",
        round(
            extreme_base_mae
            - extreme_mae,
            4,
        ),
    )

    print()


# ============================================================
# UPWARD SPIKE PERFORMANCE
# ============================================================

upward_spike_mask = (
    delta_test.values >= 40
)


if upward_spike_mask.sum() > 0:

    actual_spike_change = (
        delta_test.values[
            upward_spike_mask
        ]
    )

    base_spike_change = (
        base_test[
            upward_spike_mask
        ]
        - current_test.values[
            upward_spike_mask
        ]
    )

    v6_spike_change = (
        best_test_predictions[
            upward_spike_mask
        ]
        - current_test.values[
            upward_spike_mask
        ]
    )

    base_spike_mae = mean_absolute_error(
        actual_spike_change,
        base_spike_change,
    )

    v6_spike_mae = mean_absolute_error(
        actual_spike_change,
        v6_spike_change,
    )

    print("=" * 80)
    print("UPWARD SPIKE PERFORMANCE")
    print("=" * 80)
    print()

    print(
        "Upward spike rows:",
        int(
            upward_spike_mask.sum()
        ),
    )

    print(
        "Base spike MAE:",
        round(
            base_spike_mae,
            4,
        ),
    )

    print(
        "V6 spike MAE:",
        round(
            v6_spike_mae,
            4,
        ),
    )

    print(
        "Spike MAE improvement:",
        round(
            base_spike_mae
            - v6_spike_mae,
            4,
        ),
    )

    print()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

output_columns = [
    column
    for column in [
        "city_name",
        "date",
        "aqi",
        "target_aqi",
    ]
    if column in test.columns
]


output = test[
    output_columns
].copy()


output["base_prediction"] = (
    base_test
)

output["delta_prediction"] = (
    delta_prediction_test
)

output["delta_only_prediction"] = (
    delta_only_test
)

output["best_prediction"] = (
    best_test_predictions
)

output["absolute_error"] = (
    output["best_prediction"]
    - output["target_aqi"]
).abs()

output["actual_change"] = (
    output["target_aqi"]
    - output["aqi"]
)

output["predicted_change"] = (
    output["best_prediction"]
    - output["aqi"]
)

output["base_error"] = (
    output["base_prediction"]
    - output["target_aqi"]
)

output["v6_error"] = (
    output["best_prediction"]
    - output["target_aqi"]
)


# ============================================================
# SAVE FILES
# ============================================================

prediction_file = (
    OUTPUT_DIR
    / "v6_predictions.parquet"
)

output.to_parquet(
    prediction_file,
    index=False,
)


results_file = (
    OUTPUT_DIR
    / "v6_strategy_results.csv"
)

results_df.to_csv(
    results_file,
    index=False,
)


# ============================================================
# SAVE DELTA MODEL
# ============================================================

model_file = (
    OUTPUT_DIR
    / "delta_xgboost_model.json"
)

delta_model.save_model(
    model_file
)


# ============================================================
# SAVE FEATURES
# ============================================================

delta_features_file = (
    OUTPUT_DIR
    / "delta_features.json"
)

with open(
    delta_features_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "features": common_features,
            "feature_count": len(
                common_features
            ),
        },
        f,
        indent=2,
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "model": "PearlsAQI Delta XGBoost V6",
    "production_feature_count": len(
        production_features
    ),
    "delta_feature_count": len(
        common_features
    ),
    "best_strategy": best_strategy,
    "validation_mae": float(
        best["val_mae"]
    ),
    "validation_rmse": float(
        best["val_rmse"]
    ),
    "validation_r2": float(
        best["val_r2"]
    ),
    "test_mae": float(
        best["test_mae"]
    ),
    "test_rmse": float(
        best["test_rmse"]
    ),
    "test_r2": float(
        best["test_r2"]
    ),
}


metadata_file = (
    OUTPUT_DIR
    / "v6_metadata.json"
)

with open(
    metadata_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# COMPLETE
# ============================================================

print("=" * 80)
print("PEARLSAQI V6 COMPLETED")
print("=" * 80)
print()

print(
    "Results      :",
    results_file,
)

print(
    "Predictions  :",
    prediction_file,
)

print(
    "Delta model  :",
    model_file,
)

print(
    "Delta features:",
    delta_features_file,
)

print(
    "Metadata     :",
    metadata_file,
)

print()