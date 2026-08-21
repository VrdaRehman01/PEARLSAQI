import os
import time
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# CONFIGURATION
# ==========================================================

TRAIN_FILE = "data/processed/v3/train.parquet"
TEST_FILE = "data/processed/v3/test.parquet"

RESULT_DIR = "models/final_comparison"

os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================================
# FEATURES
# ==========================================================

EXCLUDED_COLUMNS = [
    "target_aqi",
    "city_name",
    "date",
    "season",
]

TARGET_COLUMN = "target_aqi"


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("FINAL AQI MODEL COMPARISON - 2026 HOLDOUT")
print("=" * 70)

print()
print("Loading training data...")

train_df = pd.read_parquet(TRAIN_FILE)

print(f"Training rows: {len(train_df)}")

print()
print("Loading 2026 test data...")

test_df = pd.read_parquet(TEST_FILE)

print(f"Test rows: {len(test_df)}")

print()
print(
    f"Test period: "
    f"{test_df['date'].min()} → {test_df['date'].max()}"
)


# ==========================================================
# FEATURES
# ==========================================================

FEATURE_COLUMNS = [
    column
    for column in train_df.columns
    if column not in EXCLUDED_COLUMNS
]

print()
print(f"Number of features: {len(FEATURE_COLUMNS)}")


X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET_COLUMN]


# ==========================================================
# MODELS
# ==========================================================

models = {

    "HistGradientBoosting": HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=500,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=5.0,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=800,
        learning_rate=0.025,
        max_depth=4,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=5.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1
    ),

    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=0.01))
    ])
}


# ==========================================================
# TRAIN MODELS
# ==========================================================

predictions = {}
results = []


for model_name, model in models.items():

    print()
    print("=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    pred = model.predict(X_test)

    predictions[model_name] = pred

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )

    results.append({
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "training_time_seconds": training_time
    })

    print(f"Training time : {training_time:.2f}s")
    print(f"MAE           : {mae:.4f}")
    print(f"RMSE          : {rmse:.4f}")
    print(f"R²            : {r2:.4f}")


# ==========================================================
# ENSEMBLES
# ==========================================================

print()
print("=" * 70)
print("EVALUATING ENSEMBLES")
print("=" * 70)


ensemble_configs = {

    "HGB_XGB_50_50": (
        0.50,
        0.50,
        0.00
    ),

    "HGB_XGB_60_40": (
        0.60,
        0.40,
        0.00
    ),

    "HGB_Ridge_50_50": (
        0.50,
        0.00,
        0.50
    ),

    "HGB_Ridge_60_40": (
        0.60,
        0.00,
        0.40
    ),

    "HGB_XGB_Ridge_60_30_10": (
        0.60,
        0.30,
        0.10
    ),

    "HGB_XGB_Ridge_50_25_25": (
        0.50,
        0.25,
        0.25
    ),

    "HGB_XGB_Ridge_40_40_20": (
        0.40,
        0.40,
        0.20
    ),
}


ensemble_predictions = {}


for name, weights in ensemble_configs.items():

    hgb_weight, xgb_weight, ridge_weight = weights

    pred = (
        hgb_weight * predictions["HistGradientBoosting"]
        +
        xgb_weight * predictions["XGBoost"]
        +
        ridge_weight * predictions["Ridge"]
    )

    ensemble_predictions[name] = pred

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )

    results.append({
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "training_time_seconds": 0.0
    })


# ==========================================================
# RESULTS
# ==========================================================

results_df = pd.DataFrame(results)


# Sort primarily by MAE
results_df = results_df.sort_values(
    "mae",
    ascending=True
)


print()
print("=" * 70)
print("FINAL 2026 RESULTS - SORTED BY MAE")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==========================================================
# TOP MODELS BY EACH METRIC
# ==========================================================

print()
print("=" * 70)
print("BEST BY MAE")
print("=" * 70)

best_mae = results_df.loc[
    results_df["mae"].idxmin()
]

print(best_mae.to_string())


print()
print("=" * 70)
print("BEST BY RMSE")
print("=" * 70)

best_rmse = results_df.loc[
    results_df["rmse"].idxmin()
]

print(best_rmse.to_string())


print()
print("=" * 70)
print("BEST BY R²")
print("=" * 70)

best_r2 = results_df.loc[
    results_df["r2"].idxmax()
]

print(best_r2.to_string())


# ==========================================================
# PREDICTION ACCURACY
# ==========================================================

print()
print("=" * 70)
print("PREDICTION ACCURACY")
print("=" * 70)


def accuracy_stats(actual, prediction):

    errors = np.abs(
        actual - prediction
    )

    return {
        "within_10": np.mean(errors <= 10) * 100,
        "within_20": np.mean(errors <= 20) * 100,
        "within_30": np.mean(errors <= 30) * 100,
    }


accuracy_rows = []


all_predictions = {}

all_predictions.update(
    predictions
)

all_predictions.update(
    ensemble_predictions
)


for name, pred in all_predictions.items():

    stats = accuracy_stats(
        y_test.values,
        pred
    )

    accuracy_rows.append({
        "model": name,
        "within_10": stats["within_10"],
        "within_20": stats["within_20"],
        "within_30": stats["within_30"]
    })


accuracy_df = pd.DataFrame(
    accuracy_rows
)

accuracy_df = accuracy_df.sort_values(
    "within_20",
    ascending=False
)


print(
    accuracy_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ==========================================================
# AQI RANGE ANALYSIS
# ==========================================================

print()
print("=" * 70)
print("AQI RANGE ANALYSIS - BEST MAE MODEL")
print("=" * 70)


best_model_name = best_mae["model"]

best_prediction = all_predictions[
    best_model_name
]


analysis_df = test_df[
    [
        "city_name",
        "date",
        "aqi",
        "target_aqi"
    ]
].copy()

analysis_df["prediction"] = best_prediction

analysis_df["absolute_error"] = np.abs(
    analysis_df["target_aqi"]
    -
    analysis_df["prediction"]
)


def get_aqi_range(aqi):

    if aqi <= 50:
        return "0-50 Good"

    elif aqi <= 100:
        return "51-100 Moderate"

    elif aqi <= 150:
        return "101-150 Unhealthy"

    elif aqi <= 200:
        return "151-200 Very Unhealthy"

    elif aqi <= 300:
        return "201-300 Severe"

    else:
        return "301+ Extreme"


analysis_df["aqi_range"] = (
    analysis_df["target_aqi"]
    .apply(get_aqi_range)
)


range_metrics = (
    analysis_df
    .groupby("aqi_range")
    .agg(
        rows=("target_aqi", "size"),
        mae=("absolute_error", "mean"),
        actual_mean=("target_aqi", "mean"),
        prediction_mean=("prediction", "mean"),
        max_actual=("target_aqi", "max"),
        max_prediction=("prediction", "max")
    )
    .reset_index()
)


print(
    range_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ==========================================================
# CITY ANALYSIS
# ==========================================================

print()
print("=" * 70)
print("CITY PERFORMANCE - BEST MAE MODEL")
print("=" * 70)


city_metrics = (
    analysis_df
    .groupby("city_name")
    .agg(
        rows=("target_aqi", "size"),
        mae=("absolute_error", "mean"),
        rmse=(
            "absolute_error",
            lambda x: np.sqrt(np.mean(x ** 2))
        ),
        mean_actual=("target_aqi", "mean"),
        mean_prediction=("prediction", "mean"),
        max_actual=("target_aqi", "max"),
        max_prediction=("prediction", "max")
    )
    .reset_index()
    .sort_values(
        "mae",
        ascending=False
    )
)


print(
    city_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ==========================================================
# WORST PREDICTIONS
# ==========================================================

print()
print("=" * 70)
print("20 WORST PREDICTIONS")
print("=" * 70)


worst_predictions = (
    analysis_df
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(20)
)


print(
    worst_predictions.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ==========================================================
# SAVE RESULTS
# ==========================================================

results_file = os.path.join(
    RESULT_DIR,
    "final_model_comparison.csv"
)

accuracy_file = os.path.join(
    RESULT_DIR,
    "prediction_accuracy.csv"
)

city_file = os.path.join(
    RESULT_DIR,
    "city_metrics.csv"
)

range_file = os.path.join(
    RESULT_DIR,
    "aqi_range_metrics.csv"
)

worst_file = os.path.join(
    RESULT_DIR,
    "worst_predictions.csv"
)


results_df.to_csv(
    results_file,
    index=False
)

accuracy_df.to_csv(
    accuracy_file,
    index=False
)

city_metrics.to_csv(
    city_file,
    index=False
)

range_metrics.to_csv(
    range_file,
    index=False
)

worst_predictions.to_csv(
    worst_file,
    index=False
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()
print("=" * 70)
print("FINAL HOLDOUT SUMMARY")
print("=" * 70)

print(
    f"Best MAE model  : {best_mae['model']}"
)

print(
    f"MAE             : {best_mae['mae']:.4f}"
)

print(
    f"RMSE            : {best_mae['rmse']:.4f}"
)

print(
    f"R²              : {best_mae['r2']:.4f}"
)

print()
print(f"Results saved to: {results_file}")
print(f"Accuracy saved to: {accuracy_file}")
print(f"City metrics saved to: {city_file}")
print(f"Range metrics saved to: {range_file}")
print(f"Worst predictions saved to: {worst_file}")

print()
print("=" * 70)
print("2026 HOLDOUT EVALUATION COMPLETE")
print("=" * 70)