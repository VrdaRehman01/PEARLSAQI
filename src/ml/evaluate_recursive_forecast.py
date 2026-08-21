"""
PEARLSAQI - RECURSIVE FORECAST EVALUATION

Evaluates the production XGBoost model using rolling historical origins.

For each historical origin:
    +1 day -> 24h forecast
    +2 days -> 48h forecast
    +3 days -> 72h forecast

IMPORTANT:
- Uses only information available at the forecast origin.
- Never uses future actual AQI as an input.
- Uses the same recursive logic as production.
- Does NOT modify the production model.
"""

from pathlib import Path
from datetime import timedelta
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.features.feature_builder_v4 import (
    add_lag_features,
    add_aqi_rolling_features,
    add_aqi_trend_features,
    add_pollution_features,
    add_weather_features,
    add_regime_features,
    add_city_history_features,
    add_calendar_features,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "features_v4.parquet"
)

MODEL_FILE = (
    ROOT
    / "models"
    / "final_production_xgboost"
    / "final_xgboost_model.json"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "evaluation"
    / "recursive_forecast"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "recursive_results.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "recursive_summary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

HORIZONS = [1, 2, 3]

HISTORY_DAYS = 60

# Evaluate only the 2026 period that has enough future
# observations available.
EVALUATION_START = pd.Timestamp("2026-01-01")

# Last origin must leave room for +3 days.
EVALUATION_END = pd.Timestamp("2026-08-03")


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
):

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    within_10 = (
        np.abs(
            y_true - predictions
        )
        <= 10
    ).mean() * 100

    within_20 = (
        np.abs(
            y_true - predictions
        )
        <= 20
    ).mean() * 100

    within_30 = (
        np.abs(
            y_true - predictions
        )
        <= 30
    ).mean() * 100

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "within_10": float(within_10),
        "within_20": float(within_20),
        "within_30": float(within_30),
    }


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Production model not found:\n"
            f"{MODEL_FILE}"
        )

    model = xgb.XGBRegressor()

    model.load_model(
        str(MODEL_FILE)
    )

    print(
        f"Production model loaded:\n"
        f"{MODEL_FILE}"
    )

    return model


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n"
            f"{DATA_FILE}"
        )

    df = pd.read_parquet(
        DATA_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        [
            "city_id",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Cities: "
        f"{df['city_name'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"-> "
        f"{df['date'].max().date()}"
    )

    return df


# ============================================================
# FEATURE REBUILD
# ============================================================

def remove_engineered_columns(df):

    columns = [

        "aqi_lag_1",
        "aqi_lag_2",
        "aqi_lag_3",
        "aqi_lag_7",
        "aqi_lag_14",

        "aqi_change_1d",
        "aqi_change_2d",
        "aqi_change_3d",
        "aqi_change_7d",

        "aqi_acceleration_1d",
        "aqi_acceleration_2d",

        "aqi_rolling_3",
        "aqi_std_3",
        "aqi_min_3",
        "aqi_max_3",
        "aqi_range_3",

        "aqi_rolling_7",
        "aqi_std_7",
        "aqi_min_7",
        "aqi_max_7",
        "aqi_range_7",

        "aqi_rolling_14",
        "aqi_std_14",
        "aqi_min_14",
        "aqi_max_14",
        "aqi_range_14",

        "aqi_trend_3d",
        "aqi_trend_7d",

        "aqi_distance_from_max_7",
        "aqi_distance_from_max_14",

        "aqi_percentile_7",
        "aqi_percentile_14",

        "pm25_change_1d",
        "pm25_change_3d",
        "pm25_rolling_3",
        "pm25_rolling_7",

        "pm10_change_1d",
        "pm10_change_3d",
        "pm10_rolling_3",
        "pm10_rolling_7",

        "no2_change_1d",
        "no2_change_3d",
        "no2_rolling_3",
        "no2_rolling_7",

        "so2_change_1d",
        "so2_change_3d",
        "so2_rolling_3",
        "so2_rolling_7",

        "co_change_1d",
        "co_change_3d",
        "co_rolling_3",
        "co_rolling_7",

        "o3_change_1d",
        "o3_change_3d",
        "o3_rolling_3",
        "o3_rolling_7",

        "pm25_trend_3d",
        "pm25_trend_7d",
        "pm10_trend_3d",
        "pm10_trend_7d",

        "pm25_pm10_ratio",
        "pm25_no2_interaction",
        "pm25_co_interaction",
        "pm25_o3_interaction",
        "pollution_sum",

        "temperature_change_1d",
        "humidity_change_1d",
        "windspeed_change_1d",
        "precipitation_change_1d",
        "precipitation_rolling_3",
        "windspeed_rolling_3",

        "aqi_moderate",
        "aqi_unhealthy",
        "aqi_very_unhealthy",
        "aqi_severe",
        "aqi_extreme",
        "aqi_regime_change",
        "high_aqi_recent",
        "extreme_aqi_recent",

        "city_aqi_mean",
        "city_aqi_std",
        "city_pm25_mean",
        "city_recent_mean",
        "city_recent_max",

        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]

    existing = [
        c
        for c in columns
        if c in df.columns
    ]

    return df.drop(
        columns=existing
    )


def rebuild_features(df):

    work = df.copy()

    if "target_aqi" in work.columns:

        work = work.drop(
            columns=["target_aqi"]
        )

    work = remove_engineered_columns(
        work
    )

    dates = pd.to_datetime(
        work["date"]
    )

    work["year"] = dates.dt.year
    work["month"] = dates.dt.month
    work["day"] = dates.dt.day
    work["day_of_week"] = dates.dt.dayofweek
    work["day_of_year"] = dates.dt.dayofyear

    work["week_of_year"] = (
        dates.dt.isocalendar()
        .week
        .astype(int)
    )

    work["is_weekend"] = (
        work["day_of_week"] >= 5
    ).astype(int)

    work = add_lag_features(work)
    work = add_aqi_rolling_features(work)
    work = add_aqi_trend_features(work)
    work = add_pollution_features(work)
    work = add_weather_features(work)
    work = add_regime_features(work)
    work = add_city_history_features(work)
    work = add_calendar_features(work)

    return work


# ============================================================
# MODEL FEATURES
# ============================================================

def get_feature_names(model):

    names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if names is not None:

        return list(names)

    booster = model.get_booster()

    if booster.feature_names:

        return list(
            booster.feature_names
        )

    raise ValueError(
        "Could not determine model feature names."
    )


# ============================================================
# SAFE INPUT
# ============================================================

def prepare_input(
    row,
    feature_names,
):

    values = []

    for feature in feature_names:

        if feature not in row.index:

            raise ValueError(
                f"Missing feature: {feature}"
            )

        value = row[feature]

        if pd.isna(value):

            value = 0.0

        try:

            value = float(value)

            if not np.isfinite(value):

                value = 0.0

        except Exception:

            value = 0.0

        values.append(
            value
        )

    return pd.DataFrame(
        [values],
        columns=feature_names,
    )


# ============================================================
# CREATE FUTURE STATE
# ============================================================

def create_future_row(
    latest_row,
    future_date,
    seed_aqi,
):

    row = latest_row.copy()

    row["date"] = future_date

    row["aqi"] = float(
        seed_aqi
    )

    return row


# ============================================================
# FORECAST ONE ORIGIN
# ============================================================

def forecast_origin(
    city_history,
    origin_date,
    model,
    feature_names,
):

    city_history = (
        city_history
        .sort_values("date")
        .copy()
    )

    available = city_history[
        city_history["date"]
        <= origin_date
    ]

    if available.empty:

        return []

    history = available.tail(
        HISTORY_DAYS
    ).copy()

    latest_row = (
        history
        .sort_values("date")
        .iloc[-1]
    )

    latest_aqi = float(
        latest_row["aqi"]
    )

    previous_prediction = (
        latest_aqi
    )

    results = []

    for horizon in HORIZONS:

        forecast_date = (
            origin_date
            + timedelta(
                days=horizon
            )
        )

        seed_aqi = (
            latest_aqi
            if horizon == 1
            else previous_prediction
        )

        latest_state = (
            history
            .sort_values("date")
            .iloc[-1]
        )

        future_row = create_future_row(
            latest_state,
            forecast_date,
            seed_aqi,
        )

        working = pd.concat(
            [
                history,
                pd.DataFrame(
                    [future_row]
                ),
            ],
            ignore_index=True,
        )

        working = (
            working
            .sort_values("date")
            .reset_index(drop=True)
        )

        rebuilt = rebuild_features(
            working
        )

        future = rebuilt[
            rebuilt["date"]
            == forecast_date
        ]

        if future.empty:

            raise RuntimeError(
                f"Could not create features "
                f"for {forecast_date}"
            )

        row = future.iloc[-1]

        X = prepare_input(
            row,
            feature_names,
        )

        prediction = float(
            model.predict(X)[0]
        )

        prediction = max(
            0.0,
            prediction
        )

        results.append({

            "origin_date":
                origin_date,

            "forecast_date":
                forecast_date,

            "horizon":
                horizon,

            "prediction":
                prediction,

            "actual":
                np.nan,
        })

        # Recursive update
        next_row = future_row.copy()

        next_row["aqi"] = (
            prediction
        )

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [next_row]
                ),
            ],
            ignore_index=True,
        )

        history = (
            history
            .sort_values("date")
            .tail(
                HISTORY_DAYS
            )
            .reset_index(drop=True)
        )

        previous_prediction = (
            prediction
        )

    return results


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "RECURSIVE 24H / 48H / 72H "
        "FORECAST EVALUATION"
    )
    print("=" * 70)

    df = load_data()

    model = load_model()

    feature_names = get_feature_names(
        model
    )

    print(
        f"Model features: "
        f"{len(feature_names)}"
    )

    print()
    print(
        f"Evaluation period: "
        f"{EVALUATION_START.date()} "
        f"-> "
        f"{EVALUATION_END.date()}"
    )

    all_results = []

    cities = (
        df[
            [
                "city_id",
                "city_name",
            ]
        ]
        .drop_duplicates()
        .sort_values("city_id")
    )

    for _, city in cities.iterrows():

        city_id = int(
            city["city_id"]
        )

        city_name = str(
            city["city_name"]
        )

        print(
            f"\nEvaluating {city_name}..."
        )

        city_df = df[
            df["city_id"]
            == city_id
        ].copy()

        origin_dates = pd.date_range(
            EVALUATION_START,
            EVALUATION_END,
            freq="D",
        )

        for origin_date in origin_dates:

            results = forecast_origin(
                city_history=city_df,
                origin_date=origin_date,
                model=model,
                feature_names=feature_names,
            )

            for result in results:

                actual = city_df[
                    city_df["date"]
                    == result["forecast_date"]
                ]

                if actual.empty:

                    continue

                result["city_id"] = (
                    city_id
                )

                result["city_name"] = (
                    city_name
                )

                result["actual"] = float(
                    actual["target_aqi"].iloc[0]
                )

                result["error"] = (
                    result["prediction"]
                    - result["actual"]
                )

                result["absolute_error"] = abs(
                    result["error"]
                )

                all_results.append(
                    result
                )

        print(
            f"  completed"
        )

    results_df = pd.DataFrame(
        all_results
    )

    if results_df.empty:

        raise RuntimeError(
            "No evaluation results generated."
        )

    results_df = results_df.sort_values(
        [
            "city_name",
            "origin_date",
            "horizon",
        ]
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("HORIZON PERFORMANCE")
    print("=" * 70)

    summary = {}

    for horizon in HORIZONS:

        subset = results_df[
            results_df["horizon"]
            == horizon
        ]

        metrics = calculate_metrics(
            subset["actual"],
            subset["prediction"],
        )

        summary[
            f"{horizon * 24}h"
        ] = metrics

        print()
        print(
            f"{horizon * 24}h"
        )

        print(
            f"MAE        : "
            f"{metrics['mae']:.4f}"
        )

        print(
            f"RMSE       : "
            f"{metrics['rmse']:.4f}"
        )

        print(
            f"R2         : "
            f"{metrics['r2']:.4f}"
        )

        print(
            f"Within ±10 : "
            f"{metrics['within_10']:.2f}%"
        )

        print(
            f"Within ±20 : "
            f"{metrics['within_20']:.2f}%"
        )

        print(
            f"Within ±30 : "
            f"{metrics['within_30']:.2f}%"
        )

    # --------------------------------------------------------
    # City-level performance
    # --------------------------------------------------------

    city_summary = []

    for city_name, group in (
        results_df
        .groupby("city_name")
    ):

        metrics = calculate_metrics(
            group["actual"],
            group["prediction"],
        )

        city_summary.append({

            "city_name":
                city_name,

            **metrics,
        })

    city_summary_df = (
        pd.DataFrame(city_summary)
        .sort_values("mae")
    )

    print()
    print("=" * 70)
    print("CITY PERFORMANCE")
    print("=" * 70)

    print(
        city_summary_df
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Worst predictions
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP 20 WORST RECURSIVE PREDICTIONS")
    print("=" * 70)

    print(
        results_df
        .sort_values(
            "absolute_error",
            ascending=False,
        )
        [
            [
                "city_name",
                "origin_date",
                "forecast_date",
                "horizon",
                "actual",
                "prediction",
                "absolute_error",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    output = {

        "evaluation_type":
            "recursive_forecast",

        "model":
            str(MODEL_FILE),

        "evaluation_start":
            str(EVALUATION_START.date()),

        "evaluation_end":
            str(EVALUATION_END.date()),

        "history_days":
            HISTORY_DAYS,

        "forecast_horizons":
            [
                "24h",
                "48h",
                "72h",
            ],

        "rows":
            int(len(results_df)),

        "horizon_metrics":
            summary,

    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Results : {RESULTS_FILE}"
    )

    print(
        f"Summary : {SUMMARY_FILE}"
    )


if __name__ == "__main__":

    main()