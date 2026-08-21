"""
PEARLSAQI FORECAST ENGINE V1

Generates recursive 24h / 48h / 72h AQI forecasts
for all 12 Pakistan cities using the existing
production XGBoost model and V4 feature pipeline.

IMPORTANT:
- Does NOT use future actual AQI.
- Does NOT leak target_aqi.
- Uses the latest known pollution/weather state.
- Future AQI is generated recursively.
"""

from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

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

FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "features_v4.parquet"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "final_production_xgboost"
)

PREDICTION_FILE = (
    MODEL_DIR
    / "predictions"
    / "latest_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "forecast"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "forecast_predictions.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "forecast_predictions.csv"
)

OUTPUT_METADATA = (
    OUTPUT_DIR
    / "forecast_metadata.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

HORIZONS = [1, 2, 3]

HORIZON_LABELS = {
    1: "24h",
    2: "48h",
    3: "72h",
}

FORECAST_HISTORY_DAYS = 60


# ============================================================
# AQI CATEGORY
# ============================================================

def get_category(aqi):

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


def get_health_message(category):

    messages = {

        "Good":
            "Air quality is satisfactory.",

        "Moderate":
            "Air quality is acceptable.",

        "Unhealthy for Sensitive Groups":
            "Sensitive groups may experience health effects.",

        "Unhealthy":
            "Everyone may begin to experience health effects.",

        "Very Unhealthy":
            "Health alert: everyone may experience more serious effects.",

        "Hazardous":
            "Health emergency conditions. Everyone is likely to be affected.",
    }

    return messages.get(
        category,
        "Air quality information available.",
    )


# ============================================================
# FIND MODEL
# ============================================================

def find_model():

    candidates = [

        MODEL_DIR / "model.json",

        MODEL_DIR / "xgboost_model.json",

        MODEL_DIR / "final_model.json",

        MODEL_DIR / "production_model.json",

    ]

    for path in candidates:

        if path.exists():
            return path

    json_models = list(
        MODEL_DIR.glob("*.json")
    )

    # Ignore metrics/metadata JSON files
    for path in json_models:

        name = path.name.lower()

        if (
            "metric" not in name
            and "metadata" not in name
            and "feature" not in name
            and "prediction" not in name
        ):
            return path

    raise FileNotFoundError(
        f"No XGBoost model JSON found in {MODEL_DIR}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model_path = find_model()

    print()
    print("Loading production XGBoost model...")
    print(f"Model: {model_path}")

    model = xgb.XGBRegressor()

    model.load_model(
        str(model_path)
    )

    return model, model_path


# ============================================================
# LOAD PRODUCTION FEATURE NAMES
# ============================================================

def load_feature_names(model):

    """
    Prefer the feature names embedded in XGBoost.

    If unavailable, fall back to the model's
    feature count and the existing feature dataset.
    """

    feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if feature_names is not None:

        return list(feature_names)

    booster = model.get_booster()

    if booster.feature_names:

        return list(
            booster.feature_names
        )

    return None


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_features():

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    print()
    print("Loading V4 feature dataset...")

    df = pd.read_parquet(
        FEATURE_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        ["city_id", "date"]
    ).reset_index(drop=True)

    print(
        f"Feature rows: {len(df):,}"
    )

    print(
        f"Cities: {df['city_name'].nunique()}"
    )

    return df


# ============================================================
# BUILD FEATURES FOR FORECAST
# ============================================================

def rebuild_features(df):

    """
    Rebuild the V4 engineered features.

    The functions operate only on historical/current
    information, so target_aqi is excluded.
    """

    work = df.copy()

    # Remove target because it must never enter inference.
    if "target_aqi" in work.columns:

        work = work.drop(
            columns=["target_aqi"]
        )

    # Remove existing engineered columns.
    # They will be regenerated from raw values.
    engineered_columns = [

        # AQI
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

        # Pollution
        "pm25_change_1d",
        "pm25_change_3d",
        "pm25_rolling_3",
        "pm25_rolling_7",
        "pm25_trend_3d",
        "pm25_trend_7d",

        "pm10_change_1d",
        "pm10_change_3d",
        "pm10_rolling_3",
        "pm10_rolling_7",
        "pm10_trend_3d",
        "pm10_trend_7d",

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

        "pm25_pm10_ratio",
        "pm25_no2_interaction",
        "pm25_co_interaction",
        "pm25_o3_interaction",
        "pollution_sum",

        # Weather
        "temperature_change_1d",
        "humidity_change_1d",
        "windspeed_change_1d",
        "precipitation_change_1d",
        "precipitation_rolling_3",
        "windspeed_rolling_3",

        # Regime
        "aqi_moderate",
        "aqi_unhealthy",
        "aqi_very_unhealthy",
        "aqi_severe",
        "aqi_extreme",
        "aqi_regime_change",
        "high_aqi_recent",
        "extreme_aqi_recent",

        # City
        "city_aqi_mean",
        "city_aqi_std",
        "city_pm25_mean",
        "city_recent_mean",
        "city_recent_max",

        # Calendar
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]

    existing_engineered = [
        c
        for c in engineered_columns
        if c in work.columns
    ]

    if existing_engineered:

        work = work.drop(
            columns=existing_engineered
        )

    # Calendar
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
# CREATE FUTURE ROW
# ============================================================

def create_future_row(
    latest_row,
    future_date,
    predicted_aqi,
):

    row = latest_row.copy()

    row["date"] = future_date

    row["aqi"] = float(
        predicted_aqi
    )

    # --------------------------------------------------------
    # Future pollutant assumption
    # --------------------------------------------------------
    #
    # Until dedicated pollutant forecasting models exist,
    # we use the most recently observed pollutant state.
    #
    # This is intentionally conservative.
    #

    pollutant_columns = [
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3",
    ]

    for column in pollutant_columns:

        if column in row.index:

            row[column] = float(
                latest_row[column]
            )

    # --------------------------------------------------------
    # Future weather assumption
    # --------------------------------------------------------
    #
    # V1 uses persistence.
    # Real weather forecasts will replace this later.
    #

    weather_columns = [
        "temperature",
        "humidity",
        "precipitation",
        "windspeed",
    ]

    for column in weather_columns:

        if column in row.index:

            row[column] = float(
                latest_row[column]
            )

    # target MUST NOT be used
    if "target_aqi" in row.index:

        row["target_aqi"] = np.nan

    return row


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    feature_df,
    feature_names,
):

    row = feature_df.iloc[-1].copy()

    available = set(
        feature_df.columns
    )

    missing = [
        feature
        for feature in feature_names
        if feature not in available
    ]

    if missing:

        raise ValueError(
            "Forecast feature mismatch.\n"
            f"Missing features:\n{missing}"
        )

    X = pd.DataFrame(
        [[
            row[feature]
            for feature in feature_names
        ]],
        columns=feature_names,
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if X.isna().any().any():

        missing_values = (
            X.columns[
                X.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "NaN values detected in "
            f"forecast features:\n{missing_values}"
        )

    return X


# ============================================================
# FORECAST ONE CITY
# ============================================================

def forecast_city(
    city_df,
    model,
    feature_names,
):

    city_df = city_df.sort_values(
        "date"
    ).copy()

    # Keep enough historical rows for
    # 14/30-day features.
    history = city_df.tail(
        FORECAST_HISTORY_DAYS
    ).copy()

    latest_date = pd.to_datetime(
        history["date"].iloc[-1]
    )

    latest_aqi = float(
        history["aqi"].iloc[-1]
    )

    city_name = str(
        history["city_name"].iloc[-1]
    )

    city_id = int(
        history["city_id"].iloc[-1]
    )

    results = []

    previous_prediction = latest_aqi

    for horizon in HORIZONS:

        forecast_date = (
            latest_date
            + timedelta(days=horizon)
        )

        # For recursive prediction:
        # use the latest forecast state.
        if horizon == 1:

            seed_aqi = latest_aqi

        else:

            seed_aqi = previous_prediction

        latest_state = history.iloc[-1]

        future_row = create_future_row(
            latest_state,
            forecast_date,
            seed_aqi,
        )

        # Append future state.
        working = pd.concat(
            [
                history,
                pd.DataFrame(
                    [future_row]
                ),
            ],
            ignore_index=True,
        )

        working = working.sort_values(
            "date"
        ).reset_index(drop=True)

        # Rebuild V4 features.
        rebuilt = rebuild_features(
            working
        )

        # Select future row.
        future_features = rebuilt[
            rebuilt["date"]
            == forecast_date
        ].copy()

        if future_features.empty:

            raise RuntimeError(
                f"Could not build forecast row "
                f"for {city_name} {forecast_date}"
            )

        # The last row is the forecast state.
        future_features = (
            future_features
            .sort_values("date")
            .tail(1)
        )

        X = prepare_model_input(
            future_features,
            feature_names,
        )

        prediction = float(
            model.predict(X)[0]
        )

        # AQI cannot be negative.
        prediction = max(
            0.0,
            prediction,
        )

        category = get_category(
            prediction
        )

        # Change relative to previous state.
        change = (
            prediction
            - previous_prediction
        )

        results.append({

            "city_id":
                city_id,

            "city_name":
                city_name,

            "forecast_generated_at":
                datetime.now().isoformat(),

            "base_date":
                latest_date.strftime(
                    "%Y-%m-%d"
                ),

            "forecast_date":
                forecast_date.strftime(
                    "%Y-%m-%d"
                ),

            "horizon":
                horizon,

            "horizon_label":
                HORIZON_LABELS[horizon],

            "current_aqi":
                round(
                    latest_aqi,
                    2,
                ),

            "predicted_aqi":
                round(
                    prediction,
                    2,
                ),

            "change_from_previous":
                round(
                    change,
                    2,
                ),

            "aqi_category":
                category,

            "health_message":
                get_health_message(
                    category
                ),

            "forecast_method":
                "V4 XGBoost recursive forecast",

        })

        previous_prediction = prediction

        # ----------------------------------------------------
        # IMPORTANT:
        # Carry the predicted AQI into the next step.
        # ----------------------------------------------------

        next_row = future_row.copy()

        next_row["aqi"] = prediction

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [next_row]
                ),
            ],
            ignore_index=True,
        )

        history = history.sort_values(
            "date"
        ).tail(
            FORECAST_HISTORY_DAYS
        )

    return results


# ============================================================
# MAIN FORECAST
# ============================================================

def generate_forecast():

    print()
    print("=" * 70)
    print("PEARLSAQI FORECAST ENGINE V1")
    print("=" * 70)

    print()
    print(
        "Forecast horizons: 24h / 48h / 72h"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_features()

    model, model_path = load_model()

    feature_names = load_feature_names(
        model
    )

    if feature_names is None:

        raise ValueError(
            "Could not determine production "
            "XGBoost feature names."
        )

    print()
    print(
        f"Production model features: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    dataset_features = [
        column
        for column in df.columns
        if column not in [
            "target_aqi",
            "city_name",
            "date",
        ]
    ]

    missing = [
        feature
        for feature in feature_names
        if feature not in dataset_features
    ]

    if missing:

        raise ValueError(
            "Production feature dataset is "
            "missing model features:\n"
            + "\n".join(
                f"- {x}"
                for x in missing
            )
        )

    # --------------------------------------------------------
    # Forecast every city
    # --------------------------------------------------------

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

    print()
    print(
        f"Generating forecasts for "
        f"{len(cities)} cities..."
    )

    for _, city in cities.iterrows():

        city_id = int(
            city["city_id"]
        )

        city_name = str(
            city["city_name"]
        )

        print()
        print(
            f"Forecasting: {city_name}"
        )

        city_df = df[
            df["city_id"]
            == city_id
        ].copy()

        try:

            results = forecast_city(
                city_df,
                model,
                feature_names,
            )

            all_results.extend(
                results
            )

            for result in results:

                print(
                    f"  "
                    f"{result['horizon_label']} "
                    f"→ "
                    f"{result['predicted_aqi']:.2f} "
                    f"({result['aqi_category']})"
                )

        except Exception as error:

            print(
                f"  ERROR: {error}"
            )

            raise

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    forecast_df = pd.DataFrame(
        all_results
    )

    if forecast_df.empty:

        raise ValueError(
            "No forecast results generated."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    forecast_df.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    forecast_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    metadata = {

        "project":
            "PearlsAQI",

        "engine":
            "forecast_engine_v1",

        "model":
            str(model_path),

        "feature_count":
            len(feature_names),

        "cities":
            int(
                forecast_df[
                    "city_name"
                ]
                .nunique()
            ),

        "forecasts_per_city":
            3,

        "total_forecasts":
            len(forecast_df),

        "horizons":
            [
                "24h",
                "48h",
                "72h",
            ],

        "generated_at":
            datetime.now().isoformat(),

        "weather_strategy":
            "latest known state persistence",

        "pollutant_strategy":
            "latest known state persistence",

        "recursive":
            True,

        "target_leakage":
            False,

    }

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FORECAST COMPLETE")
    print("=" * 70)

    print(
        f"Cities       : "
        f"{metadata['cities']}"
    )

    print(
        f"Forecasts    : "
        f"{metadata['total_forecasts']}"
    )

    print(
        f"Horizons     : "
        f"24h / 48h / 72h"
    )

    print()
    print(
        f"Parquet      : "
        f"{OUTPUT_PARQUET}"
    )

    print(
        f"CSV          : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Metadata     : "
        f"{OUTPUT_METADATA}"
    )

    print()
    print(
        "PearlsAQI Forecast Engine V1 "
        "completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    generate_forecast()