import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from src.database.connection import get_connection

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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "models/final_production_xgboost/"
    "final_xgboost_model.json"
)

FEATURES_PATH = (
    "models/final_production_xgboost/"
    "features.json"
)


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"


# ============================================================
# HEALTH MESSAGE
# ============================================================

def get_health_message(aqi):

    if aqi <= 50:
        return "Air quality is good."

    elif aqi <= 100:
        return "Air quality is acceptable."

    elif aqi <= 150:
        return (
            "Sensitive groups may experience health effects."
        )

    elif aqi <= 200:
        return (
            "Everyone may begin to experience health effects."
        )

    elif aqi <= 300:
        return (
            "Health alert: everyone may experience "
            "more serious health effects."
        )

    else:
        return (
            "Health emergency conditions. "
            "Avoid prolonged outdoor exposure."
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading production XGBoost model...")

    model = xgb.XGBRegressor()

    model.load_model(MODEL_PATH)

    print("Model loaded.")

    return model


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():

    print("Loading feature list...")

    with open(
        FEATURES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # Support either:
    #
    # ["feature1", "feature2", ...]
    #
    # or:
    #
    # {"features": [...]}

    if isinstance(data, list):

        features = data

    elif isinstance(data, dict):

        features = data.get("features")

    else:

        raise ValueError(
            "Invalid features.json format."
        )

    if not features:

        raise ValueError(
            "Feature list is empty."
        )

    print(
        f"Features loaded: {len(features)}"
    )

    return features


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_data():

    print("Loading AQI + weather data...")

    conn = get_connection()

    query = """
    SELECT
        a.city_id,
        c.city_name,
        a.date,
        a.aqi,
        a.pm25,
        a.pm10,
        a.no2,
        a.so2,
        a.co,
        a.o3,

        w.temperature,
        w.humidity,
        w.precipitation,
        w.windspeed

    FROM aqi a

    INNER JOIN cities c
        ON a.city_id = c.city_id

    INNER JOIN weather w
        ON a.city_id = w.city_id
        AND a.date = w.date

    ORDER BY
        a.city_id,
        a.date
    """

    df = conn.execute(query).fetchdf()

    conn.close()

    print(
        f"Rows loaded: {len(df)}"
    )

    return df


# ============================================================
# BUILD FEATURES
# ============================================================

def build_prediction_features(df):

    print("Building V4 features...")

    df = df.copy()

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    dates = pd.to_datetime(df["date"])

    df["year"] = dates.dt.year
    df["month"] = dates.dt.month
    df["day"] = dates.dt.day
    df["day_of_week"] = dates.dt.dayofweek
    df["day_of_year"] = dates.dt.dayofyear

    df["week_of_year"] = (
        dates.dt.isocalendar()
        .week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # V4 feature groups
    # --------------------------------------------------------

    df = add_lag_features(df)

    df = add_aqi_rolling_features(df)

    df = add_aqi_trend_features(df)

    df = add_pollution_features(df)

    df = add_weather_features(df)

    df = add_regime_features(df)

    df = add_city_history_features(df)

    df = add_calendar_features(df)

    return df


# ============================================================
# PREDICT
# ============================================================

def predict():

    print()
    print("=" * 70)
    print("AQI PRODUCTION PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model = load_model()

    features = load_features()

    df = load_data()

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    df = build_prediction_features(df)

    # --------------------------------------------------------
    # Remove rows without enough history
    # --------------------------------------------------------

    df = df.dropna(
        subset=features
    ).copy()

    # --------------------------------------------------------
    # Latest observation per city
    # --------------------------------------------------------

    latest = (
        df.sort_values(
            ["city_id", "date"]
        )
        .groupby(
            "city_id",
            as_index=False
        )
        .tail(1)
        .copy()
    )

    if latest.empty:

        raise ValueError(
            "No valid prediction rows found."
        )

    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in features
        if feature not in latest.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing production features: "
            + str(missing_features)
        )

    X = latest[features].copy()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print()
    print("Generating predictions...")

    predictions = model.predict(X)

    latest["prediction"] = predictions

    # AQI cannot be negative
    latest["prediction"] = (
        latest["prediction"]
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # Prediction date
    #
    # Current row represents today's known AQI.
    # Target is next-day AQI.
    # --------------------------------------------------------

    latest["prediction_date"] = (
        pd.to_datetime(latest["date"])
        + pd.Timedelta(days=1)
    )

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    latest["aqi_category"] = (
        latest["prediction"]
        .apply(get_aqi_category)
    )

    latest["health_message"] = (
        latest["prediction"]
        .apply(get_health_message)
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    result = latest[
        [
            "city_id",
            "city_name",
            "date",
            "prediction_date",
            "aqi",
            "prediction",
            "aqi_category",
            "health_message",
        ]
    ].copy()

    result = result.sort_values(
        "city_name"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = (
        "models/final_production_xgboost/"
        "predictions"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = (
        output_dir
        + "/latest_predictions.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NEXT-DAY AQI PREDICTIONS")
    print("=" * 70)

    print(
        result.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to: {output_file}"
    )

    print()
    print(
        f"Cities predicted: {len(result)}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    predict()