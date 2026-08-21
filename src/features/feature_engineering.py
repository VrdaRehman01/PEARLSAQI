import os
import pandas as pd

RAW_FILE = "data/raw/aqi_weather.csv"
OUTPUT_FILE = "data/features/live_features.csv"

# Keep this aligned with FEATURE_COLUMNS in src/models/base_model.py --
# the live pipeline and training pipeline must produce the same schema
# or the trained model can't score live predictions.
FEATURE_COLUMNS = [
    "timestamp", "city", "hour", "day", "month", "dayofweek",
    "aqi", "pm25", "pm10", "no2", "so2", "o3", "co",
    "temperature", "humidity", "pressure", "wind_speed", "weather_code", "precipitation",
    "aqi_change_rate",
]


def _add_time_features(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    return df


def _add_change_rate(df):
    df = df.sort_values(["city", "timestamp"])
    hours_between = df.groupby("city")["timestamp"].diff().dt.total_seconds() / 3600
    df["aqi_change_rate"] = (df.groupby("city")["aqi"].diff() / hours_between).fillna(0)
    return df


def engineer_features():
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"{RAW_FILE} not found. Run the ingestion pipeline first.")

    df = pd.read_csv(RAW_FILE)
    df = _add_time_features(df)
    df = _add_change_rate(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[FEATURE_COLUMNS]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Engineered {len(df)} rows -> {OUTPUT_FILE}")
    return df
