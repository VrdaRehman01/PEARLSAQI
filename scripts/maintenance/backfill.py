import os
import pandas as pd

from src.ingestion.load_historical_data import load_historical_data
from src.preprocessing.merge_historical_data import create_daily_aqi, merge_aqi_and_weather
from src.features.feature_store import build_historical_feature_store
from src.features.lag_rolling_features import add_lag_features, add_rolling_features, add_calendar_features

WEATHER_FILE = "data/historical/historical_weather.csv"
OUTPUT_FILE = "data/features/historical_features.csv"


def add_time_features(df):
    df["hour"] = 0  # daily-resolution data has no hour granularity
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    return df


def add_change_rate(df):
    df = df.sort_values(["city", "date"])
    df["aqi_change_rate"] = df.groupby("city")["aqi"].diff().fillna(0)
    return df


def add_horizon_targets(df, horizons=(1, 2, 3)):
    """
    Creates aqi_h1, aqi_h2, aqi_h3 -- the AQI value 1/2/3 days ahead
    per city. This is what makes a "3-day forecast" possible: rather
    than one model predicting a single day out, three separate targets
    are trained against, one per horizon (see src/models/data_loader.py
    and train_models.py). The last `max(horizons)` rows per city will
    have NaN targets since there's no future data for them yet --
    that's expected and handled by dropna() at training time.
    """
    df = df.sort_values(["city", "date"])
    for h in horizons:
        df[f"aqi_h{h}"] = df.groupby("city")["aqi"].shift(-h)
    return df


def run_backfill():
    print("\n" + "=" * 60)
    print("BACKFILL: BUILDING HISTORICAL TRAINING DATASET")
    print("=" * 60)

    print("\nLoading historical pollution data...")
    historical_df = load_historical_data()

    print("\nCollapsing to daily AQI per city...")
    daily_aqi = create_daily_aqi(historical_df)

    if not os.path.exists(WEATHER_FILE):
        raise FileNotFoundError(
            f"{WEATHER_FILE} not found. Run download_weather_history.py first."
        )

    print("\nLoading historical weather data...")
    weather_df = pd.read_csv(WEATHER_FILE)

    print("\nMerging AQI + weather...")
    merged = merge_aqi_and_weather(daily_aqi, weather_df)

    print("\nAdding time features + AQI change rate...")
    merged = add_time_features(merged)
    merged = add_change_rate(merged)

    print("\nAdding lag, rolling, and calendar features...")
    merged = add_lag_features(merged)
    merged = add_rolling_features(merged)
    merged = add_calendar_features(merged)

    print("\nAdding 3-day forecast targets (aqi_h1, aqi_h2, aqi_h3)...")
    merged = add_horizon_targets(merged)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\nLocal cache written -> {OUTPUT_FILE}")

    print("\nPushing historical features to Hopsworks Feature Store...")
    build_historical_feature_store(merged)

    print(f"\nBackfill complete: {len(merged)} rows")
    return merged


if __name__ == "__main__":
    run_backfill()
