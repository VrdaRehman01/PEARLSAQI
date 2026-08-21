import os
import pandas as pd

from src.config_locations import CITY_COORDINATES
from src.ingestion.weather_history import fetch_weather_history
from src.ingestion.load_historical_data import load_historical_data
from src.preprocessing.merge_historical_data import create_daily_aqi


OUTPUT_FILE = "data/historical/historical_weather.csv"


def download_weather_history():

    print("\n========== DOWNLOADING HISTORICAL WEATHER ==========\n")

    historical_df = load_historical_data()
    daily_df = create_daily_aqi(historical_df)

    start_date = daily_df["date"].min().strftime("%Y-%m-%d")
    end_date = daily_df["date"].max().strftime("%Y-%m-%d")

    print(f"Date Range : {start_date} -> {end_date}")

    all_weather = []

    for city in CITY_COORDINATES.keys():

        print(f"Downloading {city}...")

        weather_df = fetch_weather_history(
            city=city,
            start_date=start_date,
            end_date=end_date
        )

        if weather_df is not None:
            all_weather.append(weather_df)

    if not all_weather:
        raise ValueError("No weather data downloaded.")

    final_weather = pd.concat(
        all_weather,
        ignore_index=True
    )

    os.makedirs("data/historical", exist_ok=True)

    final_weather.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nHistorical weather saved successfully!")

    print(final_weather.head())

    print(f"\nRows : {len(final_weather)}")

    print(f"Saved to : {OUTPUT_FILE}")


if __name__ == "__main__":
    print("Running as main")
    download_weather_history()
