from src.pipelines.ingestion_pipeline import run_ingestion_pipeline
from src.pipelines.feature_pipeline import run_feature_pipeline

import os

RAW_FILE = "data/raw/aqi_weather.csv"


def main():

    print("\n")
    print("=" * 60)
    print("AQI FORECASTING ML PIPELINE")
    print("=" * 60)

    run_ingestion_pipeline()

    if not os.path.exists(RAW_FILE):
        print("\n" + "!" * 60)
        print("STOPPING: ingestion collected 0 records this run.")
        print("Check the 'No AQI data for <city>' messages above --")
        print("they now show the actual reason from AQICN's API.")
        print("!" * 60)
        return

    run_feature_pipeline()

    print("\n")
    print("=" * 60)
    print("PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
