"""
PEARLSAQI DAILY PRODUCTION PIPELINE

Daily execution order:

1. Update AQI
2. Update historical weather
3. Update future weather forecasts
4. Rebuild V4 features
5. Create V4 time-series train/validation/test splits
6. Automatically retrain and evaluate candidate model
7. Promote candidate only when it passes the promotion gate
8. Generate production V9 forecasts
9. Evaluate forecasts that now have actual AQI

Important
---------
Future weather MUST be downloaded before V9 forecasting.

V4 features MUST be rebuilt before automatic retraining.

Dataset splits MUST be rebuilt before automatic retraining.

V9 always loads the currently registered production model.

The production model is NEVER overwritten directly.

If any pipeline stage fails, the daily pipeline stops immediately.
"""


# ============================================================
# INGESTION / SERVICES
# ============================================================

from src.ingestion.aqi_downloader import (
    AQIDownloader
)

from src.services.weather_service import (
    WeatherService
)

from src.services.weather_forecast_service import (
    WeatherForecastService
)


# ============================================================
# FEATURE / ML PIPELINES
# ============================================================

from src.features.feature_builder_v4 import (
    build_features
)

from src.ml.dataset_splitter_v4 import (
    create_splits
)

from src.ml.auto_retrain import (
    run_auto_retrain
)


# ============================================================
# FORECAST / EVALUATION
# ============================================================

from src.ml.forecast_engine_v9 import (
    main as run_forecast
)

from src.ml.forecast.forecast_evaluation import (
    run_evaluation
)


# ============================================================
# HELPER
# ============================================================

def print_step(
    step_number,
    title
):

    print()
    print("=" * 70)
    print(
        f"STEP {step_number} : {title}"
    )
    print("=" * 70)


# ============================================================
# MAIN DAILY PIPELINE
# ============================================================

def run_daily_pipeline():

    print()
    print("=" * 70)
    print("PEARLSAQI DAILY PRODUCTION PIPELINE")
    print("=" * 70)

    print()
    print(
        "Execution mode : AUTOMATED"
    )

    print(
        "Production model protection : ENABLED"
    )

    print(
        "Fail-fast mode : ENABLED"
    )

    # ========================================================
    # STEP 1
    # AQI UPDATE
    # ========================================================

    print_step(
        1,
        "AQI UPDATE"
    )

    aqi_service = AQIDownloader()

    aqi_service.download_incremental()

    print()
    print(
        "AQI update completed."
    )

    # ========================================================
    # STEP 2
    # HISTORICAL WEATHER UPDATE
    # ========================================================

    print_step(
        2,
        "HISTORICAL WEATHER UPDATE"
    )

    weather_service = WeatherService()

    weather_service.download_incremental()

    print()
    print(
        "Historical weather update completed."
    )

    # ========================================================
    # STEP 3
    # FUTURE WEATHER FORECAST
    # ========================================================

    print_step(
        3,
        "FUTURE WEATHER FORECAST"
    )

    future_weather_service = (
        WeatherForecastService()
    )

    future_weather_service.download_forecast()

    print()
    print(
        "Future weather forecast completed."
    )

    # ========================================================
    # STEP 4
    # V4 FEATURE ENGINEERING
    # ========================================================

    print_step(
        4,
        "V4 FEATURE ENGINEERING"
    )

    print()
    print(
        "Rebuilding V4 features from updated "
        "AQI and weather data..."
    )

    build_features()

    print()
    print(
        "V4 feature engineering completed."
    )

    # ========================================================
    # STEP 5
    # V4 DATASET SPLITTING
    # ========================================================

    print_step(
        5,
        "V4 DATASET SPLITTING"
    )

    print()
    print(
        "Rebuilding time-series train/"
        "validation/test datasets..."
    )

    create_splits()

    print()
    print(
        "V4 dataset splitting completed."
    )

    # ========================================================
    # STEP 6
    # AUTOMATIC MODEL RETRAINING
    # ========================================================

    print_step(
        6,
        "AUTOMATIC MODEL RETRAINING"
    )

    print()
    print(
        "Running automatic model evaluation..."
    )

    print(
        "Candidate models will only be promoted "
        "if they pass the production gate."
    )

    retrain_result = run_auto_retrain()

    print()
    print("=" * 70)
    print("AUTOMATIC RETRAINING RESULT")
    print("=" * 70)

    if retrain_result is None:

        print(
            "Result : COMPLETED"
        )

    else:

        promoted = retrain_result.get(
            "promoted",
            False
        )

        if promoted:

            print(
                "Result : MODEL PROMOTED"
            )

            if (
                "production_version"
                in retrain_result
            ):

                print(
                    "Previous production version:",
                    retrain_result[
                        "production_version"
                    ]
                )

        else:

            print(
                "Result : CANDIDATE REJECTED"
            )

            if (
                "production_version"
                in retrain_result
            ):

                print(
                    "Active production version:",
                    retrain_result[
                        "production_version"
                    ]
                )

    print()
    print(
        "Automatic retraining stage completed."
    )

    # ========================================================
    # STEP 7
    # V9 PRODUCTION FORECAST
    # ========================================================

    print_step(
        7,
        "V9 PRODUCTION FORECAST"
    )

    print()
    print(
        "Generating forecasts using the "
        "currently registered production model..."
    )

    run_forecast()

    print()
    print(
        "V9 forecast generation completed."
    )

    # ========================================================
    # STEP 8
    # FORECAST EVALUATION
    # ========================================================

    print_step(
        8,
        "FORECAST EVALUATION"
    )

    print()
    print(
        "Evaluating forecasts that now have "
        "actual AQI values..."
    )

    run_evaluation()

    print()
    print(
        "Forecast evaluation completed."
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("PEARLSAQI DAILY PIPELINE COMPLETE")
    print("=" * 70)

    print()
    print(
        "All automated stages completed successfully."
    )

    print(
        "AQI + Weather + Features + Splits + "
        "Retraining + Forecast + Evaluation"
    )

    print()
    print(
        "Production model protection : ENABLED"
    )

    print(
        "Forecast duplicate protection : ENABLED"
    )

    print(
        "Pipeline status : SUCCESS"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_daily_pipeline()