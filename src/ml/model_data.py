import pandas as pd

from src.ml.dataset_splitter import DatasetSplitter


class ModelData:

    # ---------------------------------------------------------
    # Features used by ML models
    # ---------------------------------------------------------

    FEATURES = [

        # Current pollution
        "aqi",
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3",

        # Weather
        "temperature",
        "humidity",
        "precipitation",
        "windspeed",

        # Historical AQI
        "aqi_lag_1",
        "aqi_lag_2",
        "aqi_lag_3",
        "aqi_lag_7",

        "aqi_rolling_3",
        "aqi_rolling_7",

        # Calendar
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "is_weekend",

        # City
        "city_id"
    ]

    TARGET = "target_aqi"

    # ---------------------------------------------------------
    # Prepare datasets
    # ---------------------------------------------------------

    def prepare(self):

        splitter = DatasetSplitter()

        df = splitter.load_dataset()

        train, validation, test = splitter.split(
            df
        )

        X_train = train[
            self.FEATURES
        ].copy()

        y_train = train[
            self.TARGET
        ].copy()

        X_validation = validation[
            self.FEATURES
        ].copy()

        y_validation = validation[
            self.TARGET
        ].copy()

        X_test = test[
            self.FEATURES
        ].copy()

        y_test = test[
            self.TARGET
        ].copy()

        print("\n" + "=" * 60)
        print("MODEL DATA")
        print("=" * 60)

        print(
            f"X_train      : {X_train.shape}"
        )

        print(
            f"y_train      : {y_train.shape}"
        )

        print(
            f"X_validation : {X_validation.shape}"
        )

        print(
            f"y_validation : {y_validation.shape}"
        )

        print(
            f"X_test       : {X_test.shape}"
        )

        print(
            f"y_test       : {y_test.shape}"
        )

        return (
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test
        )


if __name__ == "__main__":

    data = ModelData()

    data.prepare()
    