import pandas as pd


class DataValidator:

    REQUIRED_COLUMNS = [
        "city_id",
        "city_name",
        "date",

        "aqi",
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3",

        "temperature",
        "humidity",
        "precipitation",
        "windspeed",

        "aqi_lag_1",
        "aqi_lag_2",
        "aqi_lag_3",
        "aqi_lag_7",

        "aqi_rolling_3",
        "aqi_rolling_7",

        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "is_weekend",
        "season",

        "target_aqi"
    ]

    # ---------------------------------------------------------
    # Run all validation checks
    # ---------------------------------------------------------

    def validate(self, df):

        print("\n" + "=" * 60)
        print("DATA VALIDATION")
        print("=" * 60)

        checks = {}

        checks["columns"] = self.check_columns(df)

        checks["missing_values"] = (
            self.check_missing_values(df)
        )

        checks["duplicates"] = (
            self.check_duplicates(df)
        )

        checks["dates"] = (
            self.check_dates(df)
        )

        checks["cities"] = (
            self.check_cities(df)
        )

        checks["numeric_values"] = (
            self.check_numeric_values(df)
        )

        checks["target"] = (
            self.check_target(df)
        )

        print("\n" + "=" * 60)

        if all(checks.values()):

            print("DATA VALIDATION PASSED")

            return True

        print("DATA VALIDATION FAILED")

        failed = [
            name
            for name, result in checks.items()
            if not result
        ]

        print("\nFailed checks:")

        for name in failed:

            print(f"    {name}")

        return False

    # ---------------------------------------------------------
    # Columns
    # ---------------------------------------------------------

    def check_columns(self, df):

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:

            print(
                "\nMissing columns:"
            )

            for column in missing:

                print(
                    f"   - {column}"
                )

            return False

        print(
            "Required columns present"
        )

        return True

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    def check_missing_values(self, df):

        missing = (
            df[self.REQUIRED_COLUMNS]
            .isnull()
            .sum()
        )

        total_missing = (
            missing.sum()
        )

        if total_missing > 0:

            print(
                f"\n❌ Missing values: "
                f"{total_missing}"
            )

            print(
                missing[
                    missing > 0
                ]
            )

            return False

        print(
            "No missing values"
        )

        return True

    # ---------------------------------------------------------
    # Duplicates
    # ---------------------------------------------------------

    def check_duplicates(self, df):

        duplicates = df.duplicated(
            subset=[
                "city_id",
                "date"
            ]
        ).sum()

        if duplicates > 0:

            print(
                f"Duplicate city/date "
                f"records: {duplicates}"
            )

            return False

        print(
            "No duplicate city/date records"
        )

        return True

    # ---------------------------------------------------------
    # Dates
    # ---------------------------------------------------------

    def check_dates(self, df):

        dates = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        invalid = dates.isna().sum()

        if invalid > 0:

            print(
                f"Invalid dates: {invalid}"
            )

            return False

        if not dates.is_monotonic_increasing:

            # This is not necessarily an error because
            # the dataset contains multiple cities.
            # We therefore check per city.

            for city_id, group in df.groupby(
                "city_id"
            ):

                city_dates = pd.to_datetime(
                    group["date"]
                )

                if not city_dates.is_monotonic_increasing:

                    print(
                        f"Dates not ordered "
                        f"for city {city_id}"
                    )

                    return False

        print(
            "Date validation passed"
        )

        return True

    # ---------------------------------------------------------
    # Cities
    # ---------------------------------------------------------

    def check_cities(self, df):

        city_count = (
            df["city_id"]
            .nunique()
        )

        if city_count != 12:

            print(
                f"Expected 12 cities, "
                f"found {city_count}"
            )

            return False

        print(
            f"{city_count} cities detected"
        )

        return True

    # ---------------------------------------------------------
    # Numeric values
    # ---------------------------------------------------------

    def check_numeric_values(self, df):

        numeric_columns = [
            "aqi",
            "pm25",
            "pm10",
            "no2",
            "so2",
            "co",
            "o3",
            "temperature",
            "humidity",
            "precipitation",
            "windspeed",
            "target_aqi"
        ]

        for column in numeric_columns:

            if not pd.api.types.is_numeric_dtype(
                df[column]
            ):

                print(
                    f" {column} "
                    f"is not numeric"
                )

                return False

        print(
            "Numeric feature validation passed"
        )

        return True

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------

    def check_target(self, df):

        if (
            df["target_aqi"] < 0
        ).any():

            print(
                "Negative target AQI detected"
            )

            return False

        if df["target_aqi"].isna().any():

            print(
                "Missing target values"
            )

            return False

        print(
            "Target validation passed"
        )

        return True