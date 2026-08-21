from pathlib import Path

import pandas as pd

from src.database.connection import get_connection
from src.database.repositories.city_repository import CityRepository
from src.database.repositories.aqi_repository import AQIRepository


# ---------------------------------------------------------
# Existing historical AQI dataset
# ---------------------------------------------------------

INPUT_FILE = Path("data/raw/aqi_weather.csv")


def import_historical_aqi():

    print("\n" + "=" * 60)
    print("IMPORTING HISTORICAL AQI DATA")
    print("=" * 60)

    # -----------------------------------------------------
    # Check file
    # -----------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nAQI dataset not found:\n{INPUT_FILE}\n\n"
            "Put your historical AQI CSV inside data/raw/"
        )

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df)}")

    print("\nColumns found:")
    print(df.columns.tolist())

    # -----------------------------------------------------
    # Standardize column names
    # -----------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = [
        "city",
        "aqi"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column '{column}' "
                f"was not found in the dataset."
            )

    # -----------------------------------------------------
    # Handle timestamp/date
    # -----------------------------------------------------

    if "timestamp" in df.columns:

        df["date"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        ).dt.date

    elif "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        ).dt.date

    else:

        raise ValueError(
            "Dataset must contain either "
            "'timestamp' or 'date'."
        )

    # -----------------------------------------------------
    # Remove invalid dates
    # -----------------------------------------------------

    df = df.dropna(
        subset=["date", "city", "aqi"]
    )

    # -----------------------------------------------------
    # Convert numeric columns
    # -----------------------------------------------------

    numeric_columns = [
        "aqi",
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        else:

            # Missing pollutant columns become NULL
            df[column] = None

    # -----------------------------------------------------
    # Load city mapping from DuckDB
    # -----------------------------------------------------

    city_repo = CityRepository()

    cities = city_repo.get_all()

    city_repo.close()

    print("\nCities in DuckDB:")
    print(cities[["city_id", "city_name"]])

    # -----------------------------------------------------
    # Create city mapping
    # -----------------------------------------------------

    city_mapping = {

        row["city_name"].strip().lower():
        int(row["city_id"])

        for _, row in cities.iterrows()

    }

    # -----------------------------------------------------
    # Map city → city_id
    # -----------------------------------------------------

    df["city_key"] = (
        df["city"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["city_id"] = df["city_key"].map(
        city_mapping
    )

    # -----------------------------------------------------
    # Check unknown cities
    # -----------------------------------------------------

    unknown_cities = sorted(
        df.loc[
            df["city_id"].isna(),
            "city"
        ]
        .dropna()
        .unique()
    )

    if unknown_cities:

        print("\nWARNING: Unknown cities found:")

        for city in unknown_cities:

            print(f"   - {city}")

        print(
            "\nThese rows will NOT be imported."
        )

    # -----------------------------------------------------
    # Remove unknown cities
    # -----------------------------------------------------

    df = df.dropna(
        subset=["city_id"]
    )

    df["city_id"] = df["city_id"].astype(int)

    # -----------------------------------------------------
    # Remove duplicate city/date records
    # -----------------------------------------------------

    df = (
        df.sort_values(
            ["city_id", "date"]
        )
        .drop_duplicates(
            subset=["city_id", "date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Prepare records
    # -----------------------------------------------------

    records = []

    for index, row in df.iterrows():

        records.append(
            (
                index + 1,
                row["city_id"],
                row["date"],
                row["aqi"],
                row["pm25"],
                row["pm10"],
                row["no2"],
                row["so2"],
                row["co"],
                row["o3"]
            )
        )

    # -----------------------------------------------------
    # Insert into DuckDB
    # -----------------------------------------------------

    repo = AQIRepository()

    print(
        f"\nInserting {len(records)} records into DuckDB..."
    )

    # We are importing the historical dataset from scratch,
    # so clear the current AQI table first.
    repo.clear()

    repo.insert_many(records)

    total = repo.count()

    repo.close()

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("AQI IMPORT COMPLETE")
    print("=" * 60)

    print(f"Records imported : {total}")

    print(
        f"Cities included  : "
        f"{df['city_id'].nunique()}"
    )

    print(
        f"Date range       : "
        f"{df['date'].min()} → {df['date'].max()}"
    )

    print("\nAQI records by city:")

    summary = (
        df.groupby("city_id")
        .size()
        .reset_index(name="records")
    )

    summary = summary.merge(
        cities[
            ["city_id", "city_name"]
        ],
        on="city_id",
        how="left"
    )

    print(
        summary[
            ["city_id", "city_name", "records"]
        ]
        .sort_values("city_id")
        .to_string(index=False)
    )

    print("\nHistorical AQI successfully stored in DuckDB.")


if __name__ == "__main__":

    import_historical_aqi()