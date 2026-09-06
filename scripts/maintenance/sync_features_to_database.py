import pandas as pd

from src.database.connection import get_connection


PARQUET_FILE = "data/processed/features_v4.parquet"


def sync_features():

    print("=" * 70)
    print("SYNCING V4 FEATURES INTO DUCKDB")
    print("=" * 70)

    print()
    print("Loading:", PARQUET_FILE)

    df = pd.read_parquet(PARQUET_FILE)

    print()
    print("Parquet rows :", len(df))
    print("Cities       :", df["city_id"].nunique())
    print("First date   :", df["date"].min())
    print("Last date    :", df["date"].max())

    conn = get_connection()

    try:

        print()
        print("Replacing DuckDB features table...")

        conn.register(
            "features_dataframe",
            df
        )

        conn.execute(
            "DROP TABLE IF EXISTS features"
        )

        conn.execute(
            """
            CREATE TABLE features AS
            SELECT *
            FROM features_dataframe
            """
        )

        conn.unregister(
            "features_dataframe"
        )

        print("Features table replaced successfully.")

        print()
        print("=== VERIFYING FEATURES TABLE ===")

        result = conn.execute(
            """
            SELECT
                MIN(CAST(date AS DATE)) AS first_date,
                MAX(CAST(date AS DATE)) AS last_date,
                COUNT(*) AS rows,
                COUNT(DISTINCT city_id) AS cities
            FROM features
            """
        ).fetchdf()

        print(result.to_string(index=False))

        print()
        print("=== LATEST FEATURE DATES ===")

        latest = conn.execute(
            """
            SELECT
                CAST(date AS DATE) AS date,
                COUNT(*) AS rows,
                COUNT(DISTINCT city_id) AS cities
            FROM features
            GROUP BY CAST(date AS DATE)
            ORDER BY date DESC
            LIMIT 5
            """
        ).fetchdf()

        print(latest.to_string(index=False))

    finally:

        conn.close()

    print()
    print("=" * 70)
    print("FEATURE SYNC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    sync_features()
