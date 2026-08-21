import os
from pathlib import Path

import duckdb
import pandas as pd


# ============================================================
# LOCAL FEATURE STORE
# ============================================================
#
# Hopsworks replacement:
#
#   Historical features -> Parquet + DuckDB
#   Live features       -> Parquet/CSV + DuckDB
#   Feature retrieval   -> pandas DataFrames
#
# Public functions intentionally keep the same names as the
# previous Hopsworks implementation so the rest of the project
# can continue using the same interface.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_FILE = PROJECT_ROOT / "database" / "aqi.duckdb"

FEATURE_STORE_DIR = PROJECT_ROOT / "data" / "feature_store"
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

HISTORICAL_FEATURES_FILE = (
    FEATURE_STORE_DIR / "historical_features.parquet"
)

LIVE_FEATURES_FILE = (
    PROJECT_ROOT / "data" / "features" / "live_features.csv"
)

LIVE_FEATURES_PARQUET = (
    FEATURE_STORE_DIR / "live_features.parquet"
)

HISTORICAL_TABLE = "feature_store_historical"
LIVE_TABLE = "feature_store_live"


# ============================================================
# CONNECTION
# ============================================================

def _connect():
    """
    Open the project's DuckDB database.

    The database is the source of truth for the local feature
    store whenever the corresponding feature-store table exists.
    """
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)

    return duckdb.connect(str(DATABASE_FILE))


# ============================================================
# UTILITY
# ============================================================

def _normalise_dates(df):
    """
    Convert common date/timestamp columns into consistent pandas
    datetime values without assuming a particular schema.
    """
    df = df.copy()

    for column in ("date", "timestamp", "forecast_origin"):
        if column in df.columns:
            try:
                df[column] = pd.to_datetime(df[column])
            except Exception:
                pass

    return df


def _write_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ============================================================
# FEATURE STORE INITIALISATION
# ============================================================

def initialise_feature_store():
    """
    Create local Feature Store tables if they do not already exist.
    """
    connection = _connect()

    try:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {HISTORICAL_TABLE} AS
            SELECT *
            FROM (
                SELECT
                    CAST(NULL AS VARCHAR) AS _feature_store_placeholder
            )
            WHERE FALSE
            """
        )

        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {LIVE_TABLE} AS
            SELECT *
            FROM (
                SELECT
                    CAST(NULL AS VARCHAR) AS _feature_store_placeholder
            )
            WHERE FALSE
            """
        )

    finally:
        connection.close()

    print("LOCAL FEATURE STORE INITIALISED")
    print(f"Database : {DATABASE_FILE}")


# ============================================================
# LIVE FEATURE STORE
# ============================================================

def build_feature_store(df=None):
    """
    Persist production/live V4 features locally.

    If a dataframe is supplied, it is used directly.
    Otherwise the legacy live_features.csv source is used.

    Storage:
        data/feature_store/live_features.parquet
        database/aqi.duckdb -> feature_store_live

    Update strategy:
        - preserve existing historical/live rows
        - upsert incoming city/date records
        - remove duplicate city/date keys
    """

    # --------------------------------------------------------
    # Load incoming features
    # --------------------------------------------------------

    if df is None:

        if not LIVE_FEATURES_FILE.exists():

            raise FileNotFoundError(
                f"{LIVE_FEATURES_FILE} not found. "
                "Run the feature engineering pipeline first."
            )

        new_df = pd.read_csv(
            LIVE_FEATURES_FILE
        )

    else:

        new_df = df.copy()

    if new_df.empty:

        raise ValueError(
            "Live feature dataframe contains 0 rows."
        )

    new_df = _normalise_dates(
        new_df
    )

    # --------------------------------------------------------
    # Validate required keys
    # --------------------------------------------------------

    required_columns = [
        "city_id",
        "date"
    ]

    missing = [
        column
        for column in required_columns
        if column not in new_df.columns
    ]

    if missing:

        raise ValueError(
            f"Live features missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Read existing live Feature Store
    # --------------------------------------------------------

    connection = _connect()

    try:

        tables = connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [LIVE_TABLE]
        ).fetchall()

        if tables:

            existing_df = connection.execute(
                f"SELECT * FROM {LIVE_TABLE}"
            ).fetchdf()

        else:

            existing_df = pd.DataFrame()

    finally:

        connection.close()

    # --------------------------------------------------------
    # Combine existing + incoming
    # --------------------------------------------------------

    if not existing_df.empty:

        existing_df = _normalise_dates(
            existing_df
        )

        combined = pd.concat(
            [
                existing_df,
                new_df
            ],
            ignore_index=True,
            sort=False
        )

    else:

        combined = new_df.copy()

    # --------------------------------------------------------
    # Remove duplicate city/date records
    #
    # Incoming data wins.
    # --------------------------------------------------------

    combined = (
        combined
        .drop_duplicates(
            subset=["city_id", "date"],
            keep="last"
        )
        .sort_values(
            ["city_id", "date"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Persist Parquet
    # --------------------------------------------------------

    _write_parquet(
        combined,
        LIVE_FEATURES_PARQUET
    )

    # --------------------------------------------------------
    # Persist DuckDB
    # --------------------------------------------------------

    connection = _connect()

    try:

        connection.register(
            "combined_live_features",
            combined
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE {LIVE_TABLE} AS
            SELECT *
            FROM combined_live_features
            """
        )

        connection.unregister(
            "combined_live_features"
        )

    finally:

        connection.close()

    print()
    print(
        f"Local Feature Store: processed "
        f"{len(new_df)} incoming live row(s)."
    )

    print(
        f"Live Feature Store total: "
        f"{len(combined)} row(s)."
    )

    print(
        f"Parquet: {LIVE_FEATURES_PARQUET}"
    )

    print(
        f"DuckDB table: {LIVE_TABLE}"
    )

    return combined

def _read_live_features():

    connection = _connect()

    try:

        tables = connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [LIVE_TABLE]
        ).fetchall()

        if tables:

            df = connection.execute(
                f"SELECT * FROM {LIVE_TABLE}"
            ).fetchdf()

            if not df.empty:
                return _normalise_dates(df)

    finally:
        connection.close()

    # --------------------------------------------------------
    # Fallback to Parquet
    # --------------------------------------------------------

    if LIVE_FEATURES_PARQUET.exists():

        return _normalise_dates(
            pd.read_parquet(LIVE_FEATURES_PARQUET)
        )

    # --------------------------------------------------------
    # Fallback to CSV
    # --------------------------------------------------------

    if LIVE_FEATURES_FILE.exists():

        return _normalise_dates(
            pd.read_csv(LIVE_FEATURES_FILE)
        )

    return pd.DataFrame()


def read_latest_features(city):
    """
    Return the latest live feature row for a city.
    """

    df = _read_live_features()

    if df.empty:
        raise ValueError(
            "Local live Feature Store contains no data."
        )

    city_column = None

    for candidate in ("city", "city_name"):
        if candidate in df.columns:
            city_column = candidate
            break

    if city_column is None:
        raise KeyError(
            "Live Feature Store has neither 'city' nor "
            "'city_name' column."
        )

    city_df = df[
        df[city_column].astype(str).str.lower()
        == str(city).lower()
    ].copy()

    if city_df.empty:
        raise ValueError(
            f"No live features found for {city}."
        )

    if "timestamp" in city_df.columns:

        city_df = city_df.sort_values(
            "timestamp"
        )

    elif "date" in city_df.columns:

        city_df = city_df.sort_values(
            "date"
        )

    return city_df.iloc[-1]


def read_recent_daily_aqi(city, days=10):
    """
    Read recent live AQI history and aggregate it to daily values.

    Returns:
        pandas Series indexed by date, oldest first.
    """

    df = _read_live_features()

    if df.empty:
        return pd.Series(dtype=float)

    city_column = None

    for candidate in ("city", "city_name"):
        if candidate in df.columns:
            city_column = candidate
            break

    if city_column is None or "aqi" not in df.columns:
        return pd.Series(dtype=float)

    city_df = df[
        df[city_column].astype(str).str.lower()
        == str(city).lower()
    ].copy()

    if city_df.empty:
        return pd.Series(dtype=float)

    if "timestamp" in city_df.columns:

        city_df["timestamp"] = pd.to_datetime(
            city_df["timestamp"]
        )

        city_df["date"] = city_df[
            "timestamp"
        ].dt.date

    elif "date" in city_df.columns:

        city_df["date"] = pd.to_datetime(
            city_df["date"]
        ).dt.date

    else:
        return pd.Series(dtype=float)

    daily = (
        city_df
        .groupby("date")["aqi"]
        .mean()
        .sort_index()
    )

    return daily.tail(days)


# ============================================================
# HISTORICAL FEATURE STORE
# ============================================================

def build_historical_feature_store(df):
    """
    Incrementally persist historical V4 features locally.

    The local Feature Store uses DuckDB as the source of truth and
    Parquet as a portable snapshot.

    Records are uniquely identified by:
        (city_id, date)

    Existing historical records are preserved.
    New dates are added.
    Existing city/date records are replaced by the newest incoming
    version.

    This provides Hopsworks-like historical Feature Store behavior
    without requiring an external service.
    """

    if df is None:
        raise ValueError(
            "Historical feature dataframe is None."
        )

    df = df.copy()

    if df.empty:
        raise ValueError(
            "Cannot update historical Feature Store from 0 rows."
        )

    df = _normalise_dates(df)

    required_keys = ["city_id", "date"]

    missing_keys = [
        column
        for column in required_keys
        if column not in df.columns
    ]

    if missing_keys:
        raise ValueError(
            f"Historical Feature Store requires columns: "
            f"{missing_keys}"
        )

    # --------------------------------------------------------
    # Remove duplicate keys inside incoming data
    # --------------------------------------------------------

    df = (
        df
        .sort_values(["city_id", "date"])
        .drop_duplicates(
            subset=["city_id", "date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    connection = _connect()

    try:

        # ----------------------------------------------------
        # Determine whether an existing historical table exists
        # ----------------------------------------------------

        tables = connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [HISTORICAL_TABLE]
        ).fetchall()

        existing = pd.DataFrame()

        if tables:

            existing = connection.execute(
                f"""
                SELECT *
                FROM {HISTORICAL_TABLE}
                """
            ).fetchdf()

            existing = _normalise_dates(existing)

        # ----------------------------------------------------
        # First write
        # ----------------------------------------------------

        if existing.empty:

            combined = df.copy()

        else:

            # ------------------------------------------------
            # Schema compatibility
            # ------------------------------------------------

            all_columns = list(
                dict.fromkeys(
                    list(existing.columns)
                    + list(df.columns)
                )
            )

            for column in all_columns:

                if column not in existing.columns:
                    existing[column] = pd.NA

                if column not in df.columns:
                    df[column] = pd.NA

            existing = existing[all_columns]
            df = df[all_columns]

            # ------------------------------------------------
            # Incremental merge
            # ------------------------------------------------

            combined = pd.concat(
                [existing, df],
                ignore_index=True
            )

            combined = (
                combined
                .sort_values(
                    ["city_id", "date"]
                )
                .drop_duplicates(
                    subset=["city_id", "date"],
                    keep="last"
                )
                .reset_index(drop=True)
            )

        # ----------------------------------------------------
        # Save DuckDB
        # ----------------------------------------------------

        connection.register(
            "combined_historical_features",
            combined
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE {HISTORICAL_TABLE} AS
            SELECT *
            FROM combined_historical_features
            """
        )

        connection.unregister(
            "combined_historical_features"
        )

    finally:

        connection.close()

    # --------------------------------------------------------
    # Save Parquet snapshot
    # --------------------------------------------------------

    _write_parquet(
        combined,
        HISTORICAL_FEATURES_FILE
    )

    print(
        f"Local Feature Store: "
        f"processed {len(df)} incoming row(s)."
    )

    print(
        f"Historical Feature Store total: "
        f"{len(combined)} row(s)."
    )

    print(
        f"Parquet: {HISTORICAL_FEATURES_FILE}"
    )

    print(
        f"DuckDB table: {HISTORICAL_TABLE}"
    )

    return combined

def read_historical_features():
    """
    Read historical training features from the local Feature Store.

    Priority:
        1. DuckDB feature-store table
        2. historical_features.parquet
        3. existing V4 feature parquet
    """

    connection = _connect()

    try:

        tables = connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [HISTORICAL_TABLE]
        ).fetchall()

        if tables:

            df = connection.execute(
                f"SELECT * FROM {HISTORICAL_TABLE}"
            ).fetchdf()

            if not df.empty:
                return _normalise_dates(df)

    finally:
        connection.close()

    # --------------------------------------------------------
    # Local Feature Store Parquet
    # --------------------------------------------------------

    if HISTORICAL_FEATURES_FILE.exists():

        return _normalise_dates(
            pd.read_parquet(
                HISTORICAL_FEATURES_FILE
            )
        )

    # --------------------------------------------------------
    # Existing V4 feature dataset
    # --------------------------------------------------------

    v4_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "features_v4.parquet"
    )

    if v4_file.exists():

        print(
            "Historical Feature Store not yet populated."
        )

        print(
            "Loading existing V4 feature dataset "
            "as bootstrap data."
        )

        return _normalise_dates(
            pd.read_parquet(v4_file)
        )

    raise FileNotFoundError(
        "No historical features found. "
        "Run the historical feature/backfill pipeline first."
    )


# ============================================================
# FEATURE STORE INSPECTION
# ============================================================

def get_feature_store_stats():

    stats = {}

    # Historical
    try:

        historical = read_historical_features()

        stats["historical_rows"] = len(
            historical
        )

        stats["historical_columns"] = len(
            historical.columns
        )

    except Exception:

        stats["historical_rows"] = 0
        stats["historical_columns"] = 0

    # Live
    try:

        live = _read_live_features()

        stats["live_rows"] = len(live)
        stats["live_columns"] = len(live.columns)

    except Exception:

        stats["live_rows"] = 0
        stats["live_columns"] = 0

    return stats


# ============================================================
# COMMAND LINE CHECK
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PEARLSAQI LOCAL FEATURE STORE")
    print("=" * 70)

    initialise_feature_store()

    stats = get_feature_store_stats()

    print()
    print("Feature Store Status")
    print("-" * 40)
    print(
        f"Historical rows   : {stats['historical_rows']}"
    )
    print(
        f"Historical cols   : {stats['historical_columns']}"
    )
    print(
        f"Live rows         : {stats['live_rows']}"
    )
    print(
        f"Live cols         : {stats['live_columns']}"
    )

    print()
    print("LOCAL FEATURE STORE READY")
