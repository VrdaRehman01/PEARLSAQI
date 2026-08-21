import pandas as pd

# This was referenced by load_historical_data.py but never existed in
# your uploaded files. Implemented here defensively since I can't see
# the actual columns in mk12rule/pakistan_air_quality_dataset from here --
# the first time you run this, READ THE PRINTED COLUMN LIST and confirm
# the date column was detected correctly.

DATE_COLUMN_CANDIDATES = ["date", "Date", "timestamp", "Timestamp", "datetime"]


def validate_historical_data(df):
    print("\nValidating historical dataset...")
    print("Columns found:", list(df.columns))

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    date_col = next((c for c in DATE_COLUMN_CANDIDATES if c in df.columns), None)

    if date_col is None:
        raise ValueError(
            "No recognizable date column found. Columns available: "
            f"{list(df.columns)}. Add the correct column name to "
            "DATE_COLUMN_CANDIDATES in validate_historical_data.py."
        )

    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["date"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with unparseable dates.")

    if "city" not in df.columns:
        raise ValueError(
            "Expected a 'city' column (added in load_historical_data.py) "
            "but it's missing -- check that step ran before validation."
        )

    print(f"Validated {len(df)} rows across {df['city'].nunique()} cities.")
    return df
