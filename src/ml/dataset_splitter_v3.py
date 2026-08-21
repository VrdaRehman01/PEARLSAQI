import os
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE = "data/processed/features_v3.parquet"

OUTPUT_DIR = "data/processed/v3"


# ==========================================================
# Split configuration
# ==========================================================

TRAIN_YEARS = [2023, 2024]
VALIDATION_YEARS = [2025]
TEST_YEARS = [2026]


# ==========================================================
# Main
# ==========================================================

def create_splits():

    print("=" * 60)
    print("V3 TIME-SERIES DATA SPLIT")
    print("=" * 60)

    # ------------------------------------------------------
    # Load
    # ------------------------------------------------------

    df = pd.read_parquet(
        INPUT_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        ["date", "city_id"]
    ).reset_index(drop=True)

    print()
    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    # ------------------------------------------------------
    # Create year column if needed
    # ------------------------------------------------------

    if "year" not in df.columns:

        df["year"] = (
            df["date"].dt.year
        )

    # ------------------------------------------------------
    # Split
    # ------------------------------------------------------

    train_df = df[
        df["year"].isin(
            TRAIN_YEARS
        )
    ].copy()

    validation_df = df[
        df["year"].isin(
            VALIDATION_YEARS
        )
    ].copy()

    test_df = df[
        df["year"].isin(
            TEST_YEARS
        )
    ].copy()

    # ------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------

    if len(train_df) == 0:
        raise ValueError(
            "Training dataset is empty."
        )

    if len(validation_df) == 0:
        raise ValueError(
            "Validation dataset is empty."
        )

    if len(test_df) == 0:
        raise ValueError(
            "Test dataset is empty."
        )

    # ------------------------------------------------------
    # Output directory
    # ------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    train_file = os.path.join(
        OUTPUT_DIR,
        "train.parquet"
    )

    validation_file = os.path.join(
        OUTPUT_DIR,
        "validation.parquet"
    )

    test_file = os.path.join(
        OUTPUT_DIR,
        "test.parquet"
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    train_df.to_parquet(
        train_file,
        index=False
    )

    validation_df.to_parquet(
        validation_file,
        index=False
    )

    test_df.to_parquet(
        test_file,
        index=False
    )

    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAIN")
    print("=" * 60)

    print(
        f"Rows      : {len(train_df)}"
    )

    print(
        f"Start date: "
        f"{train_df['date'].min()}"
    )

    print(
        f"End date  : "
        f"{train_df['date'].max()}"
    )

    print()
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)

    print(
        f"Rows      : {len(validation_df)}"
    )

    print(
        f"Start date: "
        f"{validation_df['date'].min()}"
    )

    print(
        f"End date  : "
        f"{validation_df['date'].max()}"
    )

    print()
    print("=" * 60)
    print("TEST")
    print("=" * 60)

    print(
        f"Rows      : {len(test_df)}"
    )

    print(
        f"Start date: "
        f"{test_df['date'].min()}"
    )

    print(
        f"End date  : "
        f"{test_df['date'].max()}"
    )

    # ------------------------------------------------------
    # Check overlap
    # ------------------------------------------------------

    train_dates = set(
        train_df["date"]
    )

    validation_dates = set(
        validation_df["date"]
    )

    test_dates = set(
        test_df["date"]
    )

    print()
    print("=" * 60)
    print("OVERLAP CHECK")
    print("=" * 60)

    print(
        "Train ∩ Validation:",
        len(
            train_dates &
            validation_dates
        )
    )

    print(
        "Train ∩ Test:",
        len(
            train_dates &
            test_dates
        )
    )

    print(
        "Validation ∩ Test:",
        len(
            validation_dates &
            test_dates
        )
    )

    # ------------------------------------------------------
    # City balance
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("CITY DISTRIBUTION")
    print("=" * 60)

    print()
    print("Training:")

    print(
        train_df[
            "city_name"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Validation:")

    print(
        validation_df[
            "city_name"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Test:")

    print(
        test_df[
            "city_name"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ------------------------------------------------------
    # Files
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(train_file)
    print(validation_file)
    print(test_file)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    create_splits()