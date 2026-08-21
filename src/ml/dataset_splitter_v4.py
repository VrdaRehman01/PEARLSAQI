import os
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE = "data/processed/features_v4.parquet"

OUTPUT_DIR = "data/processed/v4"

TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "train.parquet"
)

VALIDATION_FILE = os.path.join(
    OUTPUT_DIR,
    "validation.parquet"
)

TEST_FILE = os.path.join(
    OUTPUT_DIR,
    "test.parquet"
)


# ==========================================================
# Main
# ==========================================================

def create_splits():

    print("=" * 60)
    print("V4 TIME-SERIES DATA SPLIT")
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
    ).reset_index(
        drop=True
    )

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
    # Split by time
    # ------------------------------------------------------

    train = df[
        df["date"].dt.year.isin(
            [2023, 2024]
        )
    ].copy()

    validation = df[
        df["date"].dt.year == 2025
    ].copy()

    test = df[
        df["date"].dt.year == 2026
    ].copy()

    # ------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------

    if len(train) == 0:
        raise ValueError(
            "Training set is empty."
        )

    if len(validation) == 0:
        raise ValueError(
            "Validation set is empty."
        )

    if len(test) == 0:
        raise ValueError(
            "Test set is empty."
        )

    # ------------------------------------------------------
    # Check date separation
    # ------------------------------------------------------

    train_dates = set(
        train["date"]
    )

    validation_dates = set(
        validation["date"]
    )

    test_dates = set(
        test["date"]
    )

    print()
    print(
        "Train ∩ Validation:",
        len(
            train_dates
            &
            validation_dates
        )
    )

    print(
        "Train ∩ Test:",
        len(
            train_dates
            &
            test_dates
        )
    )

    print(
        "Validation ∩ Test:",
        len(
            validation_dates
            &
            test_dates
        )
    )

    # ------------------------------------------------------
    # Print split information
    # ------------------------------------------------------

    def print_split(
        name,
        data
    ):

        print()
        print("=" * 60)
        print(name)
        print("=" * 60)

        print(
            f"Rows      : {len(data)}"
        )

        print(
            f"Start date: "
            f"{data['date'].min()}"
        )

        print(
            f"End date  : "
            f"{data['date'].max()}"
        )

        print()
        print("Cities:")

        print(
            data["city_name"]
            .value_counts()
            .sort_index()
        )

    print_split(
        "TRAIN",
        train
    )

    print_split(
        "VALIDATION",
        validation
    )

    print_split(
        "TEST",
        test
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    train.to_parquet(
        TRAIN_FILE,
        index=False
    )

    validation.to_parquet(
        VALIDATION_FILE,
        index=False
    )

    test.to_parquet(
        TEST_FILE,
        index=False
    )

    # ------------------------------------------------------
    # Verify saved files
    # ------------------------------------------------------

    train_check = pd.read_parquet(
        TRAIN_FILE
    )

    validation_check = pd.read_parquet(
        VALIDATION_FILE
    )

    test_check = pd.read_parquet(
        TEST_FILE
    )

    print()
    print("=" * 60)
    print("V4 SPLIT COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Train rows      : "
        f"{len(train_check)}"
    )

    print(
        f"Validation rows : "
        f"{len(validation_check)}"
    )

    print(
        f"Test rows       : "
        f"{len(test_check)}"
    )

    print()
    print(
        f"Saved: {TRAIN_FILE}"
    )

    print(
        f"Saved: {VALIDATION_FILE}"
    )

    print(
        f"Saved: {TEST_FILE}"
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    create_splits()