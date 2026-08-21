from src.database.connection import get_connection


def create_splits():

    conn = get_connection()

    query = """
    SELECT *
    FROM features
    ORDER BY date, city_id
    """

    df = conn.execute(query).fetchdf()

    print("=" * 60)
    print("TIME-SERIES DATA SPLIT")
    print("=" * 60)

    print(f"Total rows: {len(df)}")

    # --------------------------------------------------
    # TRAIN
    # 2023 + 2024
    # --------------------------------------------------

    train = df[
        df["year"].isin([2023, 2024])
    ].copy()

    # --------------------------------------------------
    # VALIDATION
    # 2025
    # --------------------------------------------------

    validation = df[
        df["year"] == 2025
    ].copy()

    # --------------------------------------------------
    # TEST
    # 2026
    # --------------------------------------------------

    test = df[
        df["year"] == 2026
    ].copy()

    print()
    print("TRAIN")
    print("-" * 40)
    print(f"Rows      : {len(train)}")
    print(f"Start date: {train['date'].min()}")
    print(f"End date  : {train['date'].max()}")

    print()
    print("VALIDATION")
    print("-" * 40)
    print(f"Rows      : {len(validation)}")
    print(f"Start date: {validation['date'].min()}")
    print(f"End date  : {validation['date'].max()}")

    print()
    print("TEST")
    print("-" * 40)
    print(f"Rows      : {len(test)}")
    print(f"Start date: {test['date'].min()}")
    print(f"End date  : {test['date'].max()}")

    # --------------------------------------------------
    # Check for overlap
    # --------------------------------------------------

    train_dates = set(train["date"])
    validation_dates = set(validation["date"])
    test_dates = set(test["date"])

    print()
    print("=" * 60)
    print("DATA LEAKAGE CHECK")
    print("=" * 60)

    print(
        "Train ∩ Validation:",
        len(train_dates & validation_dates)
    )

    print(
        "Train ∩ Test:",
        len(train_dates & test_dates)
    )

    print(
        "Validation ∩ Test:",
        len(validation_dates & test_dates)
    )

    # --------------------------------------------------
    # Save split datasets
    # --------------------------------------------------

    train.to_parquet(
        "data/processed/train.parquet",
        index=False
    )

    validation.to_parquet(
        "data/processed/validation.parquet",
        index=False
    )

    test.to_parquet(
        "data/processed/test.parquet",
        index=False
    )

    print()
    print("=" * 60)
    print("SPLITS SAVED")
    print("=" * 60)

    print("data/processed/train.parquet")
    print("data/processed/validation.parquet")
    print("data/processed/test.parquet")


if __name__ == "__main__":
    create_splits()