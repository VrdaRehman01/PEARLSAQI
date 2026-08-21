import pandas as pd


TRAIN_FILE = "data/processed/train.parquet"
VALIDATION_FILE = "data/processed/validation.parquet"
TEST_FILE = "data/processed/test.parquet"

TARGET = "target_aqi"


def check_dataset(df, name):

    print()
    print("=" * 60)
    print(f"{name.upper()} DATASET CHECK")
    print("=" * 60)

    print(f"Rows: {len(df)}")

    print()
    print("Date range:")
    print(f"Start: {df['date'].min()}")
    print(f"End  : {df['date'].max()}")

    print()
    print("Missing values:")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("None")
    else:
        print(missing)

    print()
    print("Target statistics:")

    print(
        df[TARGET].describe()
    )

    print()
    print("Target vs current AQI:")

    correlation = df[
        ["aqi", TARGET]
    ].corr().iloc[0, 1]

    print(
        f"Correlation: {correlation:.4f}"
    )


def check_feature_names(df):

    print()
    print("=" * 60)
    print("FEATURE LEAKAGE NAME CHECK")
    print("=" * 60)

    suspicious_words = [
        "target",
        "future",
        "next",
        "tomorrow",
        "lead"
    ]

    suspicious = []

    for column in df.columns:

        column_lower = column.lower()

        for word in suspicious_words:

            if word in column_lower:

                suspicious.append(column)

                break

    if suspicious:

        print("Potentially suspicious columns:")

        for column in suspicious:
            print(f"  - {column}")

    else:

        print(
            "No suspicious feature names found."
        )


def check_temporal_relationship(df):

    print()
    print("=" * 60)
    print("TEMPORAL TARGET CHECK")
    print("=" * 60)

    sample = df[
        [
            "city_id",
            "date",
            "aqi",
            TARGET
        ]
    ].sort_values(
        ["city_id", "date"]
    )

    print()
    print("Sample rows:")

    print(
        sample.head(15).to_string(
            index=False
        )
    )

    print()
    print(
        "The target should represent the AQI "
        "of the following day."
    )


def main():

    print("=" * 60)
    print("AQI FEATURE LEAKAGE AUDIT")
    print("=" * 60)

    train = pd.read_parquet(
        TRAIN_FILE
    )

    validation = pd.read_parquet(
        VALIDATION_FILE
    )

    test = pd.read_parquet(
        TEST_FILE
    )

    check_dataset(
        train,
        "training"
    )

    check_dataset(
        validation,
        "validation"
    )

    check_dataset(
        test,
        "test"
    )

    check_feature_names(
        train
    )

    check_temporal_relationship(
        train
    )

    print()
    print("=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()