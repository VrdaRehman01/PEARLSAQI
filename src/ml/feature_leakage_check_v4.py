import pandas as pd
import numpy as np


FILE = "data/processed/features_v4.parquet"

TARGET = "target_aqi"


def main():

    print("=" * 60)
    print("V4 FEATURE LEAKAGE AUDIT")
    print("=" * 60)

    df = pd.read_parquet(FILE)

    df["date"] = pd.to_datetime(df["date"])

    print()
    print(f"Rows: {len(df)}")

    print()
    print("Date range:")
    print(f"Start: {df['date'].min()}")
    print(f"End  : {df['date'].max()}")

    # ------------------------------------------------------
    # Missing values
    # ------------------------------------------------------

    print()
    print("Missing values:")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("None")
    else:
        print(missing)

    # ------------------------------------------------------
    # Target verification
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("TARGET VERIFICATION")
    print("=" * 60)

    df = df.sort_values(
        ["city_id", "date"]
    ).reset_index(drop=True)

    expected_target = (
        df.groupby("city_id")["aqi"]
        .shift(-1)
    )

    valid = expected_target.notna()

    matches = (
        df.loc[valid, TARGET]
        ==
        expected_target.loc[valid]
    )

    print()
    print(
        f"Rows checked : {matches.sum()}"
    )

    print(
        f"Matching     : {matches.sum()}"
    )

    print(
        f"Mismatching  : {(~matches).sum()}"
    )

    match_rate = matches.mean() * 100

    print(
        f"Match rate   : {match_rate:.2f}%"
    )

    if not matches.all():

        print()
        print("WARNING: Target mismatch detected.")

        comparison = pd.DataFrame({
            "city_id":
                df.loc[valid, "city_id"],

            "date":
                df.loc[valid, "date"],

            "aqi":
                df.loc[valid, "aqi"],

            "target_aqi":
                df.loc[valid, TARGET],

            "expected":
                expected_target.loc[valid]
        })

        comparison["difference"] = (
            comparison["target_aqi"]
            -
            comparison["expected"]
        )

        print(
            comparison[
                comparison["difference"] != 0
            ].head(20).to_string(index=False)
        )

    # ------------------------------------------------------
    # Suspicious features
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("FEATURE SAFETY AUDIT")
    print("=" * 60)

    suspicious_names = [
        "target",
        "future",
        "next_day",
        "tomorrow"
    ]

    suspicious = []

    for column in df.columns:

        name = column.lower()

        if any(
            word in name
            for word in suspicious_names
        ):

            suspicious.append(column)

    print()
    print("Potentially suspicious columns:")

    for column in suspicious:
        print(f"- {column}")

    # ------------------------------------------------------
    # Correlation with target
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("TOP TARGET CORRELATIONS")
    print("=" * 60)

    numeric = df.select_dtypes(
        include=np.number
    )

    correlations = (
        numeric.corr()[TARGET]
        .drop(TARGET)
        .abs()
        .sort_values(
            ascending=False
        )
    )

    print(
        correlations.head(20).to_string()
    )

    # ------------------------------------------------------
    # Check impossible direct leakage
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("DIRECT LEAKAGE CHECK")
    print("=" * 60)

    leakage_features = []

    for column in df.columns:

        if column in [
            TARGET,
            "city_name",
            "date"
        ]:
            continue

        if "target" in column.lower():
            leakage_features.append(column)

    if leakage_features:

        print(
            "WARNING: Possible target-derived features:"
        )

        for column in leakage_features:
            print(f"- {column}")

    else:

        print(
            "PASS: No feature names contain 'target'."
        )

    # ------------------------------------------------------
    # Final
    # ------------------------------------------------------

    print()
    print("=" * 60)

    if (
        matches.all()
        and
        len(leakage_features) == 0
    ):

        print(
            "PASS: V4 target and feature safety checks passed."
        )

    else:

        print(
            "WARNING: Review the issues above."
        )


if __name__ == "__main__":

    main()