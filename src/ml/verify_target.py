import pandas as pd


FILE = "data/processed/train.parquet"


def main():

    print("=" * 60)
    print("VERIFYING NEXT-DAY AQI TARGET")
    print("=" * 60)

    df = pd.read_parquet(FILE)

    # Sort correctly
    df = df.sort_values(
        ["city_id", "date"]
    ).reset_index(drop=True)

    # Get the next day's actual AQI
    df["next_day_aqi"] = (
        df.groupby("city_id")["aqi"]
        .shift(-1)
    )

    # Compare target with next day's AQI
    df["difference"] = (
        df["target_aqi"]
        - df["next_day_aqi"]
    )

    # Ignore the final row of each city
    comparison = df[
        df["next_day_aqi"].notna()
    ]

    total = len(comparison)

    matching = (
        comparison["difference"] == 0
    ).sum()

    mismatching = total - matching

    accuracy = (
        matching / total * 100
    )

    print()
    print(f"Rows checked : {total}")
    print(f"Matching     : {matching}")
    print(f"Mismatching  : {mismatching}")
    print(
        f"Match rate   : {accuracy:.2f}%"
    )

    print()
    print("=" * 60)
    print("SAMPLE VERIFICATION")
    print("=" * 60)

    sample = comparison[
        [
            "city_id",
            "date",
            "aqi",
            "target_aqi",
            "next_day_aqi",
            "difference"
        ]
    ].head(20)

    print(
        sample.to_string(
            index=False
        )
    )

    print()
    print("=" * 60)

    if mismatching == 0:

        print(
            "PASS: target_aqi is exactly "
            "the next day's AQI."
        )

    else:

        print(
            "WARNING: target_aqi does not "
            "always match the next day's AQI."
        )

        print()
        print("Examples of mismatches:")

        mismatches = comparison[
            comparison["difference"] != 0
        ]

        print(
            mismatches[
                [
                    "city_id",
                    "date",
                    "aqi",
                    "target_aqi",
                    "next_day_aqi",
                    "difference"
                ]
            ].head(20).to_string(
                index=False
            )
        )


if __name__ == "__main__":

    main()