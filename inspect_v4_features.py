import pandas as pd

file = "data/processed/features_v4.parquet"

df = pd.read_parquet(file)

print("=== V4 FEATURE FILE ===")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nDate range:")
print(df["date"].min(), "->", df["date"].max())

print("\nCities:")
print(df["city_name"].nunique())

print("\nTarget:")
print("target_aqi" in df.columns)

print("\nMissing values:")
missing = df.isna().sum()
print(missing[missing > 0].to_string())

print("\nLast 5 rows:")
print(
    df[
        ["city_name", "date", "aqi", "target_aqi"]
    ]
    .tail()
    .to_string(index=False)
)
