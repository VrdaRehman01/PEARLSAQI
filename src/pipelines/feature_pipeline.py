from src.features.feature_builder_v4 import build_features


def run_feature_pipeline():

    print("\n" + "=" * 60)
    print("STEP 2 : V4 FEATURE ENGINEERING")
    print("=" * 60)

    print("\nBuilding V4 features from DuckDB...")

    build_features()

    print("\nV4 Feature Pipeline Completed.")


if __name__ == "__main__":
    run_feature_pipeline()