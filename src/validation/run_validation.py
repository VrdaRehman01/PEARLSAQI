from src.features.feature_builder import FeatureBuilder
from src.validation.data_validator import DataValidator


def main():

    print("\n" + "=" * 60)
    print("AQI DATA VALIDATION PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # Build features
    # ---------------------------------------------------------

    builder = FeatureBuilder()

    df = builder.build()

    builder.close()

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    validator = DataValidator()

    passed = validator.validate(df)

    if not passed:

        raise SystemExit(
            "\nPipeline stopped because "
            "data validation failed."
        )

    print(
        "\nDataset is ready for model training."
    )


if __name__ == "__main__":

    main()