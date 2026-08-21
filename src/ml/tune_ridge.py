import os
import time
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ==========================================================
# Configuration
# ==========================================================

TRAIN_FILE = "data/processed/v3/train.parquet"
VALIDATION_FILE = "data/processed/v3/validation.parquet"

OUTPUT_DIR = "models/ridge"

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "ridge_tuning_results.csv"
)


# ==========================================================
# Features
# ==========================================================

EXCLUDED_COLUMNS = [
    "target_aqi",
    "city_name",
    "date",
    "season",
]

TARGET_COLUMN = "target_aqi"


# ==========================================================
# Models to test
# ==========================================================

MODELS = {

    "Model_1_Ridge_001": {
        "alpha": 0.01
    },

    "Model_2_Ridge_01": {
        "alpha": 0.1
    },

    "Model_3_Ridge_1": {
        "alpha": 1.0
    },

    "Model_4_Ridge_10": {
        "alpha": 10.0
    },

    "Model_5_Ridge_100": {
        "alpha": 100.0
    },

    "Model_6_Ridge_1000": {
        "alpha": 1000.0
    },

}


# ==========================================================
# Metrics
# ==========================================================

def evaluate_model(
    model,
    X,
    y
):

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = mean_squared_error(
        y,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y,
        predictions
    )

    return mae, rmse, r2


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("RIDGE HYPERPARAMETER TUNING")
    print("=" * 60)

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------

    print()
    print("Loading training data...")

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    print()
    print("Loading validation data...")

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    # ------------------------------------------------------
    # Build feature list
    # ------------------------------------------------------

    feature_columns = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    print()
    print(
        f"Number of features: {len(feature_columns)}"
    )

    # ------------------------------------------------------
    # X / y
    # ------------------------------------------------------

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_validation = validation_df[
        feature_columns
    ]

    y_validation = validation_df[
        TARGET_COLUMN
    ]

    # ------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------

    results = []

    for model_name, params in MODELS.items():

        print()
        print("=" * 60)
        print(model_name)
        print("=" * 60)

        print(
            f"alpha: {params['alpha']}"
        )

        start_time = time.time()

        # Standardization is important for Ridge
        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge(
                    alpha=params["alpha"]
                )
            )
        ])

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time.time() - start_time
        )

        # --------------------------------------------------
        # Training metrics
        # --------------------------------------------------

        train_mae, train_rmse, train_r2 = evaluate_model(
            model,
            X_train,
            y_train
        )

        # --------------------------------------------------
        # Validation metrics
        # --------------------------------------------------

        val_mae, val_rmse, val_r2 = evaluate_model(
            model,
            X_validation,
            y_validation
        )

        print()
        print(
            f"Training MAE  : {train_mae:.4f}"
        )

        print(
            f"Training RMSE : {train_rmse:.4f}"
        )

        print(
            f"Training R²   : {train_r2:.4f}"
        )

        print()
        print(
            f"Validation MAE  : {val_mae:.4f}"
        )

        print(
            f"Validation RMSE : {val_rmse:.4f}"
        )

        print(
            f"Validation R²   : {val_r2:.4f}"
        )

        print(
            f"Training time   : {training_time:.2f}s"
        )

        results.append({

            "model": model_name,

            "alpha": params["alpha"],

            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_r2": train_r2,

            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,

            "training_time_seconds":
                training_time

        })

    # ------------------------------------------------------
    # Results dataframe
    # ------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # Sort primarily by MAE
    results_df = results_df.sort_values(
        "val_mae"
    )

    print()
    print("=" * 60)
    print("RIDGE TUNING RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print()
    print(
        f"Results saved to: {RESULTS_FILE}"
    )

    # ------------------------------------------------------
    # Best model
    # ------------------------------------------------------

    best = results_df.iloc[0]

    print()
    print("=" * 60)
    print("BEST RIDGE MODEL")
    print("=" * 60)

    print(
        f"Model : {best['model']}"
    )

    print(
        f"Alpha : {best['alpha']}"
    )

    print(
        f"MAE   : {best['val_mae']:.4f}"
    )

    print(
        f"RMSE  : {best['val_rmse']:.4f}"
    )

    print(
        f"R²    : {best['val_r2']:.4f}"
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    main()