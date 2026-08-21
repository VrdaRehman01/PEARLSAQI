import os
import time
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# Configuration
# ==========================================================

TRAIN_FILE = "data/processed/v3/train.parquet"
VALIDATION_FILE = "data/processed/v3/validation.parquet"

OUTPUT_DIR = "models/xgboost"

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "xgboost_v3_tuning_results.csv"
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
# XGBoost experiments
# ==========================================================

MODELS = {

    "Model_1_Baseline": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },

    "Model_2_Regularized": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },

    "Model_3_StrongRegularization": {
        "n_estimators": 600,
        "learning_rate": 0.04,
        "max_depth": 5,
        "min_child_weight": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },

    "Model_4_Shallow": {
        "n_estimators": 600,
        "learning_rate": 0.04,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },

    "Model_5_Balanced": {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 3.0,
    },

    "Model_6_Conservative": {
        "n_estimators": 800,
        "learning_rate": 0.025,
        "max_depth": 4,
        "min_child_weight": 8,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },

    "Model_7_MoreRegularized": {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 8.0,
    },

}


# ==========================================================
# Metrics
# ==========================================================

def evaluate_model(model, X, y):

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
    print("XGBOOST V3 HYPERPARAMETER TUNING")
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
    # Feature list
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
    # Experiments
    # ------------------------------------------------------

    results = []

    for model_name, params in MODELS.items():

        print()
        print("=" * 60)
        print(model_name)
        print("=" * 60)

        print(
            f"n_estimators     : {params['n_estimators']}"
        )

        print(
            f"learning_rate    : {params['learning_rate']}"
        )

        print(
            f"max_depth        : {params['max_depth']}"
        )

        print(
            f"min_child_weight : {params['min_child_weight']}"
        )

        print(
            f"subsample        : {params['subsample']}"
        )

        print(
            f"colsample        : {params['colsample_bytree']}"
        )

        print(
            f"reg_alpha        : {params['reg_alpha']}"
        )

        print(
            f"reg_lambda       : {params['reg_lambda']}"
        )

        start_time = time.time()

        model = XGBRegressor(

            n_estimators=params["n_estimators"],

            learning_rate=params["learning_rate"],

            max_depth=params["max_depth"],

            min_child_weight=params[
                "min_child_weight"
            ],

            subsample=params["subsample"],

            colsample_bytree=params[
                "colsample_bytree"
            ],

            reg_alpha=params["reg_alpha"],

            reg_lambda=params["reg_lambda"],

            objective="reg:squarederror",

            eval_metric="rmse",

            random_state=42,

            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train,
            verbose=False
        )

        training_time = (
            time.time() - start_time
        )

        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        train_mae, train_rmse, train_r2 = (
            evaluate_model(
                model,
                X_train,
                y_train
            )
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        val_mae, val_rmse, val_r2 = (
            evaluate_model(
                model,
                X_validation,
                y_validation
            )
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

            **params,

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
    # Results
    # ------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "val_mae"
    )

    print()
    print("=" * 60)
    print("XGBOOST TUNING RESULTS")
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
    # Best
    # ------------------------------------------------------

    best = results_df.iloc[0]

    print()
    print("=" * 60)
    print("BEST XGBOOST MODEL")
    print("=" * 60)

    print(
        f"Model : {best['model']}"
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


if __name__ == "__main__":

    main()