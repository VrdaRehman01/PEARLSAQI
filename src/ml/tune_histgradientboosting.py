import os
import time
import warnings

import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ==========================================================
# Configuration
# ==========================================================

TRAIN_FILE = "data/processed/v3/train.parquet"
VALIDATION_FILE = "data/processed/v3/validation.parquet"

RESULTS_DIR = "models/histgradientboosting"
RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "histgradientboosting_tuning_results.csv"
)


# ==========================================================
# Columns
# ==========================================================

EXCLUDED_COLUMNS = [
    "target_aqi",
    "city_name",
    "date",
]

TARGET_COLUMN = "target_aqi"


# ==========================================================
# Evaluation
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

    print("=" * 70)
    print("HISTGRADIENTBOOSTING HYPERPARAMETER TUNING")
    print("=" * 70)

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

    if TARGET_COLUMN in feature_columns:

        raise ValueError(
            "target_aqi cannot be used as a feature."
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in validation_df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing validation features: {missing_features}"
        )

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

    print()
    print(
        f"Number of features: {len(feature_columns)}"
    )

    # ------------------------------------------------------
    # Hyperparameter configurations
    # ------------------------------------------------------

    configurations = [

        # ----------------------------------------------
        # Baseline
        # ----------------------------------------------

        {
            "name": "Model_1_Baseline",

            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "max_depth": None,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },

        # ----------------------------------------------
        # Lower learning rate
        # ----------------------------------------------

        {
            "name": "Model_2_LowLearningRate",

            "learning_rate": 0.03,
            "max_iter": 500,
            "max_leaf_nodes": 31,
            "max_depth": None,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },

        # ----------------------------------------------
        # More leaves
        # ----------------------------------------------

        {
            "name": "Model_3_MoreLeaves",

            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 63,
            "max_depth": None,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },

        # ----------------------------------------------
        # Fewer leaves
        # ----------------------------------------------

        {
            "name": "Model_4_FewerLeaves",

            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 15,
            "max_depth": None,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },

        # ----------------------------------------------
        # Stronger regularization
        # ----------------------------------------------

        {
            "name": "Model_5_StrongRegularization",

            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "max_depth": None,
            "min_samples_leaf": 30,
            "l2_regularization": 5.0,
        },

        # ----------------------------------------------
        # Higher min samples
        # ----------------------------------------------

        {
            "name": "Model_6_MoreRegularization",

            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "max_depth": None,
            "min_samples_leaf": 50,
            "l2_regularization": 10.0,
        },

        # ----------------------------------------------
        # Deeper model
        # ----------------------------------------------

        {
            "name": "Model_7_Deeper",

            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 63,
            "max_depth": 10,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },

        # ----------------------------------------------
        # Conservative model
        # ----------------------------------------------

        {
            "name": "Model_8_Conservative",

            "learning_rate": 0.03,
            "max_iter": 500,
            "max_leaf_nodes": 15,
            "max_depth": None,
            "min_samples_leaf": 30,
            "l2_regularization": 5.0,
        },

        # ----------------------------------------------
        # Aggressive model
        # ----------------------------------------------

        {
            "name": "Model_9_Aggressive",

            "learning_rate": 0.08,
            "max_iter": 300,
            "max_leaf_nodes": 63,
            "max_depth": None,
            "min_samples_leaf": 15,
            "l2_regularization": 0.5,
        },

        # ----------------------------------------------
        # Balanced model
        # ----------------------------------------------

        {
            "name": "Model_10_Balanced",

            "learning_rate": 0.04,
            "max_iter": 400,
            "max_leaf_nodes": 31,
            "max_depth": None,
            "min_samples_leaf": 25,
            "l2_regularization": 2.0,
        },
    ]

    # ------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------

    results = []

    for config in configurations:

        print()
        print("=" * 70)
        print(
            f"TRAINING {config['name']}"
        )
        print("=" * 70)

        print(
            f"learning_rate      : "
            f"{config['learning_rate']}"
        )

        print(
            f"max_iter           : "
            f"{config['max_iter']}"
        )

        print(
            f"max_leaf_nodes     : "
            f"{config['max_leaf_nodes']}"
        )

        print(
            f"max_depth          : "
            f"{config['max_depth']}"
        )

        print(
            f"min_samples_leaf   : "
            f"{config['min_samples_leaf']}"
        )

        print(
            f"l2_regularization  : "
            f"{config['l2_regularization']}"
        )

        model = HistGradientBoostingRegressor(

            learning_rate=
                config["learning_rate"],

            max_iter=
                config["max_iter"],

            max_leaf_nodes=
                config["max_leaf_nodes"],

            max_depth=
                config["max_depth"],

            min_samples_leaf=
                config["min_samples_leaf"],

            l2_regularization=
                config["l2_regularization"],

            loss="squared_error",

            random_state=42
        )

        start_time = time.time()

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

        train_mae, train_rmse, train_r2 = (
            evaluate_model(
                model,
                X_train,
                y_train
            )
        )

        # --------------------------------------------------
        # Validation metrics
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
            f"Training MAE       : {train_mae:.4f}"
        )

        print(
            f"Training RMSE      : {train_rmse:.4f}"
        )

        print(
            f"Training R²        : {train_r2:.4f}"
        )

        print()
        print(
            f"Validation MAE     : {val_mae:.4f}"
        )

        print(
            f"Validation RMSE    : {val_rmse:.4f}"
        )

        print(
            f"Validation R²      : {val_r2:.4f}"
        )

        print(
            f"Training time      : "
            f"{training_time:.2f}s"
        )

        results.append({

            "model":
                config["name"],

            "learning_rate":
                config["learning_rate"],

            "max_iter":
                config["max_iter"],

            "max_leaf_nodes":
                config["max_leaf_nodes"],

            "max_depth":
                config["max_depth"],

            "min_samples_leaf":
                config["min_samples_leaf"],

            "l2_regularization":
                config["l2_regularization"],

            "train_mae":
                train_mae,

            "train_rmse":
                train_rmse,

            "train_r2":
                train_r2,

            "val_mae":
                val_mae,

            "val_rmse":
                val_rmse,

            "val_r2":
                val_r2,

            "training_time_seconds":
                training_time,
        })

    # ------------------------------------------------------
    # Results table
    # ------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by=[
            "val_mae",
            "val_rmse"
        ],
        ascending=[
            True,
            True
        ]
    )

    print()
    print("=" * 70)
    print("HISTGRADIENTBOOSTING TUNING RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)

    print(
        f"Results saved to: {RESULTS_FILE}"
    )

    # ------------------------------------------------------
    # Best model
    # ------------------------------------------------------

    best = results_df.iloc[0]

    print()
    print("=" * 70)
    print("BEST HISTGRADIENTBOOSTING MODEL")
    print("=" * 70)

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

    print()
    print("Hyperparameters:")

    print(
        f"learning_rate     : "
        f"{best['learning_rate']}"
    )

    print(
        f"max_iter          : "
        f"{best['max_iter']}"
    )

    print(
        f"max_leaf_nodes    : "
        f"{best['max_leaf_nodes']}"
    )

    print(
        f"max_depth         : "
        f"{best['max_depth']}"
    )

    print(
        f"min_samples_leaf  : "
        f"{best['min_samples_leaf']}"
    )

    print(
        f"l2_regularization : "
        f"{best['l2_regularization']}"
    )


if __name__ == "__main__":

    main()