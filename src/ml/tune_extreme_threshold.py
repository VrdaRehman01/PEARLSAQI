import os
import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================================
# Configuration
# ==========================================================

VALIDATION_FILE = "data/processed/v3/validation.parquet"

MODEL_FILE = (
    "models/extreme_classifier/"
    "extreme_aqi_classifier.pkl"
)

TARGET_COLUMN = "target_aqi"

EXTREME_THRESHOLD = 200


# ==========================================================
# Main
# ==========================================================

def tune_threshold():

    print("=" * 60)
    print("EXTREME AQI CLASSIFICATION THRESHOLD TUNING")
    print("=" * 60)

    # ------------------------------------------------------
    # Load validation data
    # ------------------------------------------------------

    print()
    print("Loading validation data...")

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    # ------------------------------------------------------
    # Load classifier
    # ------------------------------------------------------

    print()
    print("Loading extreme-event classifier...")

    saved_model = joblib.load(
        MODEL_FILE
    )

    model = saved_model["model"]

    feature_columns = saved_model["features"]

    print(
        f"Features: {len(feature_columns)}"
    )

    # ------------------------------------------------------
    # Create X / y
    # ------------------------------------------------------

    X_validation = validation_df[
        feature_columns
    ]

    y_validation = (
        validation_df[TARGET_COLUMN]
        > EXTREME_THRESHOLD
    ).astype(int)

    # ------------------------------------------------------
    # Get probabilities
    # ------------------------------------------------------

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # ------------------------------------------------------
    # Test thresholds
    # ------------------------------------------------------

    results = []

    print()
    print("=" * 60)
    print("THRESHOLD RESULTS")
    print("=" * 60)

    print()

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'False Pos':<12}"
        f"{'False Neg':<12}"
    )

    print("-" * 72)

    for threshold in [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0
        )

        matrix = confusion_matrix(
            y_validation,
            predictions
        )

        # Matrix layout:
        #
        # [[TN FP]
        #  [FN TP]]

        if matrix.shape == (2, 2):

            false_positive = matrix[0, 1]

            false_negative = matrix[1, 0]

        else:

            false_positive = 0

            false_negative = 0

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{false_positive:<12}"
            f"{false_negative:<12}"
        )

        results.append({

            "threshold": threshold,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "false_positive": false_positive,

            "false_negative": false_negative
        })

    # ------------------------------------------------------
    # Results dataframe
    # ------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # ------------------------------------------------------
    # Best F1
    # ------------------------------------------------------

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    # ------------------------------------------------------
    # Best recall with reasonable precision
    # ------------------------------------------------------

    reasonable_precision = (
        results_df[
            results_df["precision"] >= 0.75
        ]
    )

    if len(reasonable_precision) > 0:

        best_recall = reasonable_precision.loc[
            reasonable_precision["recall"].idxmax()
        ]

    else:

        best_recall = best_f1

    # ------------------------------------------------------
    # Print recommendations
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("BEST F1 THRESHOLD")
    print("=" * 60)

    print(
        f"Threshold : "
        f"{best_f1['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_f1['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_f1['f1']:.4f}"
    )

    print()
    print("=" * 60)
    print("BEST RECALL WITH PRECISION >= 75%")
    print("=" * 60)

    print(
        f"Threshold : "
        f"{best_recall['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_recall['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_recall['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_recall['f1']:.4f}"
    )

    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------

    output_dir = (
        "models/extreme_classifier"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        "threshold_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)

    print(
        f"Saved to: {output_file}"
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    tune_threshold()