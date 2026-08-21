import os
import time

import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ==========================================================
# Configuration
# ==========================================================

TRAIN_FILE = "data/processed/v3/train.parquet"
VALIDATION_FILE = "data/processed/v3/validation.parquet"

MODEL_DIR = "models/extreme_classifier"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "extreme_aqi_classifier.pkl"
)

TARGET_COLUMN = "target_aqi"

# AQI threshold for extreme-event prediction
EXTREME_THRESHOLD = 200


# ==========================================================
# Feature exclusions
# ==========================================================

EXCLUDED_COLUMNS = [
    TARGET_COLUMN,
    "city_name",
    "date",
]


# ==========================================================
# Load data
# ==========================================================

def load_data():

    print("=" * 60)
    print("AQI EXTREME EVENT CLASSIFIER")
    print("=" * 60)

    print()
    print("Loading training data...")

    train_df = pd.read_parquet(TRAIN_FILE)

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

    return train_df, validation_df


# ==========================================================
# Build features
# ==========================================================

def build_features(train_df, validation_df):

    print()
    print("Building feature list...")

    feature_columns = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    missing_features = [
        column
        for column in feature_columns
        if column not in validation_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Features missing from validation data: "
            f"{missing_features}"
        )

    if TARGET_COLUMN in feature_columns:

        raise ValueError(
            "target_aqi cannot be used as a feature."
        )

    print()
    print(
        f"Number of features: {len(feature_columns)}"
    )

    return feature_columns


# ==========================================================
# Create classification targets
# ==========================================================

def create_targets(
    train_df,
    validation_df
):

    y_train = (
        train_df[TARGET_COLUMN]
        > EXTREME_THRESHOLD
    ).astype(int)

    y_validation = (
        validation_df[TARGET_COLUMN]
        > EXTREME_THRESHOLD
    ).astype(int)

    print()
    print("=" * 60)
    print("EXTREME EVENT DISTRIBUTION")
    print("=" * 60)

    print()
    print(
        f"Threshold: AQI > {EXTREME_THRESHOLD}"
    )

    print()
    print("TRAINING")

    print(
        f"Normal events   : {(y_train == 0).sum()}"
    )

    print(
        f"Extreme events  : {(y_train == 1).sum()}"
    )

    print(
        f"Extreme rate    : {y_train.mean() * 100:.2f}%"
    )

    print()
    print("VALIDATION")

    print(
        f"Normal events   : {(y_validation == 0).sum()}"
    )

    print(
        f"Extreme events  : {(y_validation == 1).sum()}"
    )

    print(
        f"Extreme rate    : {y_validation.mean() * 100:.2f}%"
    )

    return y_train, y_validation


# ==========================================================
# Evaluate classifier
# ==========================================================

def evaluate_classifier(
    model,
    X,
    y,
    dataset_name
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y,
            probabilities
        )

    except ValueError:

        roc_auc = float("nan")

    matrix = confusion_matrix(
        y,
        predictions
    )

    print()
    print("=" * 60)
    print(f"{dataset_name} RESULTS")
    print("=" * 60)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print()
    print("CONFUSION MATRIX")

    print(
        matrix
    )

    print()
    print("CLASSIFICATION REPORT")

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Normal",
                "Extreme"
            ],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# ==========================================================
# Training
# ==========================================================

def train_classifier():

    train_df, validation_df = load_data()

    feature_columns = build_features(
        train_df,
        validation_df
    )

    y_train, y_validation = create_targets(
        train_df,
        validation_df
    )

    X_train = train_df[
        feature_columns
    ]

    X_validation = validation_df[
        feature_columns
    ]

    print()
    print("=" * 60)
    print("TRAINING EXTREME EVENT CLASSIFIER")
    print("=" * 60)

    start_time = time.time()

    model = HistGradientBoostingClassifier(

        learning_rate=0.05,

        max_iter=300,

        max_leaf_nodes=15,

        max_depth=None,

        min_samples_leaf=30,

        l2_regularization=5.0,

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start_time
    )

    print()
    print(
        f"Training completed in "
        f"{training_time:.2f}s"
    )

    # ------------------------------------------------------
    # Training evaluation
    # ------------------------------------------------------

    train_metrics = evaluate_classifier(
        model,
        X_train,
        y_train,
        "TRAINING"
    )

    # ------------------------------------------------------
    # Validation evaluation
    # ------------------------------------------------------

    validation_metrics = evaluate_classifier(
        model,
        X_validation,
        y_validation,
        "VALIDATION"
    )

    # ------------------------------------------------------
    # Save model
    # ------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    import joblib

    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
            "threshold": EXTREME_THRESHOLD
        },
        MODEL_FILE
    )

    print()
    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(
        f"Model path: {MODEL_FILE}"
    )

    # ------------------------------------------------------
    # Final summary
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("EXTREME EVENT CLASSIFIER SUMMARY")
    print("=" * 60)

    print(
        f"Threshold       : AQI > {EXTREME_THRESHOLD}"
    )

    print(
        f"Validation F1   : "
        f"{validation_metrics['f1']:.4f}"
    )

    print(
        f"Validation Recall: "
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{validation_metrics['roc_auc']:.4f}"
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    train_classifier()