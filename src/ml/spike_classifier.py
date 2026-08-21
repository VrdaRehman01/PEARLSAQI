"""
PEARLSAQI - AQI SPIKE / SHOCK CLASSIFIER V2

Predicts whether tomorrow's AQI will change sharply
relative to today's AQI.

Spike definition:
abs(target_aqi - aqi) >= SPIKE_THRESHOLD

IMPORTANT:
The classifier only uses features that are actually
available in the production feature dataset.

This prevents train/inference feature mismatch.
"""

from pathlib import Path

import joblib
import numpy as np
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


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "v3"
    / "train.parquet"
)

VALIDATION_PATH = (
    ROOT
    / "data"
    / "processed"
    / "v3"
    / "validation.parquet"
)

PRODUCTION_FEATURE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "features_v4.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "spike_classifier"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    OUTPUT_DIR
    / "aqi_spike_classifier.pkl"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "spike_classifier_metrics.csv"
)

FEATURES_PATH = (
    OUTPUT_DIR
    / "spike_classifier_features.json"
)


# ============================================================
# CONFIG
# ============================================================

SPIKE_THRESHOLD = 40

TARGET = "target_aqi"

EXCLUDED_COLUMNS = {
    TARGET,
    "city_name",
    "date",
}


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("PEARLSAQI - AQI SPIKE / SHOCK CLASSIFIER V2")
print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training data...")

train = pd.read_parquet(
    TRAIN_PATH
)

validation = pd.read_parquet(
    VALIDATION_PATH
)

production_features = pd.read_parquet(
    PRODUCTION_FEATURE_PATH
)

print(
    f"Training rows          : {len(train)}"
)

print(
    f"Validation rows        : {len(validation)}"
)

print(
    f"Production feature rows: "
    f"{len(production_features)}"
)


# ============================================================
# CREATE SPIKE TARGET
# ============================================================

train_change = (
    train[TARGET]
    - train["aqi"]
)

validation_change = (
    validation[TARGET]
    - validation["aqi"]
)


y_train = (
    train_change.abs()
    >= SPIKE_THRESHOLD
).astype(int)


y_validation = (
    validation_change.abs()
    >= SPIKE_THRESHOLD
).astype(int)


print("\nSpike definition:")

print(
    f"ABS(next_day_AQI - current_AQI) "
    f">= {SPIKE_THRESHOLD}"
)


print("\nTRAINING")

print(
    f"Normal events : "
    f"{(y_train == 0).sum()}"
)

print(
    f"Spike events  : "
    f"{(y_train == 1).sum()}"
)

print(
    f"Spike rate    : "
    f"{y_train.mean() * 100:.2f}%"
)


print("\nVALIDATION")

print(
    f"Normal events : "
    f"{(y_validation == 0).sum()}"
)

print(
    f"Spike events  : "
    f"{(y_validation == 1).sum()}"
)

print(
    f"Spike rate    : "
    f"{y_validation.mean() * 100:.2f}%"
)


# ============================================================
# PRODUCTION-AVAILABLE FEATURES
# ============================================================

production_feature_columns = {

    column
    for column in production_features.columns

    if column not in EXCLUDED_COLUMNS
}


# ============================================================
# TRAINING FEATURES
# ============================================================

training_feature_columns = {

    column
    for column in train.columns

    if column not in EXCLUDED_COLUMNS
}


# ============================================================
# COMMON FEATURES
# ============================================================

feature_columns = sorted(

    training_feature_columns
    & production_feature_columns

)


# ============================================================
# CHECK FEATURES
# ============================================================

missing_from_training = sorted(

    production_feature_columns
    - training_feature_columns

)


missing_from_production = sorted(

    training_feature_columns
    - production_feature_columns

)


print("\nFEATURE ALIGNMENT")

print(
    f"Training features available   : "
    f"{len(training_feature_columns)}"
)

print(
    f"Production features available : "
    f"{len(production_feature_columns)}"
)

print(
    f"Common features used           : "
    f"{len(feature_columns)}"
)


if missing_from_production:

    print(
        "\nFeatures intentionally removed "
        "because they are unavailable "
        "during production:"
    )

    for feature in missing_from_production:

        print(
            f" - {feature}"
        )


if not feature_columns:

    raise ValueError(
        "No common production/training "
        "features were found."
    )


# ============================================================
# PREPARE X
# ============================================================

X_train = train[
    feature_columns
].copy()


X_validation = validation[
    feature_columns
].copy()


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

X_train = X_train.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)


X_validation = X_validation.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)


X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)


X_validation = X_validation.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# PRINT FEATURES
# ============================================================

print("\nSPIKE CLASSIFIER FEATURES")

for index, feature in enumerate(
    feature_columns,
    start=1
):

    print(
        f"{index:03d}. {feature}"
    )


# ============================================================
# MODEL
# ============================================================

print(
    "\nTraining "
    "HistGradientBoostingClassifier..."
)


model = HistGradientBoostingClassifier(

    learning_rate=0.05,

    max_iter=300,

    max_leaf_nodes=31,

    min_samples_leaf=20,

    l2_regularization=2.0,

    random_state=42,

)


model.fit(
    X_train,
    y_train
)


print(
    "Spike classifier training completed."
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    name,
    X,
    y
):

    probability = (
        model
        .predict_proba(X)[:, 1]
    )


    prediction = (
        probability >= 0.50
    ).astype(int)


    accuracy = accuracy_score(
        y,
        prediction
    )


    precision = precision_score(
        y,
        prediction,
        zero_division=0
    )


    recall = recall_score(
        y,
        prediction,
        zero_division=0
    )


    f1 = f1_score(
        y,
        prediction,
        zero_division=0
    )


    roc_auc = roc_auc_score(
        y,
        probability
    )


    matrix = confusion_matrix(
        y,
        prediction
    )


    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)


    print(
        f"Accuracy  : "
        f"{accuracy:.4f}"
    )


    print(
        f"Precision : "
        f"{precision:.4f}"
    )


    print(
        f"Recall    : "
        f"{recall:.4f}"
    )


    print(
        f"F1 Score  : "
        f"{f1:.4f}"
    )


    print(
        f"ROC-AUC   : "
        f"{roc_auc:.4f}"
    )


    print(
        "\nCONFUSION MATRIX"
    )

    print(
        matrix
    )


    print(
        "\nCLASSIFICATION REPORT"
    )


    print(
        classification_report(

            y,

            prediction,

            target_names=[
                "Normal",
                "Spike",
            ],

            zero_division=0,

        )
    )


    return {

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "roc_auc":
            roc_auc,

    }


# ============================================================
# TRAINING / VALIDATION
# ============================================================

train_metrics = evaluate_model(

    "TRAINING",

    X_train,

    y_train,

)


validation_metrics = evaluate_model(

    "VALIDATION",

    X_validation,

    y_validation,

)


# ============================================================
# DIRECTION ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("SPIKE DIRECTION ANALYSIS")
print("=" * 70)


validation_change_values = (
    validation_change.values
)


upward = (
    validation_change_values
    >= SPIKE_THRESHOLD
)


downward = (
    validation_change_values
    <= -SPIKE_THRESHOLD
)


print(
    f"Upward spikes   : "
    f"{upward.sum()}"
)


print(
    f"Downward spikes : "
    f"{downward.sum()}"
)


if upward.sum() > 0:

    print(
        f"Mean upward change: "
        f"{validation_change_values[upward].mean():.2f}"
    )


if downward.sum() > 0:

    print(
        f"Mean downward change: "
        f"{validation_change_values[downward].mean():.2f}"
    )


# ============================================================
# SAVE MODEL BUNDLE
# ============================================================

bundle = {

    "model":
        model,

    "features":
        feature_columns,

    "threshold":
        SPIKE_THRESHOLD,

    "prediction_threshold":
        0.50,

    "feature_count":
        len(feature_columns),

    "model_type":
        type(model).__name__,

}


joblib.dump(
    bundle,
    MODEL_PATH
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

with open(
    FEATURES_PATH,
    "w",
    encoding="utf-8"
) as f:

    import json

    json.dump(
        feature_columns,
        f,
        indent=2
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame(

    [

        {
            "dataset":
                "training",

            **train_metrics,

        },

        {
            "dataset":
                "validation",

            **validation_metrics,

        },

    ]

)


metrics.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("SPIKE CLASSIFIER SAVED")
print("=" * 70)


print(
    f"Model    : {MODEL_PATH}"
)


print(
    f"Features : {FEATURES_PATH}"
)


print(
    f"Metrics  : {METRICS_PATH}"
)


print(
    f"Feature count: "
    f"{len(feature_columns)}"
)


print(
    "\nSpike classifier V2 completed successfully."
)