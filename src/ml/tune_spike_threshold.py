from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


VALIDATION_PATH = Path(
    "data/processed/v3/validation.parquet"
)

MODEL_PATH = Path(
    "models/spike_classifier/aqi_spike_classifier.pkl"
)

OUTPUT_DIR = Path(
    "models/spike_classifier"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SPIKE_THRESHOLD = 40

print("=" * 60)
print("AQI SPIKE CLASSIFIER THRESHOLD TUNING")
print("=" * 60)

validation = pd.read_parquet(
    VALIDATION_PATH
)

bundle = joblib.load(
    MODEL_PATH
)

model = bundle["model"]
features = bundle["features"]

X = validation[features]

change = (
    validation["target_aqi"]
    - validation["aqi"]
)

y = (
    change.abs() >= SPIKE_THRESHOLD
).astype(int)

probability = model.predict_proba(X)[:, 1]


results = []

print("\nThreshold   Precision   Recall      F1"
      "       False Pos   False Neg")

for threshold in np.arange(
    0.05,
    0.96,
    0.05
):

    prediction = (
        probability >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y,
        prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        prediction,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        prediction,
    )

    tn, fp, fn, tp = matrix.ravel()

    print(
        f"{threshold:.2f}        "
        f"{precision:.4f}      "
        f"{recall:.4f}      "
        f"{f1:.4f}      "
        f"{fp:<10} "
        f"{fn}"
    )

    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive": fp,
            "false_negative": fn,
        }
    )


results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "f1",
    ascending=False,
)

best = results_df.iloc[0]

print("\n" + "=" * 60)
print("BEST F1 THRESHOLD")
print("=" * 60)

print(
    f"Threshold : {best['threshold']:.2f}"
)

print(
    f"Precision : {best['precision']:.4f}"
)

print(
    f"Recall    : {best['recall']:.4f}"
)

print(
    f"F1        : {best['f1']:.4f}"
)

output_path = (
    OUTPUT_DIR /
    "spike_threshold_results.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

print(
    f"\nSaved to: {output_path}"
)