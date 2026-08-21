import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = (
    "models/benchmark/v4/final_candidate/"
    "xgboost_v4_final_108.pkl"
)

FEATURE_FILE = (
    "models/benchmark/v4/final_candidate/"
    "feature_columns.json"
)

TRAIN_FILE = "data/processed/v4/train.parquet"

OUTPUT_DIR = "models/benchmark/v4/shap"

TOP_N = 20
EXPLANATION_SIZE = 500


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLSAQI SHAP ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading final 108-feature model...")

    model = joblib.load(
        MODEL_FILE
    )

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        feature_columns = json.load(f)

    print(
        f"Feature count: {len(feature_columns)}"
    )

    if len(feature_columns) != 108:
        raise RuntimeError(
            f"Expected 108 features, got {len(feature_columns)}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print()
    print("Loading training data...")

    df = pd.read_parquet(
        TRAIN_FILE
    )

    # --------------------------------------------------------
    # Recreate winning feature
    # --------------------------------------------------------

    epsilon = 1e-6

    df["pm25_pollution_ratio"] = (
        df["pm25"]
        / (
            df["pollution_sum"].abs()
            + epsilon
        )
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(0)

    X = df[
        feature_columns
    ]

    # --------------------------------------------------------
    # Explanation sample
    # --------------------------------------------------------

    explanation_data = X.sample(
        min(
            EXPLANATION_SIZE,
            len(X)
        ),
        random_state=123
    )

    print(
        f"Explanation rows: "
        f"{len(explanation_data)}"
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CALCULATING SHAP VALUES")
    print("=" * 70)

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        explanation_data
    )

    shap_values = np.asarray(
        shap_values
    )

    print(
        f"SHAP matrix shape: "
        f"{shap_values.shape}"
    )

    # --------------------------------------------------------
    # Global importance
    # --------------------------------------------------------

    mean_abs_shap = (
        np.abs(shap_values)
        .mean(axis=0)
    )

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "mean_abs_shap": mean_abs_shap,
    })

    importance_df = (
        importance_df
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df.insert(
        0,
        "rank",
        range(
            1,
            len(importance_df) + 1
        )
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save global importance
    # --------------------------------------------------------

    importance_file = os.path.join(
        OUTPUT_DIR,
        "shap_global_importance.csv"
    )

    importance_df.to_csv(
        importance_file,
        index=False
    )

    # --------------------------------------------------------
    # Save raw SHAP values
    # --------------------------------------------------------

    shap_matrix = pd.DataFrame(
        shap_values,
        columns=feature_columns
    )

    shap_file = os.path.join(
        OUTPUT_DIR,
        "shap_values.csv"
    )

    shap_matrix.to_csv(
        shap_file,
        index=False
    )

    # --------------------------------------------------------
    # FIXED expected_value serialization
    # --------------------------------------------------------

    expected_value = (
        explainer.expected_value
    )

    if isinstance(
        expected_value,
        np.ndarray
    ):
        expected_value = (
            expected_value.tolist()
        )

    elif isinstance(
        expected_value,
        np.generic
    ):
        expected_value = (
            expected_value.item()
        )

    elif isinstance(
        expected_value,
        (float, int)
    ):
        expected_value = float(
            expected_value
        )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_file = os.path.join(
        OUTPUT_DIR,
        "shap_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "model":
                    "xgboost_v4_final_108",

                "feature_count":
                    len(feature_columns),

                "explanation_rows":
                    len(explanation_data),

                "expected_value":
                    expected_value,

                "top_n":
                    TOP_N,
            },
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print top features
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(f"TOP {TOP_N} SHAP FEATURES")
    print("=" * 70)

    print(
        importance_df
        .head(TOP_N)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Winning feature
    # --------------------------------------------------------

    winning = importance_df[
        importance_df["feature"]
        == "pm25_pollution_ratio"
    ]

    print()
    print("=" * 70)
    print("PM25 POLLUTION RATIO SHAP RESULT")
    print("=" * 70)

    if not winning.empty:

        print(
            winning.to_string(
                index=False
            )
        )

    else:

        print(
            "pm25_pollution_ratio "
            "not found."
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SHAP COMPLETE")
    print("=" * 70)

    print(
        f"Global importance:"
        f"\n{importance_file}"
    )

    print(
        f"\nSHAP values:"
        f"\n{shap_file}"
    )

    print(
        f"\nMetadata:"
        f"\n{metadata_file}"
    )


if __name__ == "__main__":
    main()