import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer

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

OUTPUT_DIR = "models/benchmark/v4/lime"

EXPLANATION_COUNT = 5

TOP_FEATURES = 15


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_data(df):

    df = df.copy()

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

    return df
def explain_instance(model, X_row, feature_columns, top_n=5):
    """
    Generate a LIME explanation for one live prediction row
    using the real V4 training dataset as LIME background data.
    """

    train_df = pd.read_parquet(TRAIN_FILE)
    train_df = prepare_data(train_df)

    X_train = train_df[feature_columns]

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_columns,
        mode="regression",
        discretize_continuous=True,
        random_state=42,
        verbose=False,
    )

    explanation = explainer.explain_instance(
        X_row.iloc[0].values,
        model.predict,
        num_features=top_n,
    )

    return [
        {
            "feature": feature,
            "contribution": float(contribution),
        }
        for feature, contribution in explanation.as_list()
    ]

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLSAQI LIME EXPLAINABILITY")
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
            "Expected 108 features."
        )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print()
    print("Loading training data...")

    df = pd.read_parquet(
        TRAIN_FILE
    )

    df = prepare_data(
        df
    )

    X = df[
        feature_columns
    ]

    # --------------------------------------------------------
    # LIME explainer
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING LIME EXPLAINER")
    print("=" * 70)

    explainer = LimeTabularExplainer(

        training_data=X.values,

        feature_names=feature_columns,

        mode="regression",

        discretize_continuous=True,

        random_state=42,

        verbose=False
    )

    print(
        "LIME explainer created."
    )

    # --------------------------------------------------------
    # Select deterministic examples
    # --------------------------------------------------------

    explanation_data = X.sample(
        min(
            EXPLANATION_COUNT,
            len(X)
        ),
        random_state=123
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    all_results = []

    # --------------------------------------------------------
    # Generate explanations
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING LIME EXPLANATIONS")
    print("=" * 70)

    for counter, (
        index,
        row
    ) in enumerate(
        explanation_data.iterrows(),
        start=1
    ):

        city = df.loc[
            index,
            "city_name"
        ]

        date_value = df.loc[
            index,
            "date"
        ]

        prediction = model.predict(
            row.to_frame().T
        )[0]

        print()
        print(
            f"Example {counter}: "
            f"{city} | {date_value}"
        )

        # ----------------------------------------------------
        # LIME explanation
        # ----------------------------------------------------

        explanation = (
            explainer.explain_instance(

                row.values,

                model.predict,

                num_features=TOP_FEATURES
            )
        )

        local_explanation = (
            explanation.as_list()
        )

        # ----------------------------------------------------
        # Save individual JSON
        # ----------------------------------------------------

        features = []

        for rank, (
            feature,
            contribution
        ) in enumerate(
            local_explanation,
            start=1
        ):

            features.append({

                "rank":
                    rank,

                "feature":
                    feature,

                "contribution":
                    float(contribution)

            })

        result = {

            "example":
                counter,

            "city":
                str(city),

            "date":
                str(date_value),

            "prediction":
                float(prediction),

            "features":
                features

        }

        all_results.append(
            result
        )

        safe_city = (
            str(city)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{counter:02d}_{safe_city}_lime.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2
            )

        # ----------------------------------------------------
        # Print explanation
        # ----------------------------------------------------

        print(
            f"Prediction: "
            f"{prediction:.2f}"
        )

        print()
        print(
            "Top LIME contributions:"
        )

        for rank, (
            feature,
            contribution
        ) in enumerate(
            local_explanation[:10],
            start=1
        ):

            direction = (
                "INCREASES"
                if contribution > 0
                else "DECREASES"
            )

            print(
                f"{rank:02d}. "
                f"{feature:<45} "
                f"{contribution:+.4f} "
                f"({direction})"
            )

    # --------------------------------------------------------
    # Combined API JSON
    # --------------------------------------------------------

    combined_file = os.path.join(
        OUTPUT_DIR,
        "lime_summary.json"
    )

    with open(
        combined_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "model":
                    "xgboost_v4_final_108",

                "feature_count":
                    108,

                "explanation_count":
                    len(all_results),

                "explanations":
                    all_results

            },
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LIME COMPLETE")
    print("=" * 70)

    print(
        f"Individual explanations: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Combined API file: "
        f"{combined_file}"
    )


if __name__ == "__main__":
    main()