import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
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

EXPLANATION_ROWS = 500

WATERFALL_COUNT = 5


# ============================================================
# HELPERS
# ============================================================

def add_winning_feature(df):

    epsilon = 1e-6

    df = df.copy()

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PEARLSAQI SHAP VISUALIZATION + EXPLANATION")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

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
    # Load data
    # --------------------------------------------------------

    print()
    print("Loading training data...")

    df = pd.read_parquet(
        TRAIN_FILE
    )

    df = add_winning_feature(
        df
    )

    X = df[
        feature_columns
    ]

    # --------------------------------------------------------
    # Explanation sample
    # --------------------------------------------------------

    explanation_data = X.sample(
        min(
            EXPLANATION_ROWS,
            len(X)
        ),
        random_state=123
    )

    # --------------------------------------------------------
    # SHAP explainer
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
        f"SHAP shape: {shap_values.shape}"
    )

    # --------------------------------------------------------
    # SHAP Explanation object
    # --------------------------------------------------------

    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=np.repeat(
            float(explainer.expected_value),
            len(explanation_data)
        ),
        data=explanation_data.values,
        feature_names=feature_columns
    )

    # ========================================================
    # 1. BAR PLOT
    # ========================================================

    print()
    print("=" * 70)
    print("GENERATING SHAP BAR PLOT")
    print("=" * 70)

    plt.figure(
        figsize=(10, 8)
    )

    shap.summary_plot(
        shap_values,
        explanation_data,
        plot_type="bar",
        max_display=TOP_N,
        show=False
    )

    plt.title(
        "PEARLSAQI — SHAP Global Feature Importance"
    )

    plt.tight_layout()

    bar_file = os.path.join(
        OUTPUT_DIR,
        "shap_bar_plot.png"
    )

    plt.savefig(
        bar_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {bar_file}"
    )

    # ========================================================
    # 2. BEESWARM
    # ========================================================

    print()
    print("=" * 70)
    print("GENERATING SHAP BEESWARM")
    print("=" * 70)

    plt.figure(
        figsize=(11, 9)
    )

    shap.summary_plot(
        shap_values,
        explanation_data,
        max_display=TOP_N,
        show=False
    )

    plt.title(
        "PEARLSAQI — SHAP Feature Impact Distribution"
    )

    plt.tight_layout()

    beeswarm_file = os.path.join(
        OUTPUT_DIR,
        "shap_beeswarm_plot.png"
    )

    plt.savefig(
        beeswarm_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {beeswarm_file}"
    )

    # ========================================================
    # 3. PER-CITY SHAP
    # ========================================================

    print()
    print("=" * 70)
    print("GENERATING PER-CITY SHAP SUMMARIES")
    print("=" * 70)

    # Match sampled explanation rows back to original data
    explanation_indices = explanation_data.index

    city_values = df.loc[
        explanation_indices,
        "city_name"
    ].values

    city_shap_df = pd.DataFrame(
        np.abs(shap_values),
        columns=feature_columns,
        index=explanation_indices
    )

    city_shap_df["city_name"] = city_values

    city_results = []

    for city, group in city_shap_df.groupby(
        "city_name"
    ):

        mean_importance = (
            group[
                feature_columns
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        top_features = (
            mean_importance
            .head(10)
        )

        for rank, (
            feature,
            importance
        ) in enumerate(
            top_features.items(),
            start=1
        ):

            city_results.append({

                "city_name":
                    city,

                "rank":
                    rank,

                "feature":
                    feature,

                "mean_abs_shap":
                    float(importance),

            })

    city_results_df = pd.DataFrame(
        city_results
    )

    city_file = os.path.join(
        OUTPUT_DIR,
        "shap_per_city.csv"
    )

    city_results_df.to_csv(
        city_file,
        index=False
    )

    print(
        f"Cities analyzed: "
        f"{city_shap_df['city_name'].nunique()}"
    )

    print(
        f"Saved: {city_file}"
    )

    # ========================================================
    # 4. INDIVIDUAL WATERFALLS
    # ========================================================

    print()
    print("=" * 70)
    print("GENERATING INDIVIDUAL WATERFALL EXPLANATIONS")
    print("=" * 70)

    waterfall_dir = os.path.join(
        OUTPUT_DIR,
        "waterfalls"
    )

    os.makedirs(
        waterfall_dir,
        exist_ok=True
    )

    # Pick deterministic examples
    waterfall_indices = list(
        range(
            min(
                WATERFALL_COUNT,
                len(explanation_data)
            )
        )
    )

    waterfall_metadata = []

    for i in waterfall_indices:

        row = explanation_data.iloc[
            i
        ]

        row_index = explanation_data.index[
            i
        ]

        city = df.loc[
            row_index,
            "city_name"
        ]

        date_value = df.loc[
            row_index,
            "date"
        ]

        # Prediction
        prediction = model.predict(
            row.to_frame().T
        )[0]

        # Single-row SHAP explanation
        single_explanation = shap.Explanation(
            values=shap_values[i],
            base_values=float(
                explainer.expected_value
            ),
            data=row.values,
            feature_names=feature_columns
        )

        plt.figure(
            figsize=(10, 8)
        )

        shap.plots.waterfall(
            single_explanation,
            max_display=15,
            show=False
        )

        plt.tight_layout()

        safe_city = (
            str(city)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        filename = (
            f"{i+1:02d}_"
            f"{safe_city}_"
            f"waterfall.png"
        )

        filepath = os.path.join(
            waterfall_dir,
            filename
        )

        plt.savefig(
            filepath,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        waterfall_metadata.append({

            "example":
                i + 1,

            "city":
                str(city),

            "date":
                str(date_value),

            "prediction":
                float(prediction),

            "waterfall_file":
                filepath

        })

        print(
            f"Generated: {filename}"
        )

    # ========================================================
    # 5. API-READY TOP FEATURES JSON
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING API-READY SHAP JSON")
    print("=" * 70)

    global_importance = (
        pd.DataFrame({

            "feature":
                feature_columns,

            "mean_abs_shap":
                np.abs(shap_values)
                .mean(axis=0)

        })
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
    )

    api_features = []

    for rank, row in enumerate(
        global_importance.head(TOP_N).itertuples(
            index=False
        ),
        start=1
    ):

        api_features.append({

            "rank":
                rank,

            "feature":
                row.feature,

            "mean_abs_shap":
                float(
                    row.mean_abs_shap
                )

        })

    api_file = os.path.join(
        OUTPUT_DIR,
        "shap_summary.json"
    )

    with open(
        api_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "model":
                    "xgboost_v4_final_108",

                "feature_count":
                    108,

                "top_features":
                    api_features,

                "waterfall_examples":
                    waterfall_metadata

            },
            f,
            indent=2
        )

    print(
        f"Saved: {api_file}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("SHAP VISUALIZATION COMPLETE")
    print("=" * 70)

    print()
    print("FILES CREATED:")

    print(
        f"1. {bar_file}"
    )

    print(
        f"2. {beeswarm_file}"
    )

    print(
        f"3. {city_file}"
    )

    print(
        f"4. {waterfall_dir}"
    )

    print(
        f"5. {api_file}"
    )

    print()
    print("SHAP IS NOW FULLY COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    main()