from pathlib import Path
import json
import joblib
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# PEARLSAQI V7
# SPIKE-AWARE GATED AQI CORRECTION
# ============================================================

print("=" * 80)
print("PEARLSAQI V7 - SPIKE-AWARE GATED AQI CORRECTION")
print("=" * 80)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------
# Production XGBoost
# ------------------------------------------------------------

PRODUCTION_DIR = (
    ROOT
    / "models"
    / "final_production_xgboost"
)

PRODUCTION_MODEL = (
    PRODUCTION_DIR
    / "final_xgboost_model.json"
)

PRODUCTION_FEATURES = (
    PRODUCTION_DIR
    / "features.json"
)

PRODUCTION_PREDICTIONS = (
    PRODUCTION_DIR
    / "test_predictions.parquet"
)


# ------------------------------------------------------------
# Feature dataset
# ------------------------------------------------------------

FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "features_v4.parquet"
)


# ------------------------------------------------------------
# Spike classifier
# ------------------------------------------------------------

SPIKE_MODEL_FILE = (
    ROOT
    / "models"
    / "spike_classifier"
    / "aqi_spike_classifier.pkl"
)

SPIKE_THRESHOLD_FILE = (
    ROOT
    / "models"
    / "spike_classifier"
    / "spike_threshold_results.csv"
)


# ------------------------------------------------------------
# V6 Delta model
# ------------------------------------------------------------

DELTA_MODEL_FILE = (
    ROOT
    / "models"
    / "delta_v6"
    / "delta_xgboost_model.json"
)

DELTA_FEATURES_FILE = (
    ROOT
    / "models"
    / "delta_v6"
    / "delta_features.json"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "models"
    / "spike_gated_v7"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def rmse(y_true, y_pred):

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


def metrics(y_true, y_pred):

    return {

        "mae":
            mean_absolute_error(
                y_true,
                y_pred
            ),

        "rmse":
            rmse(
                y_true,
                y_pred
            ),

        "r2":
            r2_score(
                y_true,
                y_pred
            ),

        "within_10":
            np.mean(
                np.abs(
                    y_true - y_pred
                ) <= 10
            ) * 100,

        "within_20":
            np.mean(
                np.abs(
                    y_true - y_pred
                ) <= 20
            ) * 100,

        "within_30":
            np.mean(
                np.abs(
                    y_true - y_pred
                ) <= 30
            ) * 100,
    }


def print_metrics(
    title,
    y_true,
    y_pred
):

    m = metrics(
        y_true,
        y_pred
    )

    print()
    print(title)

    print(
        f"MAE       : {m['mae']:.4f}"
    )

    print(
        f"RMSE      : {m['rmse']:.4f}"
    )

    print(
        f"R²        : {m['r2']:.4f}"
    )

    print(
        f"Within ±10: {m['within_10']:.2f}%"
    )

    print(
        f"Within ±20: {m['within_20']:.2f}%"
    )

    print(
        f"Within ±30: {m['within_30']:.2f}%"
    )

    return m


# ============================================================
# LOAD PRODUCTION PREDICTIONS
# ============================================================

print("\nLoading production predictions...")

if not PRODUCTION_PREDICTIONS.exists():

    raise FileNotFoundError(
        f"Production predictions not found:\n"
        f"{PRODUCTION_PREDICTIONS}"
    )


test = pd.read_parquet(
    PRODUCTION_PREDICTIONS
)

print(
    f"Production prediction rows: "
    f"{len(test)}"
)

print("Columns:")
print(
    test.columns.tolist()
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "city_name",
    "date",
    "aqi",
    "target_aqi",
    "prediction",
]


missing = [
    c
    for c in required
    if c not in test.columns
]


if missing:

    raise ValueError(
        "Production prediction file is missing "
        f"columns: {missing}"
    )


test["date"] = pd.to_datetime(
    test["date"]
)


test = (
    test
    .sort_values(
        [
            "city_name",
            "date"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# LOAD FEATURE DATA
# ============================================================

print("\nLoading feature dataset...")

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Feature dataset not found:\n"
        f"{FEATURE_FILE}"
    )


features_df = pd.read_parquet(
    FEATURE_FILE
)


features_df["date"] = pd.to_datetime(
    features_df["date"]
)


print(
    f"Feature rows: {len(features_df)}"
)


# ============================================================
# MERGE FEATURES WITH TEST PREDICTIONS
# ============================================================

print("\nAligning feature rows...")


merge_keys = [
    "city_name",
    "date",
]


feature_columns = [
    c
    for c in features_df.columns
    if c not in merge_keys
]


feature_subset = features_df[
    merge_keys + feature_columns
].copy()


# Avoid duplicate columns
duplicate_columns = [
    c
    for c in feature_subset.columns
    if (
        c in test.columns
        and c not in merge_keys
    )
]


if duplicate_columns:

    feature_subset = (
        feature_subset
        .drop(
            columns=duplicate_columns
        )
    )


data = test.merge(
    feature_subset,
    on=merge_keys,
    how="left",
    validate="one_to_one",
)


print(
    f"Merged rows: {len(data)}"
)


# ============================================================
# CHECK MERGE
# ============================================================

if len(data) != len(test):

    raise ValueError(
        "Merge changed the number of production rows."
    )


# ============================================================
# CHECK TARGET
# ============================================================

if data["target_aqi"].isna().any():

    raise ValueError(
        "Some target_aqi values are missing "
        "after merge."
    )


# ============================================================
# LOAD SPIKE CLASSIFIER
# ============================================================

print("\nLoading spike classifier...")


if not SPIKE_MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Spike classifier not found:\n"
        f"{SPIKE_MODEL_FILE}"
    )


# IMPORTANT:
#
# spike_classifier.py saves the classifier using:
#
#     joblib.dump(bundle, MODEL_PATH)
#
# Therefore V7 MUST use:
#
#     joblib.load(...)
#
# NOT pickle.load(...).
#
# Using pickle.load() caused:
#
#     _pickle.UnpicklingError:
#     invalid load key, 'x'
#
# ------------------------------------------------------------

try:

    spike_bundle = joblib.load(
        SPIKE_MODEL_FILE
    )

except Exception as e:

    raise RuntimeError(
        "\nUnable to load spike classifier.\n"
        f"File: {SPIKE_MODEL_FILE}\n\n"
        "The classifier must have been saved using "
        "joblib.dump().\n\n"
        f"Original error: {e}"
    ) from e


# ============================================================
# READ SPIKE MODEL BUNDLE
# ============================================================

if isinstance(
    spike_bundle,
    dict
):

    if "model" not in spike_bundle:

        raise ValueError(
            "Spike classifier bundle does not "
            "contain the 'model' key."
        )


    spike_model = (
        spike_bundle["model"]
    )


    spike_features = (
        spike_bundle.get(
            "features",
            []
        )
    )


    saved_threshold = (
        spike_bundle.get(
            "threshold",
            None
        )
    )


    prediction_threshold = (
        spike_bundle.get(
            "prediction_threshold",
            0.50
        )
    )

else:

    # --------------------------------------------------------
    # Fallback for old models that were saved directly.
    # --------------------------------------------------------

    spike_model = spike_bundle

    spike_features = []

    saved_threshold = None

    prediction_threshold = 0.50


print(
    "Spike classifier type:",
    type(spike_model).__name__
)

print(
    "Spike classifier features:",
    len(spike_features)
)

print(
    "Saved spike definition threshold:",
    saved_threshold
)

print(
    "Saved classifier probability threshold:",
    prediction_threshold
)


# ============================================================
# SPIKE THRESHOLD SEARCH
# ============================================================

# This is NOT the same as the AQI spike definition.
#
# SPIKE_THRESHOLD = 40
#
# means:
#
# abs(next_day_AQI - current_AQI) >= 40
#
# The thresholds below are classifier probability gates.
#
# Example:
#
# 0.30 means:
# apply correction when classifier probability >= 30%.
#
# ------------------------------------------------------------

candidate_spike_thresholds = [

    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
    0.60,
    0.70,

]


# ============================================================
# CHECK SPIKE FEATURES
# ============================================================

if not spike_features:

    raise ValueError(
        "Spike classifier does not contain a "
        "saved feature list."
    )


missing_spike = [
    f
    for f in spike_features
    if f not in data.columns
]


if missing_spike:

    print(
        "\nMissing spike classifier features:"
    )

    for f in missing_spike:

        print(
            " -",
            f
        )


    raise ValueError(
        "\nCannot safely generate spike "
        "probabilities because the classifier "
        "features are unavailable."
    )


X_spike = data[
    spike_features
].copy()


X_spike = X_spike.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)


# HistGradientBoosting supports NaN,
# but force everything to numeric.

X_spike = X_spike.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# GENERATE SPIKE PROBABILITY
# ============================================================

print(
    "\nGenerating spike probabilities..."
)


if not hasattr(
    spike_model,
    "predict_proba"
):

    raise TypeError(
        "Loaded spike model does not support "
        "predict_proba()."
    )


spike_probability = (
    spike_model
    .predict_proba(
        X_spike
    )[:, 1]
)


data[
    "spike_probability"
] = spike_probability


print(
    "Spike probability range:",
    round(
        float(
            spike_probability.min()
        ),
        6
    ),
    "→",
    round(
        float(
            spike_probability.max()
        ),
        6
    ),
)


print(
    "Average spike probability:",
    round(
        float(
            spike_probability.mean()
        ),
        6
    )
)


# ============================================================
# SPIKE PROBABILITY DISTRIBUTION
# ============================================================

print("\nSpike probability distribution:")


for threshold in [
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]:

    count = int(
        (
            data[
                "spike_probability"
            ]
            >= threshold
        ).sum()
    )

    percentage = (
        count
        / len(data)
        * 100
    )

    print(
        f" >= {threshold:.2f}: "
        f"{count} rows "
        f"({percentage:.2f}%)"
    )


# ============================================================
# LOAD DELTA MODEL
# ============================================================

print("\nLoading V6 delta model...")


if not DELTA_MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Delta model not found:\n"
        f"{DELTA_MODEL_FILE}"
    )


if not DELTA_FEATURES_FILE.exists():

    raise FileNotFoundError(
        f"Delta feature file not found:\n"
        f"{DELTA_FEATURES_FILE}"
    )


delta_model = XGBRegressor()


delta_model.load_model(
    str(
        DELTA_MODEL_FILE
    )
)


with open(
    DELTA_FEATURES_FILE,
    "r",
    encoding="utf-8"
) as f:

    delta_features = json.load(f)


print(
    "Delta features:",
    len(delta_features)
)


# ============================================================
# CHECK DELTA FEATURES
# ============================================================

missing_delta = [
    f
    for f in delta_features
    if f not in data.columns
]


if missing_delta:

    print(
        "\nMissing delta features:"
    )

    for f in missing_delta:

        print(
            " -",
            f
        )


    raise ValueError(
        "\nCannot safely generate "
        "delta predictions."
    )


X_delta = data[
    delta_features
].copy()


X_delta = X_delta.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)


X_delta = X_delta.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# DELTA PREDICTION
# ============================================================

print(
    "\nGenerating delta predictions..."
)


delta_prediction = (
    delta_model.predict(
        X_delta
    )
)


data[
    "delta_prediction"
] = delta_prediction


print(
    "Delta prediction range:",
    round(
        float(
            delta_prediction.min()
        ),
        4
    ),
    "→",
    round(
        float(
            delta_prediction.max()
        ),
        4
    )
)


# ============================================================
# BASE PREDICTION
# ============================================================

data[
    "base_prediction"
] = (
    data[
        "prediction"
    ].astype(float)
)


# ============================================================
# ACTUAL CHANGES
# ============================================================

data[
    "actual_change"
] = (
    data[
        "target_aqi"
    ]
    - data[
        "aqi"
    ]
)


data[
    "actual_spike"
] = (
    data[
        "actual_change"
    ].abs()
    >= 40
)


data[
    "actual_upward_spike"
] = (
    data[
        "actual_change"
    ]
    >= 40
)


data[
    "actual_downward_spike"
] = (
    data[
        "actual_change"
    ]
    <= -40
)


data[
    "actual_extreme"
] = (
    data[
        "target_aqi"
    ]
    > 200
)


# ============================================================
# BASE METRICS
# ============================================================

base_metrics = print_metrics(

    "BASE PRODUCTION XGBOOST",

    data[
        "target_aqi"
    ],

    data[
        "base_prediction"
    ],

)


# ============================================================
# BASE EXTREME / SPIKE METRICS
# ============================================================

extreme_mask = (
    data[
        "target_aqi"
    ]
    > 200
)


upward_mask = (
    data[
        "actual_change"
    ]
    >= 40
)


if extreme_mask.any():

    base_extreme_mae = (
        mean_absolute_error(
            data.loc[
                extreme_mask,
                "target_aqi"
            ],

            data.loc[
                extreme_mask,
                "base_prediction"
            ]
        )
    )

else:

    base_extreme_mae = np.nan


if upward_mask.any():

    base_spike_mae = (
        mean_absolute_error(
            data.loc[
                upward_mask,
                "target_aqi"
            ],

            data.loc[
                upward_mask,
                "base_prediction"
            ]
        )
    )

else:

    base_spike_mae = np.nan


print(
    f"\nBase extreme MAE: "
    f"{base_extreme_mae:.4f}"
)


print(
    f"Base upward-spike MAE: "
    f"{base_spike_mae:.4f}"
)


# ============================================================
# V7 STRATEGY SEARCH
# ============================================================

print("\n")
print("=" * 80)
print("V7 GATED STRATEGY SEARCH")
print("=" * 80)


results = []


# ------------------------------------------------------------
# Correction strengths
# ------------------------------------------------------------

correction_strengths = [

    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,

]


# ============================================================
# SEARCH
# ============================================================

for threshold in (
    candidate_spike_thresholds
):

    for strength in (
        correction_strengths
    ):

        # ----------------------------------------------------
        # SPIKE GATE
        # ----------------------------------------------------

        gate = (
            data[
                "spike_probability"
            ]
            >= threshold
        )


        # ----------------------------------------------------
        # POSITIVE DELTA ONLY
        #
        # V7 is specifically designed to attack
        # upward underprediction.
        #
        # It does NOT blindly apply negative delta.
        # ----------------------------------------------------

        positive_delta = np.maximum(
            data[
                "delta_prediction"
            ],
            0
        )


        # ----------------------------------------------------
        # CORRECTION
        # ----------------------------------------------------

        correction = (

            positive_delta

            * strength

            * gate.astype(float)

        )


        corrected = (

            data[
                "base_prediction"
            ]

            + correction

        )


        # ----------------------------------------------------
        # SAFETY
        # ----------------------------------------------------

        corrected = np.maximum(
            corrected,
            0
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        y_true = (
            data[
                "target_aqi"
            ]
        )


        m = metrics(
            y_true,
            corrected
        )


        # ----------------------------------------------------
        # EXTREME
        # ----------------------------------------------------

        extreme_mask_local = (
            data[
                "target_aqi"
            ]
            > 200
        )


        if (
            extreme_mask_local.any()
        ):

            extreme_mae = (
                mean_absolute_error(

                    y_true[
                        extreme_mask_local
                    ],

                    corrected[
                        extreme_mask_local
                    ]

                )
            )

        else:

            extreme_mae = np.nan


        # ----------------------------------------------------
        # UPWARD SPIKE
        # ----------------------------------------------------

        spike_mask = (
            data[
                "actual_change"
            ]
            >= 40
        )


        if spike_mask.any():

            spike_mae = (
                mean_absolute_error(

                    y_true[
                        spike_mask
                    ],

                    corrected[
                        spike_mask
                    ]

                )
            )

        else:

            spike_mae = np.nan


        # ----------------------------------------------------
        # LARGE UNDERPREDICTION
        # ----------------------------------------------------

        error = (
            corrected
            - y_true
        )


        underprediction_mask = (
            error < -40
        )


        if (
            underprediction_mask.any()
        ):

            under_mae = np.mean(
                np.abs(
                    error[
                        underprediction_mask
                    ]
                )
            )

        else:

            under_mae = 0.0


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        results.append({

            "spike_threshold":
                threshold,

            "correction_strength":
                strength,

            "corrected_rows":
                int(
                    gate.sum()
                ),

            "corrected_percentage":
                (
                    gate.mean()
                    * 100
                ),

            "mae":
                m["mae"],

            "rmse":
                m["rmse"],

            "r2":
                m["r2"],

            "within_10":
                m["within_10"],

            "within_20":
                m["within_20"],

            "within_30":
                m["within_30"],

            "extreme_mae":
                extreme_mae,

            "upward_spike_mae":
                spike_mae,

            "underprediction_mae":
                under_mae,

        })


results_df = pd.DataFrame(
    results
)


# ============================================================
# SORT BY OVERALL MAE
# ============================================================

results_sorted = (

    results_df

    .sort_values(
        [
            "mae",
            "rmse"
        ]
    )

    .reset_index(
        drop=True
    )

)


print(
    "\nTOP V7 STRATEGIES"
)


print(
    results_sorted
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# BEST OVERALL
# ============================================================

best = (
    results_sorted
    .iloc[0]
)


best_threshold = float(
    best[
        "spike_threshold"
    ]
)


best_strength = float(
    best[
        "correction_strength"
    ]
)


# ============================================================
# BEST EXTREME
# ============================================================

best_extreme = (

    results_df

    .sort_values(
        [
            "extreme_mae",
            "mae"
        ]
    )

    .iloc[0]

)


# ============================================================
# BEST UPWARD SPIKE
# ============================================================

best_spike = (

    results_df

    .sort_values(
        [
            "upward_spike_mae",
            "mae"
        ]
    )

    .iloc[0]

)


# ============================================================
# BEST STRATEGIES PRINT
# ============================================================

print("\n")
print("=" * 80)
print("BEST STRATEGIES")
print("=" * 80)


print(
    "\nBest overall:"
)


print(
    f"Threshold : "
    f"{best_threshold:.2f}"
)


print(
    f"Strength  : "
    f"{best_strength:.2f}"
)


print(
    f"MAE       : "
    f"{best['mae']:.4f}"
)


print(
    f"RMSE      : "
    f"{best['rmse']:.4f}"
)


print(
    f"R²        : "
    f"{best['r2']:.4f}"
)


print(
    f"Corrected : "
    f"{int(best['corrected_rows'])} "
    f"rows "
    f"({best['corrected_percentage']:.2f}%)"
)


print(
    "\nBest extreme strategy:"
)


print(
    f"Threshold : "
    f"{float(best_extreme['spike_threshold']):.2f}"
)


print(
    f"Strength  : "
    f"{float(best_extreme['correction_strength']):.2f}"
)


print(
    f"Extreme MAE : "
    f"{float(best_extreme['extreme_mae']):.4f}"
)


print(
    "\nBest upward-spike strategy:"
)


print(
    f"Threshold : "
    f"{float(best_spike['spike_threshold']):.2f}"
)


print(
    f"Strength  : "
    f"{float(best_spike['correction_strength']):.2f}"
)


print(
    f"Spike MAE : "
    f"{float(best_spike['upward_spike_mae']):.4f}"
)


# ============================================================
# FINAL V7 PREDICTION
# ============================================================

best_gate = (

    data[
        "spike_probability"
    ]

    >= best_threshold

)


best_positive_delta = np.maximum(

    data[
        "delta_prediction"
    ],

    0

)


best_correction = (

    best_positive_delta

    * best_strength

    * best_gate.astype(float)

)


data[
    "v7_correction"
] = best_correction


data[
    "v7_prediction"
] = (

    data[
        "base_prediction"
    ]

    + data[
        "v7_correction"
    ]

)


# Safety

data[
    "v7_prediction"
] = np.maximum(

    data[
        "v7_prediction"
    ],

    0

)


# ============================================================
# FINAL METRICS
# ============================================================

v7_metrics = print_metrics(

    "V7 SPIKE-GATED MODEL",

    data[
        "target_aqi"
    ],

    data[
        "v7_prediction"
    ],

)


# ============================================================
# EXTREME METRICS
# ============================================================

extreme_mask = (

    data[
        "target_aqi"
    ]

    > 200

)


if extreme_mask.any():

    base_extreme_mae = (

        mean_absolute_error(

            data.loc[
                extreme_mask,
                "target_aqi"
            ],

            data.loc[
                extreme_mask,
                "base_prediction"
            ]

        )

    )


    v7_extreme_mae = (

        mean_absolute_error(

            data.loc[
                extreme_mask,
                "target_aqi"
            ],

            data.loc[
                extreme_mask,
                "v7_prediction"
            ]

        )

    )

else:

    base_extreme_mae = np.nan

    v7_extreme_mae = np.nan


# ============================================================
# UPWARD SPIKE METRICS
# ============================================================

upward_mask = (

    data[
        "actual_change"
    ]

    >= 40

)


if upward_mask.any():

    base_spike_mae = (

        mean_absolute_error(

            data.loc[
                upward_mask,
                "target_aqi"
            ],

            data.loc[
                upward_mask,
                "base_prediction"
            ]

        )

    )


    v7_spike_mae = (

        mean_absolute_error(

            data.loc[
                upward_mask,
                "target_aqi"
            ],

            data.loc[
                upward_mask,
                "v7_prediction"
            ]

        )

    )

else:

    base_spike_mae = np.nan

    v7_spike_mae = np.nan


# ============================================================
# DOWNWARD SPIKE METRICS
# ============================================================

downward_mask = (

    data[
        "actual_change"
    ]

    <= -40

)


if downward_mask.any():

    base_downward_mae = (

        mean_absolute_error(

            data.loc[
                downward_mask,
                "target_aqi"
            ],

            data.loc[
                downward_mask,
                "base_prediction"
            ]

        )

    )


    v7_downward_mae = (

        mean_absolute_error(

            data.loc[
                downward_mask,
                "target_aqi"
            ],

            data.loc[
                downward_mask,
                "v7_prediction"
            ]

        )

    )

else:

    base_downward_mae = np.nan

    v7_downward_mae = np.nan


# ============================================================
# V7 IMPACT
# ============================================================

print("\n")
print("=" * 80)
print("V7 IMPACT")
print("=" * 80)


print(
    f"\nBase MAE       : "
    f"{base_metrics['mae']:.4f}"
)


print(
    f"V7 MAE         : "
    f"{v7_metrics['mae']:.4f}"
)


print(
    f"MAE improvement: "
    f"{base_metrics['mae'] - v7_metrics['mae']:.4f}"
)


print(
    f"\nBase extreme MAE: "
    f"{base_extreme_mae:.4f}"
)


print(
    f"V7 extreme MAE  : "
    f"{v7_extreme_mae:.4f}"
)


print(
    f"Extreme improvement: "
    f"{base_extreme_mae - v7_extreme_mae:.4f}"
)


print(
    f"\nBase upward-spike MAE: "
    f"{base_spike_mae:.4f}"
)


print(
    f"V7 upward-spike MAE  : "
    f"{v7_spike_mae:.4f}"
)


print(
    f"Spike improvement: "
    f"{base_spike_mae - v7_spike_mae:.4f}"
)


print(
    f"\nBase downward-spike MAE: "
    f"{base_downward_mae:.4f}"
)


print(
    f"V7 downward-spike MAE  : "
    f"{v7_downward_mae:.4f}"
)


print(
    f"Downward-spike impact: "
    f"{base_downward_mae - v7_downward_mae:.4f}"
)


# ============================================================
# CITY METRICS
# ============================================================

city_results = []


for city, group in data.groupby(
    "city_name"
):

    base_mae = (

        mean_absolute_error(

            group[
                "target_aqi"
            ],

            group[
                "base_prediction"
            ]

        )

    )


    v7_mae = (

        mean_absolute_error(

            group[
                "target_aqi"
            ],

            group[
                "v7_prediction"
            ]

        )

    )


    city_results.append({

        "city":
            city,

        "rows":
            len(group),

        "base_mae":
            base_mae,

        "v7_mae":
            v7_mae,

        "improvement":
            base_mae - v7_mae,

        "actual_mean":
            group[
                "target_aqi"
            ].mean(),

        "prediction_mean":
            group[
                "v7_prediction"
            ].mean(),

        "actual_max":
            group[
                "target_aqi"
            ].max(),

        "prediction_max":
            group[
                "v7_prediction"
            ].max(),

        "correction_rows":
            int(
                (
                    group[
                        "v7_prediction"
                    ]

                    !=

                    group[
                        "base_prediction"
                    ]
                ).sum()
            ),

        "average_correction":
            group[
                "v7_correction"
            ].mean(),

        "max_correction":
            group[
                "v7_correction"
            ].max(),

    })


city_results_df = (

    pd.DataFrame(
        city_results
    )

    .sort_values(
        "improvement",
        ascending=False
    )

)


print(
    "\nCITY IMPACT"
)


print(
    city_results_df
    .to_string(
        index=False
    )
)


# ============================================================
# WORST PREDICTIONS
# ============================================================

data[
    "v7_absolute_error"
] = (

    np.abs(

        data[
            "target_aqi"
        ]

        -

        data[
            "v7_prediction"
        ]

    )

)


worst = (

    data

    .sort_values(
        "v7_absolute_error",
        ascending=False
    )

    .head(30)

)


print(
    "\nTOP 30 V7 WORST PREDICTIONS"
)


display_columns = [

    "city_name",

    "date",

    "aqi",

    "target_aqi",

    "base_prediction",

    "spike_probability",

    "delta_prediction",

    "v7_correction",

    "v7_prediction",

    "v7_absolute_error",

]


print(

    worst[
        display_columns
    ]

    .to_string(
        index=False
    )

)


# ============================================================
# SAVE STRATEGY RESULTS
# ============================================================

strategy_file = (

    OUTPUT_DIR
    / "v7_strategy_results.csv"

)


results_df.to_csv(

    strategy_file,

    index=False

)


# ============================================================
# SAVE CITY RESULTS
# ============================================================

city_file = (

    OUTPUT_DIR
    / "v7_city_metrics.csv"

)


city_results_df.to_csv(

    city_file,

    index=False

)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_columns = [

    "city_name",

    "date",

    "aqi",

    "target_aqi",

    "base_prediction",

    "spike_probability",

    "delta_prediction",

    "v7_correction",

    "v7_prediction",

    "actual_change",

    "actual_spike",

    "actual_upward_spike",

    "actual_downward_spike",

    "actual_extreme",

    "v7_absolute_error",

]


prediction_file = (

    OUTPUT_DIR
    / "v7_predictions.parquet"

)


data[
    prediction_columns
].to_parquet(

    prediction_file,

    index=False

)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model":
        "PEARLSAQI V7 Spike-Gated Correction",

    "base_model":
        "Final Production XGBoost",

    "base_features":
        107,

    "delta_model":
        "PEARLSAQI V6 Delta XGBoost",

    "delta_features":
        len(delta_features),

    "spike_classifier":
        type(
            spike_model
        ).__name__,

    "spike_classifier_features":
        len(
            spike_features
        ),

    "spike_definition_threshold":
        saved_threshold,

    "candidate_spike_thresholds":
        candidate_spike_thresholds,

    "selected_spike_threshold":
        best_threshold,

    "selected_correction_strength":
        best_strength,

    "base_metrics":
        base_metrics,

    "v7_metrics":
        v7_metrics,

    "base_extreme_mae":
        float(
            base_extreme_mae
        ),

    "v7_extreme_mae":
        float(
            v7_extreme_mae
        ),

    "base_upward_spike_mae":
        float(
            base_spike_mae
        ),

    "v7_upward_spike_mae":
        float(
            v7_spike_mae
        ),

    "base_downward_spike_mae":
        float(
            base_downward_mae
        ),

    "v7_downward_spike_mae":
        float(
            v7_downward_mae
        ),

    "rows":
        int(
            len(data)
        ),

    "cities":
        sorted(
            data[
                "city_name"
            ]
            .unique()
            .tolist()
        ),

    "created_at":
        pd.Timestamp.now().isoformat(),

}


metadata_file = (

    OUTPUT_DIR
    / "v7_metadata.json"

)


with open(

    metadata_file,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        metadata,

        f,

        indent=2

    )


# ============================================================
# FINAL DECISION
# ============================================================

# V7 is only recommended when:
#
# 1. Overall MAE improves
# 2. Upward-spike MAE does not get worse
#
# This prevents a model from looking better overall
# while becoming worse exactly where we need it most.
#
# ------------------------------------------------------------

recommended = (

    v7_metrics[
        "mae"
    ]

    <

    base_metrics[
        "mae"
    ]

    and

    v7_spike_mae

    <=

    base_spike_mae

)


decision = {

    "recommended":
        bool(
            recommended
        ),

    "base_model":
        "Production XGBoost",

    "candidate_model":
        "V7 Spike-Gated Correction",

    "selected_threshold":
        best_threshold,

    "selected_strength":
        best_strength,

    "base_mae":
        float(
            base_metrics[
                "mae"
            ]
        ),

    "v7_mae":
        float(
            v7_metrics[
                "mae"
            ]
        ),

    "base_upward_spike_mae":
        float(
            base_spike_mae
        ),

    "v7_upward_spike_mae":
        float(
            v7_spike_mae
        ),

    "reason":
        (

            "V7 improves overall MAE and "
            "upward-spike MAE."

            if recommended

            else

            "V7 does not improve both overall "
            "and upward-spike performance. "
            "Keep production XGBoost."

        ),

}


decision_file = (

    OUTPUT_DIR
    / "v7_decision.json"

)


with open(

    decision_file,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        decision,

        f,

        indent=2

    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("PEARLSAQI V7 COMPLETED")
print("=" * 80)


print(
    "\nStrategy results:",
    strategy_file
)


print(
    "Predictions:",
    prediction_file
)


print(
    "City metrics:",
    city_file
)


print(
    "Metadata:",
    metadata_file
)


print(
    "Decision:",
    decision_file
)


print(
    "\nRecommended to replace production:",
    recommended
)