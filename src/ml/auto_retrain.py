import os
import time
import joblib
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models.model_registry import (
    get_best_model,
    record_result,
    promote_model,
)


# ============================================================
# PEARLSAQI AUTOMATIC MODEL RETRAINING
# ============================================================
#
# SAFE AUTOMATIC RETRAINING PIPELINE
#
# 1. Uses the EXACT V4 training dataset.
# 2. Uses the EXACT V4 feature selection.
# 3. Uses the EXACT V4 XGBoost configuration.
# 4. Evaluates production and candidate on the SAME
#    validation dataset.
# 5. NEVER overwrites production directly.
# 6. Candidate is registered ONLY if it passes the gate.
# 7. Accepted candidate is automatically promoted.
# 8. Old production version is automatically archived.
# 9. Exactly ONE version remains production.
# 10. Promotion is verified before returning success.
#
# ============================================================


TRAIN_FILE = "data/processed/v4/train.parquet"

VALIDATION_FILE = (
    "data/processed/v4/validation.parquet"
)

CANDIDATE_DIR = (
    "models/candidates"
)

CANDIDATE_FILE = os.path.join(
    CANDIDATE_DIR,
    "xgboost_h1_candidate.pkl",
)


# ============================================================
# PROMOTION CONFIGURATION
# ============================================================

# Candidate must improve production RMSE
# by at least this percentage.

MIN_RMSE_IMPROVEMENT_PERCENT = 0.25


# ============================================================
# METRICS
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name,
):

    predictions = model.predict(
        X
    )

    mae = mean_absolute_error(
        y,
        predictions,
    )

    rmse = mean_squared_error(
        y,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y,
        predictions,
    )

    print()
    print("=" * 70)
    print(dataset_name)
    print("=" * 70)

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R2   : {r2:.4f}"
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================

def load_production_model():

    print()
    print("=" * 70)
    print("CHECKING CURRENT PRODUCTION MODEL")
    print("=" * 70)

    production = get_best_model(
        horizon=1
    )

    if production is None:

        raise RuntimeError(
            "No production model is registered "
            "for horizon 1."
        )

    model_path = os.path.abspath(
        production["model_path"]
    )

    print(
        f"Production model : "
        f"{production['name']}"
    )

    print(
        f"Production version: "
        f"v{production['version']}"
    )

    print(
        f"Registry RMSE    : "
        f"{production['rmse']:.4f}"
    )

    print(
        f"Registry MAE     : "
        f"{production['mae']:.4f}"
    )

    print(
        f"Registry R2      : "
        f"{production['r2']:.4f}"
    )

    print(
        f"Production path  : "
        f"{model_path}"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            "Production model not found:\n"
            f"{model_path}"
        )

    # --------------------------------------------------------
    # Load model according to file format
    # --------------------------------------------------------

    suffix = (
        os.path.splitext(
            model_path
        )[1]
        .lower()
    )

    if suffix in (
        ".pkl",
        ".pickle",
        ".joblib",
    ):

        model = joblib.load(
            model_path
        )

    elif suffix == ".json":

        model = XGBRegressor()

        model.load_model(
            model_path
        )

    else:

        raise ValueError(
            "Unsupported production model format:\n"
            f"{model_path}"
        )

    # --------------------------------------------------------
    # Validate model type
    # --------------------------------------------------------

    if not isinstance(
        model,
        XGBRegressor,
    ):

        raise TypeError(
            "Registered production model is not "
            "an XGBRegressor."
        )

    return (
        production,
        model,
    )


# ============================================================
# LOAD V4 DATA
# ============================================================

def load_v4_data():

    print()
    print("=" * 70)
    print("LOADING V4 TRAINING DATA")
    print("=" * 70)

    if not os.path.exists(
        TRAIN_FILE
    ):

        raise FileNotFoundError(
            "Training file not found:\n"
            f"{TRAIN_FILE}"
        )

    if not os.path.exists(
        VALIDATION_FILE
    ):

        raise FileNotFoundError(
            "Validation file not found:\n"
            f"{VALIDATION_FILE}"
        )

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Training rows   : "
        f"{len(train_df)}"
    )

    print(
        f"Validation rows : "
        f"{len(validation_df)}"
    )

    return (
        train_df,
        validation_df,
    )


# ============================================================
# PREPARE EXACT V4 FEATURES
# ============================================================

def prepare_v4_data(
    train_df,
    validation_df,
    production_model=None,
):

    # EXACTLY copied from V4 training logic.

    EXCLUDED_COLUMNS = [
        "target_aqi",
        "city_name",
        "date",
    ]

    TARGET_COLUMN = (
        "target_aqi"
    )

    # --------------------------------------------------------
    # Determine the feature set.
    #
    # The V4 feature store may contain additional columns that
    # were not part of the currently registered production
    # model. For production-safe retraining/evaluation, use
    # the production model's exact feature contract when it
    # is available.
    # --------------------------------------------------------

    production_feature_columns = None

    if production_model is not None and hasattr(
        production_model,
        "feature_names_in_"
    ):
        production_feature_columns = [
            str(column)
            for column in production_model.feature_names_in_
        ]

    if production_feature_columns:

        FEATURE_COLUMNS = production_feature_columns

        print()
        print(
            "Using production model feature contract."
        )
        print(
            f"Production model features: "
            f"{len(FEATURE_COLUMNS)}"
        )

        # ----------------------------------------------------
        # Safety check:
        # every production feature must exist in the
        # newly generated V4 dataset.
        # ----------------------------------------------------

        missing_train_features = [
            column
            for column in FEATURE_COLUMNS
            if column not in train_df.columns
        ]

        missing_validation_features = [
            column
            for column in FEATURE_COLUMNS
            if column not in validation_df.columns
        ]

        if missing_train_features:

            raise ValueError(
                "Features missing from training data: "
                f"{missing_train_features}"
            )

        if missing_validation_features:

            raise ValueError(
                "Features missing from validation data: "
                f"{missing_validation_features}"
            )

    else:

        # ----------------------------------------------------
        # Fallback for cases where no production model is
        # available. Preserve the normal V4 feature
        # selection behaviour.
        # ----------------------------------------------------

        FEATURE_COLUMNS = [
            column
            for column in train_df.columns
            if column not in EXCLUDED_COLUMNS
        ]

        missing_features = [
            column
            for column in FEATURE_COLUMNS
            if column not in validation_df.columns
        ]

        if missing_features:

            raise ValueError(
                "Features missing from validation data: "
                f"{missing_features}"
            )

    # --------------------------------------------------------
    # Safety check:
    # target must never become a feature.
    # --------------------------------------------------------

    if TARGET_COLUMN in FEATURE_COLUMNS:

        raise ValueError(
            "target_aqi cannot be used as a feature."
        )

    # --------------------------------------------------------
    # Production model compatibility check.
    #
    # If a production model exists, the feature count must
    # exactly match its trained feature count.
    # --------------------------------------------------------

    if production_model is not None and hasattr(
        production_model,
        "feature_names_in_"
    ):

        expected_feature_count = len(
            production_model.feature_names_in_
        )

        if len(FEATURE_COLUMNS) != expected_feature_count:

            raise RuntimeError(
                "Production feature mismatch. "
                f"Expected {expected_feature_count}, "
                f"got {len(FEATURE_COLUMNS)}."
            )

    # --------------------------------------------------------
    # Build matrices.
    # --------------------------------------------------------

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_validation = validation_df[
        FEATURE_COLUMNS
    ]

    y_validation = validation_df[
        TARGET_COLUMN
    ]

    print()

    print(
        f"Feature count: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Training matrix: "
        f"{X_train.shape}"
    )

    print(
        f"Validation matrix: "
        f"{X_validation.shape}"
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        FEATURE_COLUMNS,
    )


# ============================================================
# TRAIN EXACT V4 XGBOOST
# ============================================================

def train_candidate(
    X_train,
    y_train,
    X_validation,
    y_validation,
):

    print()
    print("=" * 70)
    print("TRAINING XGBOOST H1 CANDIDATE")
    print("=" * 70)

    start_time = time.time()

    # EXACT V4 XGBoost configuration.

    model = XGBRegressor(

        n_estimators=700,

        learning_rate=0.035,

        max_depth=6,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=0.1,

        reg_lambda=2.0,

        objective="reg:squarederror",

        eval_metric="rmse",

        random_state=42,

        n_jobs=-1,
    )

    # --------------------------------------------------------
    # V007 EXTREME-EVENT TRAINING WEIGHTS
    #
    # Keep the exact V4 feature contract and XGBoost
    # configuration unchanged.
    #
    # The weighting only affects candidate training.
    # V006 production is never modified.
    # --------------------------------------------------------

    train_sample_weight = np.ones(
        len(y_train),
        dtype=np.float32,
    )

    train_sample_weight[
        (y_train >= 150)
        & (y_train < 200)
    ] = 1.5

    train_sample_weight[
        (y_train >= 200)
        & (y_train < 300)
    ] = 2.0

    train_sample_weight[
        y_train >= 300
    ] = 3.0

    print()
    print("V007 EXTREME-EVENT SAMPLE WEIGHTS")
    print("-" * 70)
    print(
        "Target <150   : weight 1.0"
    )
    print(
        "Target 150-199: weight 1.5"
    )
    print(
        "Target 200-299: weight 2.0"
    )
    print(
        "Target >=300  : weight 3.0"
    )

    model.fit(

        X_train,

        y_train,

        sample_weight=train_sample_weight,

        eval_set=[
            (
                X_train,
                y_train,
            ),
            (
                X_validation,
                y_validation,
            ),
        ],

        verbose=False,
    )

    training_time = (
        time.time()
        - start_time
    )

    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        "CANDIDATE VALIDATION",
    )

    print()

    print(
        f"Training time  : "
        f"{training_time:.2f}s"
    )

    return (
        model,
        metrics,
        training_time,
    )


# ============================================================
# PROMOTION DECISION
# ============================================================

def should_promote(
    production_metrics,
    candidate_metrics,
):

    production_rmse = float(
        production_metrics["rmse"]
    )

    candidate_rmse = float(
        candidate_metrics["rmse"]
    )

    # --------------------------------------------------------
    # Prevent division by zero.
    # --------------------------------------------------------

    if production_rmse <= 0:

        raise ValueError(
            "Production RMSE must be greater than zero."
        )

    improvement_percent = (

        (
            production_rmse
            -
            candidate_rmse
        )

        /

        production_rmse

    ) * 100.0

    print()
    print("=" * 70)
    print("PRODUCTION VS CANDIDATE")
    print("=" * 70)

    print(
        "SAME VALIDATION DATASET"
    )

    print("-" * 70)

    print(
        f"Production RMSE : "
        f"{production_rmse:.4f}"
    )

    print(
        f"Candidate RMSE  : "
        f"{candidate_rmse:.4f}"
    )

    print()

    print(
        f"Production MAE  : "
        f"{production_metrics['mae']:.4f}"
    )

    print(
        f"Candidate MAE   : "
        f"{candidate_metrics['mae']:.4f}"
    )

    print()

    print(
        f"Production R2   : "
        f"{production_metrics['r2']:.4f}"
    )

    print(
        f"Candidate R2    : "
        f"{candidate_metrics['r2']:.4f}"
    )

    print()

    print(
        f"RMSE improvement: "
        f"{improvement_percent:+.4f}%"
    )

    print(
        f"Required minimum: "
        f"{MIN_RMSE_IMPROVEMENT_PERCENT:.4f}%"
    )

    if (
        improvement_percent
        >= MIN_RMSE_IMPROVEMENT_PERCENT
    ):

        print()
        print(
            "PROMOTION DECISION: ACCEPT"
        )

        return True

    print()
    print(
        "PROMOTION DECISION: REJECT"
    )

    return False


# ============================================================
# SAVE CANDIDATE
# ============================================================

def save_candidate(
    model,
):

    os.makedirs(
        CANDIDATE_DIR,
        exist_ok=True,
    )

    joblib.dump(
        model,
        CANDIDATE_FILE,
    )

    print()

    print(
        "Candidate saved:"
    )

    print(
        CANDIDATE_FILE
    )

    # --------------------------------------------------------
    # CRITICAL SAFETY CHECK
    #
    # Reload the ACTUAL serialized artifact before it can ever
    # be registered or promoted.
    # --------------------------------------------------------

    verify_model_artifact(
        CANDIDATE_FILE,
        expected_feature_count=107,
    )

    return CANDIDATE_FILE


# ============================================================
# VERIFY MODEL ARTIFACT
# ============================================================

def verify_model_artifact(
    model_path,
    expected_feature_count=107,
):
    """
    Verify that a serialized XGBoost artifact can actually be
    reloaded and used for inference.

    This is intentionally performed AFTER serialization so that
    a corrupted/incomplete pickle can never be accepted merely
    because the in-memory model was valid.
    """

    print()
    print("=" * 70)
    print("VERIFYING MODEL ARTIFACT")
    print("=" * 70)

    model_path = os.path.abspath(
        model_path
    )

    print(
        f"Artifact path : {model_path}"
    )

    if not os.path.exists(
        model_path
    ):
        raise FileNotFoundError(
            "Model artifact does not exist:\n"
            f"{model_path}"
        )

    try:

        model = joblib.load(
            model_path
        )

    except Exception as e:

        raise RuntimeError(
            "MODEL ARTIFACT VERIFICATION FAILED.\n"
            f"Could not reload artifact:\n"
            f"{model_path}\n"
            f"{type(e).__name__}: {e}"
        ) from e

    if not isinstance(
        model,
        XGBRegressor,
    ):
        raise TypeError(
            "MODEL ARTIFACT VERIFICATION FAILED.\n"
            "Reloaded object is not an XGBRegressor.\n"
            f"Loaded type: {type(model).__name__}"
        )

    actual_features = int(
        model.n_features_in_
    )

    print(
        f"Reloaded type : "
        f"{type(model).__name__}"
    )

    print(
        f"Features      : "
        f"{actual_features}"
    )

    if actual_features != int(
        expected_feature_count
    ):
        raise RuntimeError(
            "MODEL ARTIFACT VERIFICATION FAILED.\n"
            f"Expected {expected_feature_count} features, "
            f"got {actual_features}."
        )

    # --------------------------------------------------------
    # One-row inference smoke test.
    #
    # The model itself is tested using a finite synthetic row.
    # This does NOT touch the database or validation data.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Preserve the exact feature-name contract stored in the
    # serialized XGBoost model.
    #
    # A bare NumPy array would create columns named:
    #
    #     0, 1, 2, ..., 106
    #
    # while this model expects the real V4 feature names.
    # --------------------------------------------------------

    model_feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if model_feature_names is None:
        try:
            model_feature_names = (
                model.get_booster().feature_names
            )
        except Exception:
            model_feature_names = None

    if model_feature_names is None or len(model_feature_names) == 0:
        raise RuntimeError(
            "MODEL ARTIFACT VERIFICATION FAILED.\n"
            "Serialized model does not expose feature names."
        )

    model_feature_names = list(
        model_feature_names
    )

    if len(model_feature_names) != actual_features:
        raise RuntimeError(
            "MODEL ARTIFACT VERIFICATION FAILED.\n"
            "Model feature-name count does not match "
            "n_features_in_.\n"
            f"n_features_in_: {actual_features}\n"
            f"Feature names  : {len(model_feature_names)}"
        )

    smoke_input = pd.DataFrame(
        np.zeros(
            (
                1,
                actual_features,
            ),
            dtype=np.float32,
        ),
        columns=model_feature_names,
    )

    try:

        prediction = model.predict(
            smoke_input
        )

    except Exception as e:

        raise RuntimeError(
            "MODEL ARTIFACT VERIFICATION FAILED.\n"
            "Reloaded model could not perform inference.\n"
            f"{type(e).__name__}: {e}"
        ) from e

    if len(prediction) != 1:

        raise RuntimeError(
            "MODEL ARTIFACT VERIFICATION FAILED.\n"
            "Unexpected smoke-test prediction count."
        )

    if not np.all(
        np.isfinite(prediction)
    ):

        raise RuntimeError(
            "MODEL ARTIFACT VERIFICATION FAILED.\n"
            "Smoke-test prediction is not finite."
        )

    print(
        f"Smoke prediction: "
        f"{float(prediction[0]):.4f}"
    )

    print()
    print(
        "MODEL ARTIFACT VERIFICATION: PASSED"
    )

    return model


# ============================================================
# REGISTER ACCEPTED MODEL
# ============================================================

def register_candidate(
    candidate_path,
    candidate_metrics,
    training_rows,
    feature_count,
    training_time,
):

    print()
    print("=" * 70)
    print("REGISTERING IMPROVED MODEL")
    print("=" * 70)

    metadata = record_result(

        name="xgboost",

        horizon=1,

        metrics={

            "rmse":
                candidate_metrics[
                    "rmse"
                ],

            "mae":
                candidate_metrics[
                    "mae"
                ],

            "r2":
                candidate_metrics[
                    "r2"
                ],

            "model_path":
                candidate_path,

            "training_rows":
                training_rows,

            "feature_count":
                feature_count,

            "training_time_seconds":
                training_time,
        },
    )

    return metadata


# ============================================================
# VERIFY PRODUCTION
# ============================================================

def verify_production(
    expected_version,
):

    print()
    print("=" * 70)
    print("VERIFYING PRODUCTION MODEL")
    print("=" * 70)

    production = get_best_model(
        horizon=1
    )

    if production is None:

        raise RuntimeError(
            "No production model exists after promotion."
        )

    actual_version = int(
        production["version"]
    )

    expected_version = int(
        expected_version
    )

    print(
        f"Expected version : "
        f"v{expected_version:03d}"
    )

    print(
        f"Actual version   : "
        f"v{actual_version:03d}"
    )

    print(
        f"Status           : "
        f"{production.get('status')}"
    )

    print(
        f"RMSE             : "
        f"{production['rmse']:.4f}"
    )

    print(
        f"Path             : "
        f"{production['model_path']}"
    )

    if (
        actual_version
        != expected_version
    ):

        raise RuntimeError(
            "Production verification failed. "
            "Unexpected production version."
        )

    model_path = os.path.abspath(
        production["model_path"]
    )

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            "Promoted production model file does not exist:\n"
            f"{model_path}"
        )

    # --------------------------------------------------------
    # CRITICAL POST-PROMOTION SAFETY CHECK
    #
    # Reload the exact artifact that is now marked production.
    # A file merely existing is NOT sufficient.
    # --------------------------------------------------------

    verify_model_artifact(
        model_path,
        expected_feature_count=107,
    )

    print()
    print(
        "PRODUCTION VERIFICATION: PASSED"
    )

    return production


# ============================================================
# MAIN AUTO RETRAIN
# ============================================================

def run_auto_retrain():

    print()
    print("=" * 70)
    print("PEARLSAQI AUTOMATIC MODEL RETRAINING")
    print("=" * 70)

    print(
        "Production model will NOT be overwritten directly."
    )

    # --------------------------------------------------------
    # 1. LOAD CURRENT PRODUCTION
    # --------------------------------------------------------

    (
        production,
        production_model,
    ) = load_production_model()

    # --------------------------------------------------------
    # 2. LOAD V4 DATA
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
    ) = load_v4_data()

    # --------------------------------------------------------
    # 3. PREPARE EXACT V4 FEATURES
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    ) = prepare_v4_data(
        train_df,
        validation_df,
        production_model=production_model,
    )

    # --------------------------------------------------------
    # 4. EVALUATE CURRENT PRODUCTION
    #
    # IMPORTANT:
    #
    # Production and candidate MUST be evaluated against
    # the EXACT SAME validation dataset.
    #
    # This evaluation happens ONCE.
    # --------------------------------------------------------

    production_metrics = evaluate_model(
        production_model,
        X_validation,
        y_validation,
        "CURRENT PRODUCTION VALIDATION",
    )

    # --------------------------------------------------------
    # 5. TRAIN CANDIDATE
    # --------------------------------------------------------

    (
        candidate_model,
        candidate_metrics,
        training_time,
    ) = train_candidate(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    # --------------------------------------------------------
    # 6. FAIR PROMOTION GATE
    # --------------------------------------------------------

    promote = should_promote(
        production_metrics,
        candidate_metrics,
    )

    # --------------------------------------------------------
    # 7. REJECT CANDIDATE
    # --------------------------------------------------------

    if not promote:

        print()
        print("=" * 70)
        print("CANDIDATE REJECTED")
        print("=" * 70)

        print(
            "Existing production model remains active."
        )

        print(
            f"Active production version: "
            f"v{production['version']:03d}"
        )

        return {
            "promoted":
                False,

            "production_version":
                production["version"],

            "production_metrics":
                production_metrics,

            "candidate_metrics":
                candidate_metrics,
        }

    # --------------------------------------------------------
    # 8. SAVE CANDIDATE
    # --------------------------------------------------------

    candidate_path = (
        save_candidate(
            candidate_model
        )
    )

    # --------------------------------------------------------
    # 9. REGISTER CANDIDATE
    # --------------------------------------------------------

    metadata = (
        register_candidate(

            candidate_path=
                candidate_path,

            candidate_metrics=
                candidate_metrics,

            training_rows=
                len(train_df),

            feature_count=
                len(feature_columns),

            training_time=
                training_time,
        )
    )

    # --------------------------------------------------------
    # 10. PROMOTE REGISTERED VERSION
    #
    # promote_model() is responsible for:
    #
    # - making this version production
    # - archiving the old production version
    # - ensuring only one version is production
    #
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROMOTING ACCEPTED MODEL")
    print("=" * 70)

    promoted = promote_model(

        name="xgboost",

        horizon=1,

        version=metadata["version"],
    )

    # --------------------------------------------------------
    # 11. VERIFY PROMOTION
    # --------------------------------------------------------

    verified_production = (
        verify_production(
            expected_version=
                metadata["version"]
        )
    )

    # --------------------------------------------------------
    # 12. FINAL SUCCESS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("AUTOMATIC MODEL PROMOTION COMPLETE")
    print("=" * 70)

    print(
        f"Previous version : "
        f"v{production['version']:03d}"
    )

    print(
        f"New version      : "
        f"v{metadata['version']:03d}"
    )

    print(
        f"RMSE             : "
        f"{metadata['rmse']:.4f}"
    )

    print(
        f"MAE              : "
        f"{metadata['mae']:.4f}"
    )

    print(
        f"R2               : "
        f"{metadata['r2']:.4f}"
    )

    print(
        f"Production path  : "
        f"{metadata['model_path']}"
    )

    print(
        "Production status: VERIFIED"
    )

    return {

        "promoted":
            True,

        "previous_version":
            production["version"],

        "new_version":
            metadata["version"],

        "production_metrics":
            production_metrics,

        "candidate_metrics":
            candidate_metrics,

        "metadata":
            metadata,

        "production":
            verified_production,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_auto_retrain()