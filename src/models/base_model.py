import os
import time
import joblib
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.models.data_loader import load_training_data


MODEL_DIR = "models"


class BaseModel:

    name = "base"
    estimator = None

    def __init__(self, horizon=1):
        self.horizon = horizon

    def train(self):

        start_time = time.time()

        X_train, X_test, y_train, y_test = (
            load_training_data(self.horizon)
        )

        print(
            f"Training {self.name} "
            f"(t+{self.horizon} day)..."
        )

        print(
            f"Training rows   : {len(X_train)}"
        )

        print(
            f"Validation rows : {len(X_test)}"
        )

        print(
            f"Features        : {X_train.shape[1]}"
        )

        # --------------------------------------------------
        # TRAIN
        # --------------------------------------------------

        self.estimator.fit(
            X_train,
            y_train
        )

        # --------------------------------------------------
        # PREDICT
        # --------------------------------------------------

        preds = self.estimator.predict(
            X_test
        )

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    preds
                )
            )
        )

        mae = float(
            mean_absolute_error(
                y_test,
                preds
            )
        )

        r2 = float(
            r2_score(
                y_test,
                preds
            )
        )

        training_time = (
            time.time()
            - start_time
        )

        # --------------------------------------------------
        # OUTPUT
        # --------------------------------------------------

        print()

        print(
            f"{self.name} "
            f"(h{self.horizon})"
        )

        print(
            f"RMSE : {rmse:.4f}"
        )

        print(
            f"MAE  : {mae:.4f}"
        )

        print(
            f"R2   : {r2:.4f}"
        )

        print(
            f"Time : {training_time:.2f}s"
        )

        # --------------------------------------------------
        # SAVE MODEL
        # --------------------------------------------------

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        model_dir = os.path.join(
            MODEL_DIR,
            f"{self.name}_h{self.horizon}"
        )

        os.makedirs(
            model_dir,
            exist_ok=True
        )

        model_path = os.path.join(
            model_dir,
            "model.pkl"
        )

        joblib.dump(
            self.estimator,
            model_path
        )

        print(
            f"Saved -> {model_path}"
        )

        # --------------------------------------------------
        # RETURN BENCHMARK METADATA
        # --------------------------------------------------

        return {

            "rmse": rmse,

            "mae": mae,

            "r2": r2,

            "model_path": model_path,

            "training_rows": int(
                len(X_train)
            ),

            "validation_rows": int(
                len(X_test)
            ),

            "feature_count": int(
                X_train.shape[1]
            ),

            "training_time_seconds": float(
                training_time
            ),
        }