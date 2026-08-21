import os
import time
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.models.data_loader import load_training_data


MODEL_DIR = "models"


class DeepLearningModel:

    name = "deep_learning"

    def __init__(self, horizon=1):
        self.horizon = horizon

    def train(self):

        import tensorflow as tf

        start_time = time.time()

        # --------------------------------------------------
        # LOAD DATA
        # --------------------------------------------------

        X_train, X_test, y_train, y_test = (
            load_training_data(
                self.horizon
            )
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
        # SCALE FEATURES
        # --------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        # --------------------------------------------------
        # BUILD NETWORK
        # --------------------------------------------------

        model = tf.keras.Sequential([

            tf.keras.layers.Input(
                shape=(X_train_scaled.shape[1],)
            ),

            tf.keras.layers.Dense(
                64,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                1
            ),
        ])

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"],
        )

        # --------------------------------------------------
        # TRAIN
        # --------------------------------------------------

        model.fit(
            X_train_scaled,
            y_train,
            epochs=50,
            batch_size=16,
            validation_split=0.1,
            verbose=0,
        )

        # --------------------------------------------------
        # PREDICT
        # --------------------------------------------------

        preds = (
            model
            .predict(
                X_test_scaled,
                verbose=0
            )
            .flatten()
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
        # RESULTS
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
            "model.keras"
        )

        scaler_path = os.path.join(
            model_dir,
            "scaler.pkl"
        )

        model.save(
            model_path
        )

        joblib.dump(
            scaler,
            scaler_path
        )

        print(
            f"Saved -> {model_path}"
        )

        # --------------------------------------------------
        # RETURN STANDARD BENCHMARK CONTRACT
        # --------------------------------------------------

        return {

            "rmse": rmse,

            "mae": mae,

            "r2": r2,

            "model_path": model_path,

            "scaler_path": scaler_path,

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