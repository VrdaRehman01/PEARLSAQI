import os
import time
import pandas as pd

from src.models.random_forest import RandomForestModel
from src.models.ridge import RidgeModel
from src.models.xgboost import XGBoostModel
from src.models.deep_model import DeepLearningModel
from src.models.model_registry import record_result


HORIZONS = [1, 2, 3]

BENCHMARK_DIR = "analytics"
BENCHMARK_FILE = os.path.join(
    BENCHMARK_DIR,
    "model_benchmark.csv"
)


def train_all_models():

    print()
    print("=" * 80)
    print("PEARLSAQI V4 FULL MODEL BENCHMARK")
    print("=" * 80)

    os.makedirs(
        BENCHMARK_DIR,
        exist_ok=True
    )

    results = []

    for horizon in HORIZONS:

        print()
        print("#" * 80)
        print(
            f"# FORECAST HORIZON: t+{horizon} DAY"
        )
        print("#" * 80)

        models = [

            RandomForestModel(
                horizon
            ),

            RidgeModel(
                horizon
            ),

            XGBoostModel(
                horizon
            ),

            DeepLearningModel(
                horizon
            ),
        ]

        for model in models:

            print()
            print("-" * 80)

            print(
                f"MODEL: {model.name}"
            )

            print(
                f"HORIZON: H{horizon}"
            )

            print("-" * 80)

            started = time.time()

            try:

                metrics = model.train()

                # --------------------------------------------------
                # REGISTER MODEL
                # --------------------------------------------------

                metadata = record_result(
                    model.name,
                    horizon,
                    metrics
                )

                # --------------------------------------------------
                # BENCHMARK RESULT
                # --------------------------------------------------

                results.append({

                    "model":
                        model.name,

                    "horizon":
                        horizon,

                    "rmse":
                        metrics["rmse"],

                    "mae":
                        metrics["mae"],

                    "r2":
                        metrics["r2"],

                    "training_rows":
                        metrics.get(
                            "training_rows"
                        ),

                    "validation_rows":
                        metrics.get(
                            "validation_rows"
                        ),

                    "feature_count":
                        metrics.get(
                            "feature_count"
                        ),

                    "training_time_seconds":
                        metrics.get(
                            "training_time_seconds",
                            time.time() - started
                        ),

                    "registry_version":
                        metadata["version"],

                    "status":
                        "success",

                    "error":
                        None,
                })

            except Exception as error:

                print()
                print(
                    f"ERROR: {model.name} "
                    f"H{horizon}"
                )

                print(error)

                results.append({

                    "model":
                        model.name,

                    "horizon":
                        horizon,

                    "rmse":
                        None,

                    "mae":
                        None,

                    "r2":
                        None,

                    "training_rows":
                        None,

                    "validation_rows":
                        None,

                    "feature_count":
                        None,

                    "training_time_seconds":
                        time.time() - started,

                    "registry_version":
                        None,

                    "status":
                        "failed",

                    "error":
                        str(error),
                })

    # ==========================================================
    # SAVE BENCHMARK
    # ==========================================================

    benchmark_df = pd.DataFrame(
        results
    )

    benchmark_df.to_csv(
        BENCHMARK_FILE,
        index=False
    )

    # ==========================================================
    # FINAL SUMMARY
    # ==========================================================

    print()
    print("=" * 80)
    print("FULL BENCHMARK COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Total experiments : "
        f"{len(results)}"
    )

    print(
        f"Successful        : "
        f"{sum(r['status'] == 'success' for r in results)}"
    )

    print(
        f"Failed            : "
        f"{sum(r['status'] == 'failed' for r in results)}"
    )

    # ==========================================================
    # RANKING
    # ==========================================================

    successful = benchmark_df[
        benchmark_df["status"] == "success"
    ].copy()

    if not successful.empty:

        print()
        print("=" * 80)
        print("MODEL BENCHMARK RANKING")
        print("=" * 80)

        ranking = successful.sort_values(
            [
                "horizon",
                "rmse"
            ]
        )

        print()

        print(
            ranking[
                [
                    "model",
                    "horizon",
                    "rmse",
                    "mae",
                    "r2",
                    "training_time_seconds",
                    "registry_version",
                ]
            ].to_string(
                index=False
            )
        )

        # ======================================================
        # BEST MODEL PER HORIZON
        # ======================================================

        print()
        print("=" * 80)
        print("BEST MODEL PER FORECAST HORIZON")
        print("=" * 80)

        for horizon in HORIZONS:

            horizon_df = successful[
                successful["horizon"]
                == horizon
            ]

            if horizon_df.empty:

                print(
                    f"H{horizon}: "
                    "NO SUCCESSFUL MODEL"
                )

                continue

            best = horizon_df.loc[
                horizon_df["rmse"].idxmin()
            ]

            print()
            print(
                f"H{horizon}"
            )

            print(
                f"  Model : "
                f"{best['model']}"
            )

            print(
                f"  RMSE  : "
                f"{best['rmse']:.4f}"
            )

            print(
                f"  MAE   : "
                f"{best['mae']:.4f}"
            )

            print(
                f"  R²    : "
                f"{best['r2']:.4f}"
            )

            print(
                f"  Version : "
                f"v{int(best['registry_version']):03d}"
            )

    print()
    print(
        f"Benchmark saved to: "
        f"{BENCHMARK_FILE}"
    )

    print()
    print("=" * 80)
    print("BENCHMARK PIPELINE FINISHED")
    print("=" * 80)


if __name__ == "__main__":

    train_all_models()