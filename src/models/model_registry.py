import json
import os
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd


# ============================================================
# PEARLSAQI LOCAL MODEL REGISTRY
# ============================================================
#
# Hopsworks replacement.
#
# Models:
#
#     models/registry/<model_name>/v001/
#     models/registry/<model_name>/v002/
#     models/registry/<model_name>/v003/
#
# Registry:
#
#     models/registry/registry.json
#
# Model statuses:
#
#     registered
#         Candidate/model exists in registry but is not production.
#
#     production
#         Explicitly selected production model.
#
#     archived
#         Previously registered/production model that is no longer active.
#
# IMPORTANT:
#
# Production selection is NOT based purely on historical RMSE.
# A model must explicitly have:
#
#     "status": "production"
#
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_DIR = (
    PROJECT_ROOT
    / "models"
    / "registry"
)

REGISTRY_FILE = (
    REGISTRY_DIR
    / "registry.json"
)


REGISTRY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PORTABLE REGISTRY PATHS
# ============================================================

def _relative_registry_path(path):
    """
    Convert a registry-owned file path into a project-relative
    POSIX path.

    Registry metadata must never depend on a machine-specific
    absolute path such as:

        <project-root>/...

    Instead, registry.json stores paths such as:

        models/registry/xgboost_h1/v006/model.pkl
    """

    path = Path(path)

    try:
        relative = path.resolve().relative_to(
            PROJECT_ROOT.resolve()
        )

    except ValueError:

        raise ValueError(
            "Registry model/scaler path must be inside "
            f"the project root.\n"
            f"Project root: {PROJECT_ROOT}\n"
            f"Path: {path}"
        )

    return relative.as_posix()


def _resolve_registry_path(path):
    """
    Resolve a registry path for the current environment.

    Supports both:

    1. New project-relative paths
       models/registry/...

    2. Existing absolute paths
       <project-root>/models/...

    This allows old registry entries to continue working
    while new entries remain Docker/WSL portable.
    """

    path = Path(path)

    if path.is_absolute():

        # Existing registry entry from Windows or another
        # environment.
        #
        # If it already exists, use it directly.
        if path.exists():

            return path.resolve()

        # Attempt to recover an old absolute path by locating
        # the project-relative portion after the project root.
        normalized = str(path).replace(
            "\\",
            "/"
        )

        marker = "/models/registry/"

        if marker in normalized:

            relative = (
                "models/registry/"
                +
                normalized.split(
                    marker,
                    1
                )[1]
            )

            candidate = (
                PROJECT_ROOT
                /
                Path(relative)
            )

            if candidate.exists():

                return candidate.resolve()

        return path

    return (
        PROJECT_ROOT
        /
        path
    ).resolve()


MODEL_TYPES = [
    "random_forest",
    "ridge",
    "xgboost",
    "deep_learning",
]


# ============================================================
# REGISTRY FILE
# ============================================================

def _load_registry():

    if not REGISTRY_FILE.exists():

        return {
            "models": {},
            "created_at":
                datetime.utcnow().isoformat()
        }

    try:

        with open(
            REGISTRY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if "models" not in data:

            data["models"] = {}

        return data

    except Exception:

        return {
            "models": {},
            "created_at":
                datetime.utcnow().isoformat()
        }


def _save_registry(
    data
):

    REGISTRY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = (
        REGISTRY_FILE.with_suffix(
            ".tmp"
        )
    )

    with open(
        temporary,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            default=str
        )

    temporary.replace(
        REGISTRY_FILE
    )


# ============================================================
# MODEL NAME
# ============================================================

def _model_name(
    name,
    horizon
):

    return (
        f"{name}_h{horizon}"
    )


# ============================================================
# VERSION
# ============================================================

def _next_version(
    model_name,
    registry
):

    existing = (
        registry[
            "models"
        ].get(
            model_name,
            []
        )
    )

    if not existing:

        return 1

    versions = []

    for item in existing:

        try:

            versions.append(
                int(
                    item["version"]
                )
            )

        except Exception:

            pass

    if not versions:

        return 1

    return (
        max(versions)
        + 1
    )


# ============================================================
# FIND REGISTRY ENTRY
# ============================================================

def _find_entry(
    registry,
    model_name,
    version
):

    entries = (
        registry[
            "models"
        ].get(
            model_name,
            []
        )
    )

    for entry in entries:

        if int(
            entry["version"]
        ) == int(version):

            return entry

    return None


# ============================================================
# REGISTER MODEL
# ============================================================

def record_result(
    name,
    horizon,
    metrics
):
    """
    Register a trained model locally.

    Newly registered models receive:

        status = "registered"

    They are NOT automatically production.

    Expected metrics:

        rmse
        mae
        r2
        model_path

    Optional:

        scaler_path
        training_rows
        feature_count
        training_time_seconds
    """

    required = [
        "rmse",
        "mae",
        "r2",
        "model_path"
    ]

    missing = [
        key
        for key in required
        if key not in metrics
    ]

    if missing:

        raise ValueError(
            f"Missing model metrics: "
            f"{missing}"
        )

    model_path = Path(
        metrics["model_path"]
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file does not exist: "
            f"{model_path}"
        )

    registry = _load_registry()

    model_name = _model_name(
        name,
        horizon
    )

    version = _next_version(
        model_name,
        registry
    )

    version_name = (
        f"v{version:03d}"
    )

    version_dir = (
        REGISTRY_DIR
        / model_name
        / version_name
    )

    version_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Copy model
    # --------------------------------------------------------

    destination_model = (
        version_dir
        / model_path.name
    )

    shutil.copy2(
        model_path,
        destination_model
    )

    # --------------------------------------------------------
    # Copy scaler if available
    # --------------------------------------------------------

    scaler_destination = None

    scaler_path = metrics.get(
        "scaler_path"
    )

    if scaler_path:

        scaler_path = Path(
            scaler_path
        )

        if scaler_path.exists():

            scaler_destination = (
                version_dir
                / scaler_path.name
            )

            shutil.copy2(
                scaler_path,
                scaler_destination
            )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "name":
            name,

        "model_name":
            model_name,

        "horizon":
            int(horizon),

        "version":
            version,

        "version_name":
            version_name,

        "rmse":
            float(
                metrics["rmse"]
            ),

        "mae":
            float(
                metrics["mae"]
            ),

        "r2":
            float(
                metrics["r2"]
            ),

        "model_path":
            _relative_registry_path(
                destination_model
            ),

        "scaler_path":
            (
                _relative_registry_path(
                    scaler_destination
                )
                if scaler_destination
                else None
            ),

        "training_rows":
            metrics.get(
                "training_rows"
            ),

        "feature_count":
            metrics.get(
                "feature_count"
            ),

        "training_time_seconds":
            metrics.get(
                "training_time_seconds"
            ),

        "created_at":
            datetime.utcnow().isoformat(),

        # New models are NEVER automatically production.
        "status":
            "registered"
    }

    metadata_file = (
        version_dir
        / "metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            default=str
        )

    # --------------------------------------------------------
    # Registry entry
    # --------------------------------------------------------

    registry[
        "models"
    ].setdefault(
        model_name,
        []
    ).append(
        metadata
    )

    _save_registry(
        registry
    )

    print(
        f"Registered "
        f"{model_name} "
        f"v{version}"
    )

    print(
        f"RMSE: "
        f"{metadata['rmse']:.4f}"
    )

    print(
        f"MAE : "
        f"{metadata['mae']:.4f}"
    )

    print(
        f"R²  : "
        f"{metadata['r2']:.4f}"
    )

    print(
        f"Path: "
        f"{destination_model}"
    )

    return metadata


# ============================================================
# PROMOTE MODEL TO PRODUCTION
# ============================================================

def promote_model(
    name,
    horizon,
    version
):
    """
    Promote exactly one registered model version
    to production.

    Rules:
        - The requested version must exist.
        - Its model file must exist.
        - All other versions of the same model/horizon
          become archived.
        - Only the requested version becomes production.
        - Registry is saved atomically.
    """

    registry = _load_registry()

    model_name = _model_name(
        name,
        horizon
    )

    entries = registry[
        "models"
    ].get(
        model_name,
        []
    )

    if not entries:

        raise ValueError(
            f"No registered models found for "
            f"{model_name}."
        )

    target = None

    for entry in entries:

        if int(
            entry["version"]
        ) == int(version):

            target = entry

            break

    if target is None:

        raise ValueError(
            f"Version v{int(version):03d} "
            f"does not exist for {model_name}."
        )

    model_path = Path(
        target["model_path"]
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Production model file does not exist:\n"
            f"{model_path}"
        )

    # --------------------------------------------------------
    # Archive every other version
    # --------------------------------------------------------

    for entry in entries:

        if int(
            entry["version"]
        ) == int(version):

            entry["status"] = "production"

        else:

            if entry.get(
                "status"
            ) == "production":

                entry["status"] = "archived"

    # --------------------------------------------------------
    # Save registry
    # --------------------------------------------------------

    _save_registry(
        registry
    )

    print()
    print("=" * 70)
    print("MODEL PROMOTED TO PRODUCTION")
    print("=" * 70)

    print(
        f"Model   : {name}"
    )

    print(
        f"Horizon : h{int(horizon)}"
    )

    print(
        f"Version : v{int(version):03d}"
    )

    print(
        f"RMSE    : {float(target['rmse']):.4f}"
    )

    print(
        f"MAE     : {float(target['mae']):.4f}"
    )

    print(
        f"R?      : {float(target['r2']):.4f}"
    )

    print(
        f"Path    : {model_path}"
    )

    print(
        "Status  : production"
    )

    return target


def _write_metadata_file(
    entry
):

    model_path = Path(
        entry["model_path"]
    )

    metadata_file = (
        model_path.parent
        / "metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            entry,
            f,
            indent=2,
            default=str
        )


# ============================================================
# GET PRODUCTION MODEL
# ============================================================

def get_best_model(
    horizon
):
    """
    Return the explicitly selected production model.

    IMPORTANT:

    This function no longer selects a model merely because
    it has the lowest historical RMSE.

    A model MUST have:

        status == "production"

    This prevents stale or incomparable validation metrics
    from silently controlling production inference.
    """

    registry = _load_registry()

    production_candidates = []

    for model_type in MODEL_TYPES:

        model_name = _model_name(
            model_type,
            horizon
        )

        entries = (
            registry[
                "models"
            ].get(
                model_name,
                []
            )
        )

        if not entries:
            continue

        production_entries = [
            entry
            for entry in entries
            if entry.get(
                "status"
            ) == "production"
        ]

        if not production_entries:
            continue

        latest_production = max(
            production_entries,
            key=lambda x:
                int(
                    x["version"]
                )
        )

        production_candidates.append(
            latest_production
        )

    if not production_candidates:

        return None

    # If more than one model type is marked production,
    # select the one with the best production RMSE.
    best = min(
        production_candidates,
        key=lambda x:
            float(
                x["rmse"]
            )
    )

    model_path = _resolve_registry_path(
        best["model_path"]
    )

    if not model_path.exists():

        return None

    return {

        "name":
            best["name"],

        "horizon":
            best["horizon"],

        "version":
            best["version"],

        "version_name":
            best.get(
                "version_name"
            ),

        "rmse":
            best["rmse"],

        "mae":
            best["mae"],

        "r2":
            best["r2"],

        "model_path":
            str(
                model_path
            ),

        "scaler_path":
            (
                str(
                    _resolve_registry_path(
                        best["scaler_path"]
                    )
                )
                if best.get("scaler_path")
                else None
            ),

        "status":
            best.get(
                "status"
            ),
    }


# ============================================================
# GET ALL REGISTERED MODELS
# ============================================================

def get_all_registered_models():

    registry = _load_registry()

    results = []

    for model_name, entries in (
        registry[
            "models"
        ].items()
    ):

        for entry in entries:

            results.append({

                "name":
                    entry["name"],

                "model_name":
                    model_name,

                "horizon":
                    entry["horizon"],

                "version":
                    entry["version"],

                "version_name":
                    entry.get(
                        "version_name"
                    ),

                "rmse":
                    entry["rmse"],

                "mae":
                    entry["mae"],

                "r2":
                    entry["r2"],

                "model_path":
                    entry["model_path"],

                "created_at":
                    entry.get(
                        "created_at"
                    ),

                "promoted_at":
                    entry.get(
                        "promoted_at"
                    ),

                "status":
                    entry.get(
                        "status",
                        "registered"
                    )
            })

    return results


# ============================================================
# LATEST MODEL
# ============================================================

def get_latest_model(
    name,
    horizon
):

    registry = _load_registry()

    model_name = _model_name(
        name,
        horizon
    )

    entries = (
        registry[
            "models"
        ].get(
            model_name,
            []
        )
    )

    if not entries:

        return None

    latest = max(
        entries,
        key=lambda x:
            int(
                x["version"]
            )
    )

    if not _resolve_registry_path(
        latest["model_path"]
    ).exists():

        return None

    latest = dict(
        latest
    )

    latest["model_path"] = str(
        _resolve_registry_path(
            latest["model_path"]
        )
    )

    if latest.get("scaler_path"):

        latest["scaler_path"] = str(
            _resolve_registry_path(
                latest["scaler_path"]
            )
        )

    return latest


# ============================================================
# MODEL VERSIONS
# ============================================================

def get_model_versions(
    name,
    horizon
):

    registry = _load_registry()

    model_name = _model_name(
        name,
        horizon
    )

    return (
        registry[
            "models"
        ].get(
            model_name,
            []
        )
    )


# ============================================================
# REGISTRY STATISTICS
# ============================================================

def get_registry_stats():

    models = (
        get_all_registered_models()
    )

    if not models:

        return {

            "total_models":
                0,

            "model_types":
                0,

            "horizons":
                0,

            "production_models":
                0,

            "latest_versions":
                {}
        }

    latest_versions = {}

    production_models = 0

    for model in models:

        key = model[
            "model_name"
        ]

        current = (
            latest_versions.get(
                key
            )
        )

        if (
            current is None
            or
            model["version"]
            >
            current["version"]
        ):

            latest_versions[
                key
            ] = model

        if (
            model["status"]
            == "production"
        ):

            production_models += 1

    return {

        "total_models":
            len(models),

        "model_types":
            len(
                set(
                    x["name"]
                    for x in models
                )
            ),

        "horizons":
            len(
                set(
                    x["horizon"]
                    for x in models
                )
            ),

        "production_models":
            production_models,

        "latest_versions":
            latest_versions
    }


# ============================================================
# CLI INSPECTION
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "PEARLSAQI LOCAL MODEL REGISTRY"
    )
    print("=" * 70)

    stats = (
        get_registry_stats()
    )

    print()

    print(
        f"Registered versions : "
        f"{stats['total_models']}"
    )

    print(
        f"Model types         : "
        f"{stats['model_types']}"
    )

    print(
        f"Horizons            : "
        f"{stats['horizons']}"
    )

    print(
        f"Production models   : "
        f"{stats['production_models']}"
    )

    models = (
        get_all_registered_models()
    )

    if models:

        print()
        print(
            "REGISTERED MODELS"
        )
        print(
            "-" * 70
        )

        df = pd.DataFrame(
            models
        )

        columns = [

            "name",

            "horizon",

            "version",

            "version_name",

            "rmse",

            "mae",

            "r2",

            "status",
        ]

        print(
            df[
                columns
            ].to_string(
                index=False
            )
        )

    else:

        print()
        print(
            "No models registered yet."
        )

    print()
    print(
        "LOCAL MODEL REGISTRY READY"
    )