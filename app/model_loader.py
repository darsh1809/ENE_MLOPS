"""
model_loader.py
---------------
Loads trained KMeans and RandomForest models from MLflow or local fallback.
"""

import os
import logging
import mlflow
import mlflow.sklearn
import joblib

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://54.206.46.48:5000")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "customer")

# Local fallback paths (relative to project root)
LOCAL_KMEANS_PATH = os.getenv("KMEANS_MODEL_PATH", "models/kmeans_model.pkl")
LOCAL_RF_PATH     = os.getenv("RF_MODEL_PATH",     "models/rf_model.pkl")
LOCAL_SCALER_PATH = os.getenv("SCALER_MODEL_PATH", "models/scaler.pkl")


def _load_from_mlflow(run_name: str, artifact_path: str):
    """Search the experiment for the latest run matching run_name and load its artifact."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{MLFLOW_EXPERIMENT}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No MLflow run found with name '{run_name}'.")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info("Loading model from MLflow: %s", model_uri)
    return mlflow.sklearn.load_model(model_uri)


def load_kmeans():
    """Load the KMeans clustering model (MLflow → local fallback)."""
    try:
        return _load_from_mlflow("Step4_KMeans_Clustering", "kmeans_model")
    except Exception as exc:
        logger.warning("MLflow KMeans load failed (%s); trying local file.", exc)

    if os.path.exists(LOCAL_KMEANS_PATH):
        return joblib.load(LOCAL_KMEANS_PATH)

    raise FileNotFoundError(
        f"KMeans model not found in MLflow or locally at '{LOCAL_KMEANS_PATH}'."
    )


def load_random_forest():
    """Load the RandomForest classifier (MLflow → local fallback)."""
    try:
        return _load_from_mlflow("Step6_RandomForest_Classifier", "random_forest_model")
    except Exception as exc:
        logger.warning("MLflow RF load failed (%s); trying local file.", exc)

    if os.path.exists(LOCAL_RF_PATH):
        return joblib.load(LOCAL_RF_PATH)

    raise FileNotFoundError(
        f"RandomForest model not found in MLflow or locally at '{LOCAL_RF_PATH}'."
    )


def load_scaler():
    """Load the StandardScaler used during KMeans training."""
    if os.path.exists(LOCAL_SCALER_PATH):
        return joblib.load(LOCAL_SCALER_PATH)
    logger.warning("Scaler not found at '%s'; returning None.", LOCAL_SCALER_PATH)
    return None
