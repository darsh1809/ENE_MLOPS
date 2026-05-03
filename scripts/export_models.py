"""
scripts/export_models.py
------------------------
Exports trained models from MLflow to models/*.pkl for Docker inference.

Strategy (priority order):
  1. Remote MLflow HTTP server (MLFLOW_TRACKING_URI env var — used in CI/CD)
  2. Local MLflow HTTP server on 127.0.0.1:5000 (if user ran `mlflow ui`)
  3. Disk scan of mlartifacts/ using run IDs from mlflow.db (offline / local dev)

Usage:
    # Normal (CI or with local MLflow server running):
    python scripts/export_models.py

    # Offline (no server, pure disk scan):
    OFFLINE=1 python scripts/export_models.py
"""

import os
import sys
import glob
import socket
import logging
import sqlite3
import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://54.206.46.48:5000")
LOCAL_SRV     = "http://127.0.0.1:5000"
EXPERIMENT    = os.getenv("MLFLOW_EXPERIMENT",   "customer")
MODELS_DIR    = "models"
ARTIFACTS_DIR = "mlartifacts"
SQLITE_DB     = "mlflow.db"
OFFLINE       = os.getenv("OFFLINE", "0") == "1"

os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# TCP probe
# ─────────────────────────────────────────────────────────────────────────────

def _reachable(uri: str, timeout: float = 2.0) -> bool:
    try:
        from urllib.parse import urlparse
        p = urlparse(uri)
        with socket.create_connection((p.hostname or "localhost", p.port or 80), timeout=timeout):
            return True
    except OSError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Strategy A — MLflow HTTP server
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_via_mlflow(tracking_uri: str, run_name: str, artifact_path: str):
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT}' not found at {tracking_uri}")
    flt  = "tags.mlflow.runName = '" + run_name + "'"
    runs = client.search_runs([exp.experiment_id], filter_string=flt,
                              order_by=["start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError(f"No run '{run_name}' at {tracking_uri}")
    uri = f"runs:/{runs[0].info.run_id}/{artifact_path}"
    logger.info("  MLflow URI: %s", uri)
    return mlflow.sklearn.load_model(uri)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy B — disk scan via mlflow.db run IDs
# ─────────────────────────────────────────────────────────────────────────────

def _latest_run_id_from_db(run_name: str) -> str | None:
    """Query mlflow.db for the newest run_uuid matching run_name."""
    if not os.path.exists(SQLITE_DB):
        return None
    try:
        conn = sqlite3.connect(SQLITE_DB)
        cur  = conn.cursor()
        cur.execute(
            "SELECT run_uuid FROM runs WHERE name = ? ORDER BY start_time DESC LIMIT 1",
            (run_name,),
        )
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as exc:
        logger.warning("SQLite lookup failed: %s", exc)
        return None


def _pkl_for_run_id(run_id: str) -> str | None:
    """Find model.pkl under mlartifacts/ whose MLmodel file references run_id."""
    pattern = os.path.join(ARTIFACTS_DIR, "**", "MLmodel")
    for mlmodel_path in glob.glob(pattern, recursive=True):
        try:
            with open(mlmodel_path, encoding="utf-8") as f:
                if run_id in f.read():
                    pkl = os.path.join(os.path.dirname(mlmodel_path), "model.pkl")
                    if os.path.exists(pkl):
                        return pkl
        except Exception:
            continue
    return None


def _load_from_disk(run_name: str) -> object:
    run_id = _latest_run_id_from_db(run_name)
    if run_id:
        logger.info("  DB run_id: %s", run_id)
        pkl = _pkl_for_run_id(run_id)
        if pkl:
            logger.info("  Disk path: %s", pkl)
            return joblib.load(pkl)
        logger.warning("  model.pkl not found for run_id %s; broadening search.", run_id)

    # Last resort: newest pkl in artifacts dir
    pkls = sorted(
        glob.glob(os.path.join(ARTIFACTS_DIR, "**", "model.pkl"), recursive=True),
        key=os.path.getmtime, reverse=True,
    )
    if pkls:
        logger.info("  Fallback pkl: %s", pkls[0])
        return joblib.load(pkls[0])

    raise FileNotFoundError(
        f"No model.pkl found in '{ARTIFACTS_DIR}/'. Run the pipeline first."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unified export
# ─────────────────────────────────────────────────────────────────────────────

def export_model(run_name: str, artifact_path: str, out_path: str):
    logger.info("Exporting: %s", run_name)
    model = None

    if not OFFLINE:
        for uri, label in [(TRACKING_URI, "remote"), (LOCAL_SRV, "local-server")]:
            if not _reachable(uri):
                logger.info("  %s server not reachable (%s)", label.capitalize(), uri)
                continue
            try:
                model = _fetch_via_mlflow(uri, run_name, artifact_path)
                logger.info("  Source: %s MLflow server", label)
                break
            except Exception as exc:
                logger.warning("  %s server error: %s", label.capitalize(), exc)

    if model is None:
        logger.info("  Source: disk (mlartifacts/ + mlflow.db)")
        model = _load_from_disk(run_name)

    joblib.dump(model, out_path)
    size_kb = os.path.getsize(out_path) // 1024
    logger.info("  Saved → %-35s (%d KB)", out_path, size_kb)


def main():
    export_model("Step4_KMeans_Clustering",      "kmeans_model",        os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    export_model("Step6_RandomForest_Classifier","random_forest_model", os.path.join(MODELS_DIR, "rf_model.pkl"))

    logger.info("Export complete. Models in %s/:", MODELS_DIR)
    for f in sorted(os.listdir(MODELS_DIR)):
        size_kb = os.path.getsize(os.path.join(MODELS_DIR, f)) // 1024
        if size_kb > 0:
            logger.info("  %-35s %d KB", f, size_kb)


if __name__ == "__main__":
    main()
