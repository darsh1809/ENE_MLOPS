"""
routes.py
---------
All Flask route definitions for the Customer Segmentation API.
"""

import logging
import time

from flask import Blueprint, jsonify, request

from .model_loader import load_kmeans, load_random_forest, load_scaler
from .utils import (
    build_classifier_features,
    build_rfm_features,
    cluster_to_segment,
    validate_classifier_input,
    validate_rfm_input,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

# ---------------------------------------------------------------------------
# Lazy model holders (loaded once on first request)
# ---------------------------------------------------------------------------
_kmeans   = None
_scaler   = None
_rf       = None


def _get_kmeans():
    global _kmeans
    if _kmeans is None:
        _kmeans = load_kmeans()
    return _kmeans


def _get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = load_scaler()
    return _scaler


def _get_rf():
    global _rf
    if _rf is None:
        _rf = load_random_forest()
    return _rf


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------

@api_bp.route("/health", methods=["GET"])
def health():
    """Liveness probe."""
    return jsonify({"status": "ok", "service": "customer-segmentation-api"}), 200


@api_bp.route("/ready", methods=["GET"])
def ready():
    """Readiness probe — verifies models can be loaded."""
    try:
        _get_kmeans()
        _get_rf()
        return jsonify({"status": "ready"}), 200
    except Exception as exc:
        logger.exception("Readiness check failed")
        return jsonify({"status": "not_ready", "error": str(exc)}), 503


# ---------------------------------------------------------------------------
# Main prediction endpoint
# ---------------------------------------------------------------------------

@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    Predict customer segment from RFM features.

    Request JSON:
    {
        "Recency":   <int|float>,   # days since last purchase
        "Frequency": <int|float>,   # number of orders
        "Monetary":  <int|float>    # total spend (£)
    }

    Response JSON:
    {
        "cluster":         <int>,
        "segment":         <str>,
        "probabilities":   null,
        "input":           { ... },
        "latency_ms":      <float>
    }
    """
    t0 = time.perf_counter()

    data = request.get_json(silent=True)

    # ── Validate ──────────────────────────────────────────────────────────
    valid, err = validate_rfm_input(data)
    if not valid:
        return jsonify({"error": err}), 400

    try:
        kmeans = _get_kmeans()
        scaler = _get_scaler()

        X = build_rfm_features(data)

        if scaler is not None:
            X = scaler.transform(X)

        cluster = int(kmeans.predict(X)[0])
        segment = cluster_to_segment(cluster)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify({
            "cluster":       cluster,
            "segment":       segment,
            "probabilities": None,
            "input":         data,
            "latency_ms":    latency_ms,
        }), 200

    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed.", "detail": str(exc)}), 500


# ---------------------------------------------------------------------------
# Segment classification endpoint (uses RF classifier)
# ---------------------------------------------------------------------------

@api_bp.route("/predict/classify", methods=["POST"])
def classify():
    """
    Predict segment label from basket/behavioural features via Random Forest.

    Request JSON:
    {
        "Recency":              <float>,
        "Frequency":            <float>,
        "Monetary":             <float>,
        "AvgBasketValue":       <float>,
        "AvgBasketSize":        <float>,
        "Volatility":           <float>,
        "FavoriteDay_Encoded":  <int>,
        "Country_Encoded":      <int>
    }
    """
    t0 = time.perf_counter()

    data = request.get_json(silent=True)

    valid, err = validate_classifier_input(data)
    if not valid:
        return jsonify({"error": err}), 400

    try:
        rf = _get_rf()
        X  = build_classifier_features(data)

        pred       = int(rf.predict(X)[0])
        proba      = rf.predict_proba(X)[0].tolist()
        segment    = cluster_to_segment(pred)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify({
            "segment_class":  pred,
            "segment":        segment,
            "probabilities":  proba,
            "input":          data,
            "latency_ms":     latency_ms,
        }), 200

    except Exception as exc:
        logger.exception("Classification failed")
        return jsonify({"error": "Classification failed.", "detail": str(exc)}), 500
