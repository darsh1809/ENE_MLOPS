"""
utils.py
--------
Input validation and feature-engineering helpers for the /predict endpoint.
"""

from typing import Any, Dict, Tuple

import numpy as np

# Required fields and their expected types for /predict
REQUIRED_FIELDS: Dict[str, type] = {
    "Recency":  (int, float),
    "Frequency": (int, float),
    "Monetary":  (int, float),
}

# Optional classifier fields
CLASSIFIER_FIELDS: Dict[str, type] = {
    "AvgBasketValue":  (int, float),
    "AvgBasketSize":   (int, float),
    "Volatility":      (int, float),
    "FavoriteDay_Encoded": int,
    "Country_Encoded": int,
}

SEGMENT_LABELS = {
    0: "Champions",
    1: "Potential Loyalists",
    2: "At Risk",
    3: "Lost",
}


def validate_rfm_input(data: Any) -> Tuple[bool, str]:
    """
    Validate incoming JSON for RFM-based cluster prediction.
    Returns (is_valid, error_message).
    """
    if not isinstance(data, dict):
        return False, "Request body must be a JSON object."

    for field, expected_types in REQUIRED_FIELDS.items():
        if field not in data:
            return False, f"Missing required field: '{field}'."
        if not isinstance(data[field], expected_types):
            return False, (
                f"Field '{field}' must be numeric, "
                f"got {type(data[field]).__name__}."
            )
        if data[field] < 0:
            return False, f"Field '{field}' must be non-negative."

    return True, ""


def validate_classifier_input(data: Any) -> Tuple[bool, str]:
    """
    Validate incoming JSON for segment-label classification.
    Returns (is_valid, error_message).
    """
    if not isinstance(data, dict):
        return False, "Request body must be a JSON object."

    all_fields = {**REQUIRED_FIELDS, **CLASSIFIER_FIELDS}
    for field, expected_types in all_fields.items():
        if field not in data:
            return False, f"Missing required field: '{field}'."
        if not isinstance(data[field], expected_types):
            return False, (
                f"Field '{field}' must be numeric, "
                f"got {type(data[field]).__name__}."
            )

    return True, ""


def build_rfm_features(data: Dict) -> np.ndarray:
    """Extract and return RFM feature array (1 × 3)."""
    return np.array([[data["Recency"], data["Frequency"], data["Monetary"]]])


def build_classifier_features(data: Dict) -> np.ndarray:
    """Extract and return classifier feature array (1 × 5)."""
    return np.array([[
        data["AvgBasketValue"],
        data["AvgBasketSize"],
        data["Volatility"],
        data["FavoriteDay_Encoded"],
        data["Country_Encoded"],
    ]])


def cluster_to_segment(cluster_id: int) -> str:
    """Map a raw KMeans cluster integer to a human-readable segment label."""
    return SEGMENT_LABELS.get(int(cluster_id), f"Cluster_{cluster_id}")
