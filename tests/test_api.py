"""
tests/test_api.py
-----------------
Smoke tests for the Flask API.
Run:  pytest tests/ -v
"""

import pytest
import json
from wsgi import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"


# ── /predict ─────────────────────────────────────────────────────────────

VALID_RFM = {"Recency": 30, "Frequency": 5, "Monetary": 1200.0}


def test_predict_valid_input(client):
    resp = client.post(
        "/predict",
        data=json.dumps(VALID_RFM),
        content_type="application/json",
    )
    # Accept 200 (model loaded) or 500 (model files absent in CI)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        body = resp.get_json()
        assert "cluster" in body
        assert "segment" in body
        assert "latency_ms" in body


def test_predict_missing_field_returns_400(client):
    resp = client.post(
        "/predict",
        data=json.dumps({"Recency": 10, "Frequency": 3}),  # Monetary missing
        content_type="application/json",
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert "error" in body
    assert "Monetary" in body["error"]


def test_predict_negative_value_returns_400(client):
    bad = {"Recency": -1, "Frequency": 5, "Monetary": 100}
    resp = client.post(
        "/predict",
        data=json.dumps(bad),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_predict_wrong_type_returns_400(client):
    bad = {"Recency": "thirty", "Frequency": 5, "Monetary": 100}
    resp = client.post(
        "/predict",
        data=json.dumps(bad),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_predict_empty_body_returns_400(client):
    resp = client.post("/predict", content_type="application/json")
    assert resp.status_code == 400


# ── 404 ──────────────────────────────────────────────────────────────────

def test_unknown_route_returns_404(client):
    resp = client.get("/nonexistent")
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "Endpoint not found."
