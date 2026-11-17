import json

import mlflow.xgboost
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from src.api import main

CATS = ["mastercard", "visa"]


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Train a tiny model, export it like export_model.py does, start the API on it."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "TransactionAmt": rng.uniform(1, 500, n).astype("float32"),
        "card4": pd.Categorical(rng.choice(CATS, n), categories=CATS),
        "has_identity": rng.integers(0, 2, n).astype("int8"),
        "n_missing": rng.integers(0, 3, n).astype("int16"),
    })
    y = rng.integers(0, 2, n)
    model = XGBClassifier(n_estimators=5, max_depth=2, tree_method="hist", enable_categorical=True)
    model.fit(X, y)

    mlflow.xgboost.save_model(model, str(tmp_path / "model"))
    (tmp_path / "columns.json").write_text(json.dumps({"columns": list(X.columns)}))
    (tmp_path / "categories.json").write_text(json.dumps({"card4": CATS}))
    (tmp_path / "version.txt").write_text("test")

    monkeypatch.setattr(main, "MODEL_DIR", tmp_path)
    with TestClient(main.app) as client:   # "with" runs the startup code (lifespan)
        yield client, model


def test_health(api):
    client, _ = api
    body = client.get("/health").json()
    assert body["model_loaded"] is True and body["model_version"] == "test"


def test_predict_matches_model_directly(api):
    """Training-serving consistency: API output == model on a correctly prepared row."""
    client, model = api
    body = client.post("/predict", json={"features": {"TransactionAmt": 100.0, "card4": "visa"}}).json()

    expected_row = pd.DataFrame({
        "TransactionAmt": np.array([100.0], dtype="float32"),
        "card4": pd.Categorical(["visa"], categories=CATS),
        "has_identity": np.array([0], dtype="int8"),
        "n_missing": np.array([0], dtype="int16"),
    })
    expected = round(float(model.predict_proba(expected_row)[0, 1]), 6)
    assert body["fraud_probability"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("features", [
    {"TransactonAmt": 1.0},       # typo -> unknown feature
    {"n_missing": 5},             # computed by the API
    {"TransactionAmt": "hello"},  # text in a number column
])
def test_bad_requests_are_rejected(api, features):
    client, _ = api
    assert client.post("/predict", json={"features": features}).status_code == 422


def test_unseen_category_does_not_crash(api):
    client, _ = api
    response = client.post("/predict", json={"features": {"card4": "discover"}})
    assert response.status_code == 200


def test_metrics_endpoint_reports_predictions(api):
    client, _ = api
    client.post("/predict", json={"features": {"TransactionAmt": 5.0}})
    assert "fraud_predictions_total" in client.get("/metrics").text