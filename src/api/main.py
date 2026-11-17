"""Fraud detection API.

Run from the project root:
    uvicorn src.api.main:app --reload
"""
from contextlib import asynccontextmanager

import mlflow
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.features.build_features import add_features
import time
from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
import numpy as np
import json
import os
from pathlib import Path

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model_store"))
state = {}  # holds the model and its metadata while the API runs
PREDICTIONS = Counter("fraud_predictions_total", "Predictions served")
REJECTED = Counter("fraud_requests_rejected_total", "Requests rejected by input checks")
LATENCY = Histogram("fraud_prediction_latency_seconds", "Time to prepare data and predict")
SCORES = Histogram(
    "fraud_probability",
    "Distribution of predicted fraud probabilities",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # everything comes from one exported folder, the same locally and in Docker
    state["model"] = mlflow.xgboost.load_model(str(MODEL_DIR / "model"))
    state["model"].set_params(n_jobs=1)   # one row doesn't need many threads
    state["columns"] = json.loads((MODEL_DIR / "columns.json").read_text())["columns"]
    state["categories"] = json.loads((MODEL_DIR / "categories.json").read_text())
    state["version"] = (MODEL_DIR / "version.txt").read_text().strip()

    state["cat_dtypes"] = {
        col: pd.CategoricalDtype(cats) for col, cats in state["categories"].items()
    }
    yield
    state.clear()


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


class Transaction(BaseModel):
    features: dict[str, float | str | None]


def prepare(features: dict) -> pd.DataFrame:
    """Turn one request into a one-row DataFrame shaped exactly like training data."""
    columns, cat_dtypes = state["columns"], state["cat_dtypes"]

    unknown = set(features) - set(columns)
    if unknown:
        raise HTTPException(422, f"Unknown features: {sorted(unknown)[:10]}")
    computed = {"n_missing", "has_identity"} & set(features)
    if computed:
        raise HTTPException(422, f"These features are computed by the API: {sorted(computed)}")

    # build every column's value first (cheap), then create the table once
    data = {}
    for col in columns:
        value = features.get(col)
        if col in cat_dtypes:
            # training category list; unseen values become blank explicitly
            if value not in cat_dtypes[col].categories:
                value = None
            data[col] = pd.Series([value], dtype=cat_dtypes[col])
        else:
            try:
                data[col] = np.array([np.nan if value is None else value], dtype="float32")
            except (ValueError, TypeError):
                raise HTTPException(422, f"Feature '{col}' must be a number")
    df = pd.DataFrame(data)

    # engineered features: computed here with the same code as training
    identity_cols = [c for c in columns if c.startswith("id_") or c in ("DeviceType", "DeviceInfo")]
    df["has_identity"] = df[identity_cols].notna().any(axis=1).astype("int8")
    df = add_features(df)
    return df


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state,
            "model_version": state["version"]}


@app.post("/predict")
def predict(tx: Transaction):
    start = time.perf_counter()
    try:
        df = prepare(tx.features)
    except HTTPException:
        REJECTED.inc()
        raise
    prob = float(state["model"].predict_proba(df)[0, 1])

    LATENCY.observe(time.perf_counter() - start)
    PREDICTIONS.inc()
    SCORES.observe(prob)
    return {"fraud_probability": round(prob, 6), "model_version": state["version"]}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)