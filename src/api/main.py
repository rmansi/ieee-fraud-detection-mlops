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

MODEL_NAME = "fraud-xgboost"
MODEL_VERSION = "2"
state = {}  # holds the model and its metadata while the API runs


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    state["model"] = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

    # find the training run that produced this model version, and load its metadata
    run_id = mlflow.MlflowClient().get_model_version(MODEL_NAME, MODEL_VERSION).run_id
    state["columns"] = mlflow.artifacts.load_dict(f"runs:/{run_id}/columns.json")["columns"]
    state["categories"] = mlflow.artifacts.load_dict(f"runs:/{run_id}/categories.json")
    yield
    state.clear()


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


class Transaction(BaseModel):
    features: dict[str, float | str | None]


def prepare(features: dict) -> pd.DataFrame:
    """Turn one request into a one-row DataFrame shaped exactly like training data."""
    columns, categories = state["columns"], state["categories"]

    unknown = set(features) - set(columns)
    if unknown:
        raise HTTPException(422, f"Unknown features: {sorted(unknown)[:10]}")

    computed = {"n_missing", "has_identity"} & set(features)
    if computed:
        raise HTTPException(422, f"These features are computed by the API: {sorted(computed)}")

    # every training column, in training order; anything not sent becomes blank
    df = pd.DataFrame([{c: features.get(c) for c in columns}], columns=columns)

    for col in columns:
        if col in categories:
            # use the TRAINING category list; unseen values become blank
            df[col] = pd.Categorical(df[col], categories=categories[col])
        else:
            try:
                df[col] = pd.to_numeric(df[col]).astype("float32")
            except (ValueError, TypeError):
                raise HTTPException(422, f"Feature '{col}' must be a number")

    # engineered features: computed here with the same code as training
    identity_cols = [c for c in columns if c.startswith("id_") or c in ("DeviceType", "DeviceInfo")]
    df["has_identity"] = df[identity_cols].notna().any(axis=1).astype("int8")
    df = add_features(df)
    return df


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state,
            "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(tx: Transaction):
    df = prepare(tx.features)
    prob = float(state["model"].predict_proba(df)[0, 1])
    return {"fraud_probability": round(prob, 6), "model_version": MODEL_VERSION}