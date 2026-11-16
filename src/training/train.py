"""Train a fraud model and log it to MLflow.

Run from the project root:
    python -m src.training.train --model xgboost
    python -m src.training.train --model logreg
"""
import argparse
import time

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.build_features import build_dataset
from src.training.evaluate import evaluate

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
}


def build_model(name, X_train):
    """Return (model, columns to use, params to log) for the chosen model."""
    if name == "logreg":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )
        cols = X_train.select_dtypes("number").columns
        params = {"model": "logistic_regression", "features": "numeric_only",
                  "imputer": "median", "max_iter": 1000}
    else:
        model = XGBClassifier(**XGB_PARAMS, tree_method="hist",
                              enable_categorical=True, n_jobs=-1, random_state=42)
        cols = X_train.columns
        params = {"model": "xgboost", "features": "all",
                  "imputer": "none", **XGB_PARAMS}
    return model, cols, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "xgboost"], default="xgboost")
    args = parser.parse_args()

    print("Building dataset ...")
    X_train, y_train, X_valid, y_valid = build_dataset()
    model, cols, params = build_model(args.model, X_train)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name=args.model):
        mlflow.log_params(params)

        print(f"Training {args.model} ...")
        start = time.time()
        model.fit(X_train[cols], y_train)
        mlflow.log_metric("train_seconds", round(time.time() - start, 1))

        scores = model.predict_proba(X_valid[cols])[:, 1]
        metrics = evaluate(y_valid, scores)
        mlflow.log_metrics(metrics)
        print(metrics)

        # save which columns the model expects, for the API later
        mlflow.log_dict({"columns": list(cols)}, "columns.json")

        if args.model == "xgboost":
            mlflow.xgboost.log_model(model, name="model",
                                     registered_model_name="fraud-xgboost")
        else:
            mlflow.sklearn.log_model(model, name="model")

if __name__ == "__main__":
    main()