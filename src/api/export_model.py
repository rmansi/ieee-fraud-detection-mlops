"""Copy a registered model and its metadata into model_store/ for packaging.

Run from the project root:
    python -m src.api.export_model
"""
import shutil
from pathlib import Path

import mlflow

MODEL_NAME = "fraud-xgboost"
MODEL_VERSION = "2"
OUT = Path("model_store")


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    if OUT.exists():
        shutil.rmtree(OUT)   # start clean so no old files are left behind
    OUT.mkdir()

    run_id = mlflow.MlflowClient().get_model_version(MODEL_NAME, MODEL_VERSION).run_id

    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}", dst_path=str(OUT / "model")
    )
    for name in ["columns.json", "categories.json"]:
        mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{name}", dst_path=str(OUT))
    (OUT / "version.txt").write_text(MODEL_VERSION)

    print("Exported to", OUT.resolve())
    for p in sorted(OUT.rglob("*")):
        print("  ", p.relative_to(OUT))


if __name__ == "__main__":
    main()