"""FastAPI inference service for the Titanic classifier.

Loads the @production-aliased model from the MLflow registry at startup and
exposes:
  GET  /health      — liveness check
  GET  /model-info  — name/version/alias of the loaded model
  POST /predict     — survival prediction for one passenger

Run locally from the project root:
    uv run uvicorn app.main:app --reload
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from app.schemas import ModelInfo, PassengerIn, PredictionOut

REGISTERED_MODEL_NAME = "titanic-classifier"
MODEL_ALIAS = "production"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

# MODEL_PATH overrides the registry path. Set by the Docker image so the
# container loads a model baked in at build time instead of hitting MLflow.
MODEL_PATH = os.getenv("MODEL_PATH")

# Process-wide state. Populated by the lifespan startup hook below.
state: dict = {}


def _load_from_local_path(path: str) -> None:
    """Container-mode load: model files baked into the image at build time."""
    print(f"Loading model from local path: {path}")
    state["model"] = mlflow.sklearn.load_model(path)

    metadata_path = Path(path) / "metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        state["version"] = meta.get("version", "unknown")
        state["alias"] = meta.get("alias", "unknown")
    else:
        state["version"] = "unknown"
        state["alias"] = "unknown"

    state["loaded_at"] = datetime.now(timezone.utc).isoformat()
    print(f"Loaded model v{state['version']} (@{state['alias']}) from {path}")


def _load_from_registry() -> None:
    """Dev-mode load: resolve @production via the local MLflow registry."""
    print(f"Loading {MODEL_URI} ...")
    state["model"] = mlflow.sklearn.load_model(MODEL_URI)

    client = MlflowClient()
    version_info = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME, alias=MODEL_ALIAS
    )
    state["version"] = version_info.version
    state["alias"] = MODEL_ALIAS
    state["loaded_at"] = datetime.now(timezone.utc).isoformat()
    print(f"Loaded {REGISTERED_MODEL_NAME} v{version_info.version} (@{MODEL_ALIAS})")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; clean up on shutdown."""
    if MODEL_PATH:
        _load_from_local_path(MODEL_PATH)
    else:
        _load_from_registry()

    yield

    state.clear()


app = FastAPI(
    title="Titanic Classifier API",
    description="Predict Titanic passenger survival from raw features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Cheap liveness check — does the model exist in memory?"""
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    """Return metadata about the model currently serving predictions."""
    return ModelInfo(
        name=REGISTERED_MODEL_NAME,
        version=state["version"],
        alias=state["alias"],
        loaded_at=state["loaded_at"],
    )


@app.post("/predict", response_model=PredictionOut)
def predict(passenger: PassengerIn) -> PredictionOut:
    """Predict survival for a single passenger."""
    row = pd.DataFrame([passenger.model_dump()])
    proba = float(state["model"].predict_proba(row)[0, 1])
    pred = int(state["model"].predict(row)[0])
    return PredictionOut(survived=pred, survival_probability=proba)
