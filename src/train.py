"""Train an XGBoost classifier on the Titanic dataset.

Usage:
    uv run python -m src.train --config configs/xgb.yaml

Outputs (under cfg.paths.artifact_dir, default 'models/'):
    artifact.pkl   — joblib bundle: {model, preprocessor, feature_columns, ...}
    metrics.json   — test-set accuracy, precision, recall, f1, roc_auc

The artifact is what app/main.py (FastAPI) will load at startup. Saving the
fitted preprocessor INSIDE the artifact is how we prevent training/serving
skew: inference uses the exact same fitted transformer that training used.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from xgboost import XGBClassifier

from src.config import AppConfig, load_config
from src.preprocessing import build_preprocessor, split_xy

EXPERIMENT_NAME = "titanic-xgb"
REGISTERED_MODEL_NAME = "titanic-classifier"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. Returns a namespace with .config (Path)."""
    parser = argparse.ArgumentParser(
        description="Train the Titanic XGBoost classifier."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML training config (e.g. configs/xgb.yaml).",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help=(
            "Register the trained model in the MLflow Model Registry as a "
            f"new version of '{REGISTERED_MODEL_NAME}'. Use this only when "
            "the run produces a model that's a deployment candidate."
        ),
    )
    return parser.parse_args()


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Compute standard binary-classification metrics on the test set."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def train(cfg: AppConfig, register: bool = False) -> None:
    """Run the full training pipeline.

    Assumes an MLflow run is already active (started by main()).

    If register=True, the trained model is also registered in the MLflow
    Model Registry as a new version of REGISTERED_MODEL_NAME.
    """

    # 0. Log hyperparameters to the active MLflow run
    mlflow.log_params(cfg.model.params.model_dump())
    mlflow.log_param("test_size", cfg.training.test_size)
    mlflow.log_param("random_state", cfg.training.random_state)
    mlflow.log_param("stratify", cfg.training.stratify)
    mlflow.log_param("raw_path", str(cfg.data.raw_path))
    mlflow.log_param("n_numeric_features", len(cfg.features.numeric))
    mlflow.log_param("n_categorical_features", len(cfg.features.categorical))

    # 1. Load raw data
    print(f"Loading data from {cfg.data.raw_path} ...")
    df = pd.read_csv(cfg.data.raw_path)
    print(f"  loaded {len(df)} rows, {len(df.columns)} columns")

    # 2. Split into features (X) and target (y)
    X, y = split_xy(
        df,
        target_column=cfg.data.target_column,
        id_column=cfg.data.id_column,
        drop_columns=cfg.features.drop,
    )

    # 3. Train/test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=y if cfg.training.stratify else None,
    )
    print(f"  train size: {len(X_train)} | test size: {len(X_test)}")

    # 4. Build preprocessor, FIT on train only, then transform both
    preprocessor = build_preprocessor(cfg.features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 5. Train the model
    print(f"Training XGBoost with params: {cfg.model.params.model_dump()}")
    model = XGBClassifier(**cfg.model.params.model_dump())
    model.fit(X_train_processed, y_train)

    # 6. Evaluate on the held-out test set
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    print("Test-set metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Log metrics to the active MLflow run
    mlflow.log_metrics(metrics)

    # 7. Build the artifact bundle (everything app/main.py will need)
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": list(X.columns),
        "target_classes": model.classes_.tolist(),
        "training_stats": {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        "model_version": "1.0.0",
    }

    # 8. Save artifact and metrics to disk
    artifact_dir = cfg.paths.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = artifact_dir / cfg.paths.artifact_name
    metrics_path = artifact_dir / cfg.paths.metrics_name

    joblib.dump(artifact, artifact_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nArtifact saved to: {artifact_path}")
    print(f"Metrics saved to:  {metrics_path}")

    # Also log them as MLflow artifacts (a versioned per-run copy)
    mlflow.log_artifact(str(artifact_path))
    mlflow.log_artifact(str(metrics_path))

    # 9. Log the model via the sklearn flavor — this is what's required for
    #    the MLflow 3.x Model Registry. We bundle the fitted preprocessor
    #    and the trained classifier into a single sklearn Pipeline so that
    #    inference uses the exact same preprocessing as training.
    inference_pipeline = SklearnPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    mlflow.sklearn.log_model(
        sk_model=inference_pipeline,
        name="model",
        registered_model_name=REGISTERED_MODEL_NAME if register else None,
    )
    if register:
        print(f"\nRegistered new version of '{REGISTERED_MODEL_NAME}'")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = (
        f"depth{cfg.model.params.max_depth}"
        f"-est{cfg.model.params.n_estimators}"
        f"-lr{cfg.model.params.learning_rate}"
    )
    with mlflow.start_run(run_name=run_name) as run:
        print(f"MLflow run id: {run.info.run_id}")
        print(f"MLflow run name: {run_name}")
        mlflow.log_artifact(str(args.config))
        train(cfg, register=args.register)


if __name__ == "__main__":
    main()
