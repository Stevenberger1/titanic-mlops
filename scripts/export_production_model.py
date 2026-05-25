"""Export the @production version of the registered model to a flat folder.

Why this exists:
    Our local MLflow tracking is file-based, so meta.yaml files contain
    absolute paths from the dev machine. Those paths don't exist inside
    a Docker container. To ship a self-contained image, we resolve the
    @production alias HERE (on the dev machine where the registry works)
    and copy the model artifacts to ./models/exported/, which the Docker
    build will then copy into the image.

Usage:
    uv run python scripts/export_production_model.py

Output:
    models/exported/                 — the relocatable MLmodel directory
    models/exported/metadata.json    — version + alias + export timestamp
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "titanic-classifier"
MODEL_ALIAS = "production"
EXPORT_DIR = Path("models/exported")


def main() -> None:
    client = MlflowClient()
    version_info = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME, alias=MODEL_ALIAS
    )
    print(
        f"Resolved {REGISTERED_MODEL_NAME}@{MODEL_ALIAS} -> "
        f"version {version_info.version} (run {version_info.run_id})"
    )

    if EXPORT_DIR.exists():
        print(f"Removing previous {EXPORT_DIR} ...")
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # download_artifacts handles the source URI (which points at the
    # registry-backed location) and copies the model files into dst_path.
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=version_info.source,
        dst_path=str(EXPORT_DIR),
    )
    print(f"Model artifacts written to: {local_path}")

    # The download lands inside a subfolder named after the artifact path
    # used at log_model time (we used name='model'). Flatten so the model
    # is directly at models/exported/.
    nested = Path(local_path)
    if nested != EXPORT_DIR and nested.is_dir():
        for child in nested.iterdir():
            shutil.move(str(child), str(EXPORT_DIR / child.name))
        nested.rmdir()
        print(f"Flattened contents into {EXPORT_DIR}")

    metadata = {
        "name": REGISTERED_MODEL_NAME,
        "version": version_info.version,
        "alias": MODEL_ALIAS,
        "run_id": version_info.run_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (EXPORT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Metadata written to: {EXPORT_DIR / 'metadata.json'}")
    print("\nDone. Ready for: docker build .")


if __name__ == "__main__":
    main()
