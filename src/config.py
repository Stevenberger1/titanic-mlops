"""Application configuration: experiment YAML + runtime environment settings.

This module has two responsibilities:

1. Load and validate a YAML training config (e.g. configs/xgb.yaml).
   Used by train.py and any code that needs experiment parameters.
2. Load and validate runtime settings from .env.
   Used by the FastAPI service and anything that varies per environment.

YAML  = experiment definition (committed to git, identical across machines)
.env  = runtime environment    (gitignored, may differ per machine/deployment)
"""
#uv run python -c "from src.config import load_config, settings; cfg = load_config('configs/xgb.yaml'); print('YAML loaded:'); print(cfg.model_dump_json(indent=2)); print(); print('Settings:'); print(settings.model_dump_json(indent=2))"

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# 1. YAML experiment configuration
# ─────────────────────────────────────────────────────────────────────────────


class ModelParams(BaseModel):
    """Hyperparameters forwarded as kwargs to XGBClassifier(...)."""

    n_estimators: int = Field(gt=0)
    max_depth: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    objective: str
    eval_metric: str
    random_state: int


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    type: str
    params: ModelParams


class TrainingConfig(BaseModel):
    test_size: float = Field(gt=0, lt=1)
    random_state: int
    stratify: bool


class DataConfig(BaseModel):
    raw_path: Path
    target_column: str
    id_column: str


class FeaturesConfig(BaseModel):
    numeric: list[str]
    categorical: list[str]
    drop: list[str]


class PathsConfig(BaseModel):
    artifact_dir: Path
    artifact_name: str
    metrics_name: str


class AppConfig(BaseModel):
    """Top-level experiment config — mirrors the shape of configs/xgb.yaml."""

    model_config = ConfigDict(protected_namespaces=())

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    features: FeaturesConfig
    paths: PathsConfig


def load_config(config_path: str | Path) -> AppConfig:
    """Load a YAML training config and validate it against AppConfig.

    Raises pydantic.ValidationError if the YAML is missing fields, has wrong
    types, or fails any constraint (e.g. test_size outside (0, 1)).
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Runtime environment settings (.env)
# ─────────────────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a .env file.

    Used by the FastAPI service and any long-running process. Distinct from
    the YAML config because these values can legitimately differ between
    your laptop and a deployed container.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: str = "models/artifact.pkl"
    log_level: str = "INFO"
    api_key: str | None = None


settings = Settings()
