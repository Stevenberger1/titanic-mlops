# titanic-mlops

A hands-on MLOps capstone: a production-shaped pipeline for the Kaggle Titanic
survival classification task. The goal is to take a single Jupyter-notebook
model and turn it into a deployable, observable, reproducible system step by
step.

## Stack

- **[uv](https://docs.astral.sh/uv/)** — dependency management and Python environment.
- **[Pydantic](https://docs.pydantic.dev/)** + **PyYAML** — typed validation of YAML training configs.
- **scikit-learn** (`Pipeline`, `ColumnTransformer`) — preprocessing that travels with the model.
- **[XGBoost](https://xgboost.readthedocs.io/)** — the classifier.
- **[MLflow](https://mlflow.org/)** — experiment tracking; later, model registry.
- **[ruff](https://docs.astral.sh/ruff/)** + **pre-commit** — linting and formatting on every commit.

## Project structure

```
.
├── configs/                   # YAML training configs, one per experiment variant
│   ├── xgb.yaml               # baseline
│   └── xgb-depth8.yaml        # deeper-trees variant
├── data/
│   └── raw/                   # raw datasets (gitignored contents)
├── models/                    # latest trained artifact (gitignored)
├── mlruns/                    # MLflow per-run storage (gitignored)
├── src/
│   ├── config.py              # Pydantic schemas + YAML loader
│   ├── preprocessing.py       # ColumnTransformer build helpers
│   └── train.py               # training entry point
├── .pre-commit-config.yaml
├── pyproject.toml             # project + dependencies
└── README.md
```

## Quickstart

```bash
# 1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
# 2. Sync the environment
uv sync

# 3. Install git pre-commit hooks
uv run pre-commit install

# 4. Place the dataset at data/raw/Titanic-Dataset.csv
#    Download from https://www.kaggle.com/c/titanic

# 5. Train the baseline
uv run python -m src.train --config configs/xgb.yaml
```

Each run produces:

- `models/artifact.pkl` — joblib bundle (model + fitted preprocessor + metadata) used by inference code.
- A new MLflow run under the `titanic-xgb` experiment, with params, metrics, and the YAML config logged.

## Experiment tracking

Launch the MLflow UI:

```bash
uv run mlflow ui --workers 1
```

Open `http://localhost:5000`, switch the sidebar toggle to **Model training**, click into
`titanic-xgb`, and you'll see every run with its params, metrics, and artifacts. Tick two
runs and hit **Compare** for a side-by-side diff of params and metrics.

## Adding an experiment variant

1. Copy `configs/xgb.yaml` to `configs/xgb-<descriptor>.yaml` (e.g., `xgb-tiny.yaml`).
2. Add a header comment block: what changed vs baseline, the hypothesis, expected outcome.
3. Edit the field(s) you want to vary.
4. Train against the new config:

   ```bash
   uv run python -m src.train --config configs/xgb-<descriptor>.yaml
   ```

MLflow logs the variant's YAML file alongside its run, so the run is reproducible by
downloading that YAML and re-pointing training at it.

## Configuration

Training config lives in YAML under `configs/`. The schema is enforced at load time by
the Pydantic models in `src/config.py` — if a required field is missing or a value
violates its constraint (e.g., `test_size` must be in `(0, 1)`), training fails immediately
with a descriptive error.

Runtime environment (model path, log level, API key) is loaded from a `.env` file by
`pydantic-settings`. See `.env.example` for the template.

## Development

Pre-commit hooks (`.pre-commit-config.yaml`):

- `ruff` — Python lint with auto-fix.
- `ruff-format` — auto-format Python source.
- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files` —
  language-agnostic hygiene.

Run hooks manually against every file (not just staged changes):

```bash
uv run pre-commit run --all-files
```

## Project status

- ✅ **Phase 1** — Reproducible training pipeline (uv, Pydantic configs, sklearn Pipeline, XGBoost, joblib artifact).
- ✅ **Phase 2** — Experiment tracking with MLflow (params, metrics, artifacts logged per run).
- ⏳ **Phase 3** — MLflow Model Registry (versioned, promotable models).
- ⏳ **Phase 4** — FastAPI inference service.
- ⏳ **Phase 5** — Docker.
- ⏳ **Phase 6** — CI/CD + tests.
- ⏳ **Phase 7** — Cloud deployment.
- ⏳ **Phase 8** — Drift monitoring.
- ⏳ **Phase 9** — Dashboards.
- ⏳ **Phase 10** — Final polish.
