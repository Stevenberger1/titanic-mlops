"""Load the production Titanic model from the MLflow registry and predict.

Run from project root:
    uv run python scripts/predict_from_registry.py

This is a Phase-3 sanity check: it proves the registered model can be loaded
by alias (not by run id), which is exactly the indirection Phase 4 needs.
"""

import mlflow
import pandas as pd

MODEL_URI = "models:/titanic-classifier@production"


def main() -> None:
    print(f"Loading {MODEL_URI} ...")
    model = mlflow.sklearn.load_model(MODEL_URI)
    print(f"Loaded pipeline: {model}")

    # Build a single passenger as a one-row DataFrame. The columns and dtypes
    # must match what the preprocessor was fitted on during training.
    passenger = pd.DataFrame(
        [
            {
                "Pclass": 3,
                "Sex": "male",
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked": "S",
            }
        ]
    )

    # The loaded object IS the full Pipeline (preprocessor + classifier).
    # Calling .predict_proba runs the raw row through the SAME preprocessor
    # that was fitted at training time, then through XGBoost — no manual
    # feature engineering needed here.
    proba = model.predict_proba(passenger)[0, 1]
    pred = int(model.predict(passenger)[0])

    print(f"\nInput: {passenger.iloc[0].to_dict()}")
    print(f"Predicted survival probability: {proba:.4f}")
    print(f"Predicted class (1=survived): {pred}")


if __name__ == "__main__":
    main()
