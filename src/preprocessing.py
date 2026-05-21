"""Preprocessing pipeline for the Titanic dataset.

This module builds a sklearn ColumnTransformer that:
  1. Imputes missing values (median for numeric, most-frequent for categorical)
  2. One-hot encodes categorical columns
  3. Outputs a numeric matrix ready for the model

The SAME fitted pipeline is used in train.py AND app/main.py. This is how we
prevent training/serving skew: the imputer remembers the training medians,
the encoder remembers the training categories, and those exact same values
are reused at inference time.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import FeaturesConfig


def split_xy(
    df: pd.DataFrame,
    target_column: str,
    id_column: str,
    drop_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) from target (y); remove id and dropped columns.

    Used in train.py once the raw CSV is loaded. The id column is removed
    because PassengerId is just a row number — it has no predictive value
    and including it would let the model 'memorize' the training set.
    """
    columns_to_drop = [target_column, id_column, *drop_columns]
    X = df.drop(columns=columns_to_drop, errors="ignore")
    y = df[target_column]
    return X, y


def build_preprocessor(features: FeaturesConfig) -> ColumnTransformer:
    """Build an UNFITTED preprocessor from the features config.

    The returned object knows WHAT to do but has not yet learned the
    parameters (medians, categories). Call .fit(X_train) on it before use.

    Numeric pipeline:
        SimpleImputer(strategy="median")  — fills missing values

    Categorical pipeline:
        SimpleImputer(strategy="most_frequent")  — fills missing values
        OneHotEncoder(handle_unknown="ignore")   — turns labels into 0/1 cols
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, features.numeric),
            ("categorical", categorical_pipeline, features.categorical),
        ],
        remainder="drop",
    )
#