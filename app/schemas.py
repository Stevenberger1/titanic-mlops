"""Pydantic schemas for the inference API.

These models define the request/response contracts. FastAPI uses them for:
  - automatic JSON validation on incoming requests (422 on bad input),
  - automatic JSON serialization on responses,
  - generating the OpenAPI spec that powers /docs.

Field names match the raw Titanic CSV columns exactly — the preprocessor
inside the registered model was fitted on these exact column names, so
sending a DataFrame with them is what makes the whole pipeline work.
"""

from typing import Literal

from pydantic import BaseModel, Field


class PassengerIn(BaseModel):
    """One passenger's raw features — the request body for /predict."""

    Pclass: Literal[1, 2, 3] = Field(
        ..., description="Ticket class: 1=upper, 2=middle, 3=lower"
    )
    Sex: Literal["male", "female"] = Field(..., description="Passenger sex")
    Age: float = Field(..., ge=0, le=120, description="Age in years")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Fare: float = Field(..., ge=0, description="Ticket fare in pounds")
    Embarked: Literal["S", "C", "Q"] = Field(
        ..., description="Port of embarkation: S=Southampton, C=Cherbourg, Q=Queenstown"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pclass": 3,
                "Sex": "male",
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked": "S",
            }
        }
    }


class PredictionOut(BaseModel):
    """Model output for one passenger — the response body for /predict."""

    survived: int = Field(..., description="Predicted class: 1 = survived, 0 = did not")
    survival_probability: float = Field(
        ..., ge=0, le=1, description="Probability of survival in [0, 1]"
    )


class ModelInfo(BaseModel):
    """Metadata about the model currently loaded by the API."""

    name: str = Field(..., description="Registered model name")
    version: str = Field(..., description="Registry version that resolved at startup")
    alias: str = Field(..., description="Alias the API loaded from (e.g. 'production')")
    loaded_at: str = Field(
        ..., description="UTC timestamp of when the API loaded the model"
    )
