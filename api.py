from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI
from typing import Literal

from pydantic import BaseModel, Field

from src.config import ARTIFACT_PATH
from src.model import load_bundle, train_and_save

app = FastAPI(title="MedPredict AI API", version="1.0.0")


@lru_cache(maxsize=1)
def get_bundle():
    if ARTIFACT_PATH.exists():
        return load_bundle(ARTIFACT_PATH)
    return train_and_save()


class PredictionRequest(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: Literal["female", "male", "Female", "Male"]
    bmi: float = Field(ge=10, le=70)
    children: int = Field(ge=0, le=10)
    smoker: Literal["yes", "no", "Yes", "No"]
    region: Literal["northeast", "northwest", "southeast", "southwest", "Northeast", "Northwest", "Southeast", "Southwest"]


@app.get("/health")
def health():
    bundle = get_bundle()
    return {"status": "ok", "model": bundle.metrics.iloc[0]["Model"]}


@app.post("/predict")
def predict(payload: PredictionRequest):
    row = pd.DataFrame([payload.model_dump()])
    row["sex"] = row["sex"].str.lower()
    row["smoker"] = row["smoker"].str.lower()
    row["region"] = row["region"].str.lower()

    bundle = get_bundle()
    prediction = float(bundle.best_model.predict(row)[0])
    import numpy as np
    q10, q90 = np.quantile(bundle.residuals, [0.10, 0.90])
    low = max(0.0, prediction + float(q10))
    high = max(low, prediction + float(q90))
    return {
        "model": bundle.metrics.iloc[0]["Model"],
        "prediction": round(prediction, 2),
        "prediction_band_80pct": [round(low, 2), round(high, 2)],
    }
