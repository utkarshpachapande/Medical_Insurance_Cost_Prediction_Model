from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES


def scenario_impacts(model, row: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Approximate local feature impacts by one-at-a-time counterfactual perturbation.

    This is deliberately model-agnostic and gives a clear user-facing explanation:
    positive values mean the feature raises the estimate relative to a reference profile.
    """
    base_pred = float(model.predict(row)[0])
    results = []

    reference = baseline.iloc[0].to_dict()
    for feature in FEATURES:
        modified = row.copy()
        modified[feature] = reference[feature]
        counter = float(model.predict(modified)[0])
        results.append({
            "Feature": feature,
            "Impact ($)": base_pred - counter,
        })

    return pd.DataFrame(results).sort_values("Impact ($)", key=lambda s: s.abs(), ascending=False)


def format_feature_label(name: str) -> str:
    labels = {
        "age": "Age",
        "bmi": "BMI",
        "children": "Children",
        "sex": "Gender",
        "smoker": "Smoking status",
        "region": "Region",
    }
    return labels.get(name, name.replace("_", " ").title())
