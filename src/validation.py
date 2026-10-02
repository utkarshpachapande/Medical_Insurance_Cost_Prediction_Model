from __future__ import annotations

import pandas as pd

from .config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


def validate_dataset(df: pd.DataFrame) -> None:
    missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Dataset contains missing values in required fields")
    if (df[NUMERIC_FEATURES] < 0).any().any():
        raise ValueError("Numeric feature values cannot be negative")
    if not set(df["sex"].unique()) <= {"female", "male"}:
        raise ValueError("Unexpected values in sex column")
    if not set(df["smoker"].unique()) <= {"yes", "no"}:
        raise ValueError("Unexpected values in smoker column")
    if not set(df["region"].unique()) <= {"northeast", "northwest", "southeast", "southwest"}:
        raise ValueError("Unexpected values in region column")
