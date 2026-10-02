from __future__ import annotations

import pandas as pd

from .config import CATEGORICAL_FEATURES, DATA_PATH, FEATURES, TARGET
from .validation import validate_dataset


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip().str.lower()
    validate_dataset(df)
    return df
