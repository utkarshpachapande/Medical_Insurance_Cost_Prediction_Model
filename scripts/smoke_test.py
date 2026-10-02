from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data import load_data
from src.model import load_bundle, train_and_save


def main():
    df = load_data()
    bundle = train_and_save()
    row = pd.DataFrame([{
        "age": 33,
        "bmi": 27.5,
        "children": 1,
        "sex": "female",
        "smoker": "no",
        "region": "northwest",
    }])
    pred = float(bundle.best_model.predict(row)[0])
    assert pred >= 0
    assert bundle.metrics.iloc[0]["CV RMSE ($)"] > 0
    assert Path(ROOT / "artifacts" / "model_bundle.joblib").exists()
    print(f"Smoke test passed: {len(df)} rows, selected={bundle.metrics.iloc[0]['Model']}, prediction=${pred:,.2f}")


if __name__ == "__main__":
    main()
