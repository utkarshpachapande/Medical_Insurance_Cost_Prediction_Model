from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "insurance.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "model_bundle.joblib"

TARGET = "charges"
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42

MODEL_NAMES = [
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting",
    "Extra Trees",
]
