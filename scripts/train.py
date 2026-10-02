from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import train_and_save

if __name__ == "__main__":
    bundle = train_and_save()
    print("Selected model:", bundle.metrics.iloc[0]["Model"])
    print(bundle.metrics.to_string(index=False))
    print("Saved artifact to artifacts/model_bundle.joblib")
