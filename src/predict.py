import pandas as pd
import joblib
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model path
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

# Load model safely
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


def predict(df: pd.DataFrame) -> pd.Series:
    """
    Predict risk probability for input features
    """
    return model.predict_proba(df)[:, 1]


if __name__ == "__main__":
    print("Model loaded successfully from:", MODEL_PATH)
