import json
import joblib
from pathlib import Path


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved -> {path}")


def load_model(path):
    path = Path(path)
    model = joblib.load(path)
    print(f"[INFO] Model loaded <- {path}")
    return model


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"[INFO] JSON saved -> {path}")