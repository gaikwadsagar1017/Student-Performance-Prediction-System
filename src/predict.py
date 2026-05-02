import joblib
import pandas as pd

from src.feature_engineering import FeatureEngineer
from src.config import MODELS_DIR


class StudentPredictor:
    def __init__(self):
        self.model = joblib.load(
            MODELS_DIR / "best_model.pkl"
        )

        self.preprocessor = joblib.load(
            MODELS_DIR / "preprocessor.pkl"
        )

        self.label_encoder = joblib.load(
            MODELS_DIR / "label_encoder.pkl"
        )

    @staticmethod
    def grade_to_risk(grade):
        if grade in ["A", "B"]:
            return "Low"
        elif grade == "C":
            return "Medium"
        return "High"

    def predict(self, student_dict):
        df = pd.DataFrame([student_dict])

        df = FeatureEngineer.add_features(df)

        X = self.preprocessor.transform(df)

        pred_encoded = self.model.predict(X)[0]

        pred_grade = self.label_encoder.inverse_transform(
            [pred_encoded]
        )[0]

        risk = self.grade_to_risk(pred_grade)

        result = {
            "predicted_grade": pred_grade,
            "risk_level": risk
        }

        return result