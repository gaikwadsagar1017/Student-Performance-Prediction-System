from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.feature_engineering import FeatureEngineer
from src.config import MODELS_DIR


app = FastAPI(
    title="Student Performance Prediction API",
    description="ML API for predicting student academic performance",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts once
model = joblib.load(MODELS_DIR / "best_model.pkl")
preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")


class StudentInput(BaseModel):
    attendance: float = Field(..., ge=0, le=100)
    study_hours: float = Field(..., ge=0, le=12)
    quiz_score: float = Field(..., ge=0, le=100)
    assignment_score: float = Field(..., ge=0, le=100)
    midterm_score: float = Field(..., ge=0, le=100)
    sleep_hours: float = Field(..., ge=0, le=12)
    lms_login_count: int = Field(..., ge=0, le=100)
    participation: float = Field(..., ge=0, le=100)

    parent_education: str
    internet_access: str
    family_income: str
    extracurricular: str


class PredictionResponse(BaseModel):
    predicted_grade: str
    risk_level: str
    confidence: float
    recommendation: List[str]


def grade_to_risk(grade: str):
    if grade in ["A", "B"]:
        return "Low"
    elif grade == "C":
        return "Medium"
    return "High"


def recommendation_engine(risk: str):
    if risk == "High":
        return [
            "Enroll in tutoring sessions",
            "Weekly academic review",
            "Parent/mentor meeting",
            "Structured study timetable"
        ]

    if risk == "Medium":
        return [
            "Increase study hours",
            "Take weekly mock tests",
            "Complete assignments early",
            "Improve class participation"
        ]

    return [
        "Maintain consistency",
        "Practice advanced questions",
        "Join enrichment activities",
        "Keep regular revision schedule"
    ]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "API running successfully"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(student: StudentInput):
    df = pd.DataFrame([student.dict()])

    df = FeatureEngineer.add_features(df)

    X = preprocessor.transform(df)

    pred_encoded = model.predict(X)[0]

    predicted_grade = label_encoder.inverse_transform(
        [pred_encoded]
    )[0]

    # confidence
    confidence = 0.90

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(max(proba))

    risk = grade_to_risk(predicted_grade)

    recommendation = recommendation_engine(risk)

    return {
        "predicted_grade": predicted_grade,
        "risk_level": risk,
        "confidence": round(confidence, 2),
        "recommendation": recommendation
    }