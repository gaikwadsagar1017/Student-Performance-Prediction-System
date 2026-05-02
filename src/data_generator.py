import numpy as np
import pandas as pd

from src.config import SYNTHETIC_DIR


class StudentDataGenerator:
    def __init__(self, n_samples=50000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(self.random_state)

    def _grade_from_score(self, score):
        if score >= 85:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 55:
            return "C"
        elif score >= 40:
            return "D"
        return "F"

    def _risk_from_grade(self, grade):
        if grade == "A":
            return "Low"
        elif grade == "B":
            return "Low"
        elif grade == "C":
            return "Medium"
        elif grade == "D":
            return "High"
        return "High"

    def generate(self):
        n = self.n_samples

        student_id = np.arange(1, n + 1)

        attendance = np.clip(np.random.normal(78, 15, n), 35, 100)
        study_hours = np.clip(np.random.normal(3.5, 1.8, n), 0.5, 10)
        quiz_score = np.clip(np.random.normal(65, 18, n), 0, 100)
        assignment_score = np.clip(np.random.normal(68, 16, n), 0, 100)
        midterm_score = np.clip(np.random.normal(62, 20, n), 0, 100)
        sleep_hours = np.clip(np.random.normal(7, 1.3, n), 4, 10)
        lms_login_count = np.random.randint(2, 45, n)
        participation = np.clip(np.random.normal(60, 20, n), 0, 100)

        parent_education = np.random.choice(
            ["High School", "Bachelor", "Master", "PhD"],
            size=n,
            p=[0.35, 0.40, 0.20, 0.05]
        )

        internet_access = np.random.choice(
            ["Yes", "No"],
            size=n,
            p=[0.90, 0.10]
        )

        family_income = np.random.choice(
            ["Low", "Medium", "High"],
            size=n,
            p=[0.30, 0.50, 0.20]
        )

        extracurricular = np.random.choice(
            ["Yes", "No"],
            size=n,
            p=[0.45, 0.55]
        )

        # performance score simulation
        performance_score = (
            attendance * 0.20 +
            study_hours * 8 +
            quiz_score * 0.18 +
            assignment_score * 0.18 +
            midterm_score * 0.22 +
            participation * 0.07 +
            lms_login_count * 0.35 +
            sleep_hours * 1.5
        )

        # adjustments
        performance_score += np.where(internet_access == "Yes", 3, -4)
        performance_score += np.where(extracurricular == "Yes", 2, 0)
        performance_score += np.where(family_income == "High", 2, 0)
        performance_score += np.random.normal(0, 5, n)

        performance_score = np.clip(performance_score / 1.5, 0, 100)

        final_grade = [self._grade_from_score(x) for x in performance_score]
        risk_level = [self._risk_from_grade(x) for x in final_grade]

        df = pd.DataFrame({
            "student_id": student_id,
            "attendance": attendance.round(2),
            "study_hours": study_hours.round(2),
            "quiz_score": quiz_score.round(2),
            "assignment_score": assignment_score.round(2),
            "midterm_score": midterm_score.round(2),
            "sleep_hours": sleep_hours.round(2),
            "lms_login_count": lms_login_count,
            "participation": participation.round(2),
            "parent_education": parent_education,
            "internet_access": internet_access,
            "family_income": family_income,
            "extracurricular": extracurricular,
            "final_grade": final_grade,
            "risk_level": risk_level
        })

        return df

    def save(self):
        df = self.generate()
        file_path = SYNTHETIC_DIR / "student_performance.csv"
        df.to_csv(file_path, index=False)

        print(f"[INFO] Synthetic dataset saved -> {file_path}")
        print(f"[INFO] Shape -> {df.shape}")

        return df