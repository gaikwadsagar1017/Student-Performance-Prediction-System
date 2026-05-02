import numpy as np
import pandas as pd


class FeatureEngineer:
    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # Academic engagement score
        data["engagement_score"] = (
            data["attendance"] * 0.35 +
            data["participation"] * 0.25 +
            data["lms_login_count"] * 1.5
        )

        # Assignment discipline
        data["discipline_score"] = (
            data["assignment_score"] * 0.6 +
            data["quiz_score"] * 0.4
        )

        # Lifestyle balance
        data["wellness_score"] = (
            data["sleep_hours"] * 10
        )

        # Academic consistency
        data["consistency_score"] = (
            (
                data["quiz_score"] +
                data["assignment_score"] +
                data["midterm_score"]
            ) / 3
        )

        # Study efficiency
        data["study_efficiency"] = (
            data["quiz_score"] /
            (data["study_hours"] + 1)
        )

        # Attendance category
        data["attendance_band"] = pd.cut(
            data["attendance"],
            bins=[0, 50, 70, 85, 100],
            labels=["Poor", "Average", "Good", "Excellent"]
        )

        # High engagement flag
        data["high_engagement"] = np.where(
            data["engagement_score"] >= 80,
            1,
            0
        )

        return data