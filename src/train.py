import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from src.config import MODELS_DIR, REPORT_DIR
from src.preprocess import DataPreprocessor
from src.evaluate import ModelEvaluator


class ModelTrainer:
    def __init__(self):
        self.results = []

    def train(self):
        print("\nLoading processed data...")

        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=2000
            ),

            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),

            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss"
            )
        }

        best_model = None
        best_score = 0
        best_name = ""

        for name, model in models.items():
            print(f"\nTraining {name}...")

            model.fit(X_train, y_train)

            metrics = ModelEvaluator.evaluate(
                model,
                X_test,
                y_test,
                name
            )

            self.results.append(metrics)

            if metrics["accuracy"] > best_score:
                best_score = metrics["accuracy"]
                best_model = model
                best_name = name

            # save individual model
            joblib.dump(
                model,
                MODELS_DIR / f"{name.lower()}.pkl"
            )

        # save best model
        joblib.dump(
            best_model,
            MODELS_DIR / "best_model.pkl"
        )

        leaderboard = pd.DataFrame(
            self.results
        ).sort_values(
            by="accuracy",
            ascending=False
        )

        leaderboard.to_csv(
            REPORT_DIR / "model_leaderboard.csv",
            index=False
        )

        print("\n================================")
        print(f"Best Model: {best_name}")
        print(f"Accuracy: {best_score:.4f}")
        print("================================")

        return best_model