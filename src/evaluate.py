import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.config import REPORT_DIR, CHART_DIR


class ModelEvaluator:

    @staticmethod
    def evaluate(model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)

        metrics = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(
                precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            )
        }

        print(f"\n===== {model_name} =====")
        for k, v in metrics.items():
            if k != "model":
                print(f"{k}: {v:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # save metrics
        with open(
            REPORT_DIR / f"{model_name.lower()}_metrics.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(metrics, f, indent=4)

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(
            CHART_DIR / f"{model_name.lower()}_confusion_matrix.png",
            dpi=300
        )
        plt.close()

        return metrics