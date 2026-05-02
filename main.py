from pprint import pprint

from src.config import create_folders
from src.data_generator import StudentDataGenerator
from src.train import ModelTrainer
from src.predict import StudentPredictor


def main():
    create_folders()

    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("=" * 60)

    # Step 1: Generate dataset
    print("\n[1] Generating synthetic dataset...")
    generator = StudentDataGenerator(
        n_samples=50000
    )
    generator.save()

    # Step 2: Train models
    print("\n[2] Training ML models...")
    trainer = ModelTrainer()
    trainer.train()

    # Step 3: Predict
    print("\n[3] Running sample prediction...")

    predictor = StudentPredictor()

    sample_student = {
        "attendance": 82,
        "study_hours": 4.5,
        "quiz_score": 74,
        "assignment_score": 78,
        "midterm_score": 70,
        "sleep_hours": 7.5,
        "lms_login_count": 24,
        "participation": 80,
        "parent_education": "Bachelor",
        "internet_access": "Yes",
        "family_income": "Medium",
        "extracurricular": "Yes"
    }

    result = predictor.predict(
        sample_student
    )

    print("\nPrediction Result:")
    pprint(result)

    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()