import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)

from src.config import (
    SYNTHETIC_DIR,
    PROCESSED_DIR,
    MODELS_DIR
)
from src.feature_engineering import FeatureEngineer


class DataPreprocessor:
    def __init__(self):
        self.pipeline = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        file_path = SYNTHETIC_DIR / "student_performance.csv"
        df = pd.read_csv(file_path)

        print(f"[INFO] Loaded dataset -> {df.shape}")
        return df

    def clean_data(self, df):
        data = df.copy()

        data = data.drop_duplicates()

        numeric_cols = data.select_dtypes(include=["number"]).columns
        categorical_cols = data.select_dtypes(include=["object"]).columns

        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())

        for col in categorical_cols:
            data[col] = data[col].fillna(data[col].mode()[0])

        return data

    def prepare(self):
        df = self.load_data()

        df = self.clean_data(df)

        df = FeatureEngineer.add_features(df)

        target = "final_grade"

        X = df.drop(columns=["student_id", "final_grade", "risk_level"])
        y = df[target]

        y_encoded = self.label_encoder.fit_transform(y)

        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numerical_features = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        numeric_transformer = Pipeline([
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.pipeline = ColumnTransformer([
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        X_processed = self.pipeline.fit_transform(X)

        # save artifacts
        joblib.dump(
            self.pipeline,
            MODELS_DIR / "preprocessor.pkl"
        )

        joblib.dump(
            self.label_encoder,
            MODELS_DIR / "label_encoder.pkl"
        )

        processed_df = pd.DataFrame(X)
        processed_df["target"] = y

        processed_df.to_csv(
            PROCESSED_DIR / "processed_student_data.csv",
            index=False
        )

        print("[INFO] Preprocessing completed")
        print(f"[INFO] Features shape -> {X_processed.shape}")

        return X_processed, y_encoded