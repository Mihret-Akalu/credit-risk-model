import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data_processing import preprocess_data
from src.proxy_target import create_proxy_target


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
    MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

    df = pd.read_csv(DATA_PATH)

    target = create_proxy_target(df)
    df = df.merge(target, left_on="CustomerId", right_index=True)

    X = df.drop(["CustomerId", "is_high_risk"], axis=1)
    y = df["is_high_risk"]

    preprocessor = preprocess_data(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [5, 10],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="roc_auc",
        verbose=2,
    )

    grid.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(grid.best_estimator_, MODEL_PATH)


if __name__ == "__main__":
    main()
