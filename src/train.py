# src/train.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from data_processing import build_preprocessor, train_test_split_data, load_data, preprocess_features

# Load and preprocess
df = load_data("data/raw/data.csv")
df = preprocess_features(df)

# Define features
num_features = ['Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
cat_features = ['ProductCategory', 'ChannelId', 'ProviderId']

preprocessor = build_preprocessor(num_features, cat_features)
X_train, X_test, y_train, y_test = train_test_split_data(df)

# Build pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10],
    'classifier__min_samples_split': [2, 5]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', verbose=2)
grid.fit(X_train, y_train)

# Evaluate
y_pred = grid.predict(X_test)
y_proba = grid.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# MLflow logging
mlflow.start_run()
mlflow.log_params(grid.best_params_)
mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
mlflow.log_metric('roc_auc', roc_auc_score(y_test, y_proba))
mlflow.sklearn.log_model(grid.best_estimator_, "credit_risk_model")
mlflow.end_run()

# Save model
joblib.dump(grid.best_estimator_, "models/credit_risk_model.pkl")
