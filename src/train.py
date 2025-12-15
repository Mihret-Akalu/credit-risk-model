import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from data_processing import load_data, compute_rfm, assign_risk_cluster

# Load data
df = load_data("../data/raw/data.csv")

# Compute RFM
rfm = compute_rfm(df, snapshot_date="2018-12-31")
rfm = assign_risk_cluster(rfm)

# Merge target back to transaction-level features (example: aggregate by CustomerId)
X = rfm[['Recency','Frequency','Monetary']]
y = rfm['is_high_risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing (numerical only in this example)
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['Recency','Frequency','Monetary'])
])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [50,100],
    'classifier__max_depth': [5,10,None],
    'classifier__min_samples_split': [2,5,10]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', verbose=2)
grid.fit(X_train, y_train)

# Evaluation
y_pred = grid.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))

# Save model
joblib.dump(grid.best_estimator_, "../models/rf_model.pkl")
