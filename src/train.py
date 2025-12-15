import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_processing import preprocess_data
from proxy_target import create_proxy_target

df = pd.read_csv('../data/raw/data.csv')
target = create_proxy_target(df)
df = df.merge(target, left_on='CustomerId', right_index=True)

X = df.drop(['CustomerId', 'is_high_risk'], axis=1)
y = df['is_high_risk']

preprocessor = preprocess_data(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', verbose=2)
grid.fit(X_train, y_train)

joblib.dump(grid.best_estimator_, '../models/best_model.pkl')
