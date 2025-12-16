# train_fixed.py - Simplified working version
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.proxy_target import create_proxy_target


def create_preprocessor():
    """Create preprocessing pipeline WITHOUT WoE transformer for simplicity."""
    
    # Define feature groups
    numerical_features = ['Amount', 'Value', 'PricingStrategy']
    categorical_features = ['CurrencyCode', 'ProviderId', 'ProductCategory', 'ChannelId']
    
    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline (simple one-hot encoding)
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    # Complete preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def extract_temporal_features(df):
    """Extract temporal features separately."""
    df = df.copy()
    if 'TransactionStartTime' in df.columns:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
        df = df.drop(columns=['TransactionStartTime'])
    return df


def evaluate_model(model, X_test, y_test, model_name=""):
    """Comprehensive model evaluation."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    if model_name:
        print(f"\n=== {model_name} Evaluation ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics


def main():
    # Setup paths
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data shape: {df.shape}")
    
    # Create proxy target
    print("\nCreating proxy target...")
    target_df, target_metadata = create_proxy_target(df)
    
    # Merge target
    df = df.merge(target_df, left_on="CustomerId", right_index=True)
    print(f"Target distribution:\n{df['is_high_risk'].value_counts()}")
    
    # Extract temporal features
    print("\nExtracting temporal features...")
    df = extract_temporal_features(df)
    
    # Prepare features and target
    X = df.drop(["CustomerId", "is_high_risk", "TransactionId"], axis=1, errors='ignore')
    y = df["is_high_risk"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Features: {X_train.columns.tolist()}")
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessor()
    
    # Define models with hyperparameter grids
    models = {
        'logistic_regression': {
            'pipeline': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            'param_grid': {
                'classifier__C': [0.01, 0.1, 1.0],
                'classifier__solver': ['liblinear']
            }
        },
        'random_forest': {
            'pipeline': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ]),
            'param_grid': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20]
            }
        }
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    
    # Train and evaluate each model
    for model_name, config in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}")
        
        # Grid search
        grid = GridSearchCV(
            config['pipeline'],
            config['param_grid'],
            cv=3,  # Reduced from 5 for speed
            scoring='roc_auc',
            verbose=1,
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        
        # Evaluate
        test_metrics = evaluate_model(grid.best_estimator_, X_test, y_test, model_name)
        
        # Track best model
        if test_metrics['roc_auc'] > best_score:
            best_score = test_metrics['roc_auc']
            best_model = grid.best_estimator_
            best_model_name = model_name
        
        print(f"Best {model_name} params: {grid.best_params_}")
        print(f"Best {model_name} CV score: {grid.best_score_:.4f}")
        print(f"Test AUC: {test_metrics['roc_auc']:.4f}")
    
    # Save the best model
    if best_model:
        print(f"\n{'='*50}")
        print(f"Best model: {best_model_name} with AUC: {best_score:.4f}")
        
        best_model_path = MODELS_DIR / "best_model.pkl"
        joblib.dump(best_model, best_model_path)
        print(f"Saved best model to: {best_model_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()