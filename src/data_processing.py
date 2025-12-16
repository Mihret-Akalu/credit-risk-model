import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .feature_engineering import TemporalFeatures, WOETransformer

def create_feature_pipeline(use_woe=True):
    """
    Creates the complete preprocessing pipeline.
    
    Args:
        use_woe (bool): Whether to use WoE transformation for categorical features.
    
    Returns:
        ColumnTransformer: Complete preprocessing pipeline
    """
    # Define feature groups
    temporal_features = ['TransactionStartTime']
    numerical_features = ['Amount', 'Value', 'PricingStrategy']
    categorical_features = ['CurrencyCode', 'ProviderId', 'ProductCategory', 'ChannelId']
    
    # Temporal pipeline
    temporal_pipeline = Pipeline([
        ('temporal_extractor', TemporalFeatures())
    ])
    
    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    if use_woe:
        categorical_pipeline = Pipeline([
            ('woe_transformer', WOETransformer(categorical_features=categorical_features))
        ])
    else:
        categorical_pipeline = Pipeline([
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
    
    # Complete preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('temporal', temporal_pipeline, temporal_features),
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def preprocess_data(df, use_woe=True):
    """
    Main preprocessing function for backward compatibility.
    
    Args:
        df (pd.DataFrame): Input dataframe
        use_woe (bool): Whether to use WoE transformation
    
    Returns:
        ColumnTransformer: Fitted preprocessor
    """
    preprocessor = create_feature_pipeline(use_woe=use_woe)
    return preprocessor