import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

class CreditRiskPredictor:
    """Credit risk predictor with consistent preprocessing."""
    
    def __init__(self, model_path: Union[str, Path] = None):
        """
        Initialize predictor with trained model pipeline.
        
        Args:
            model_path: Path to saved model pipeline (.pkl file)
        """
        if model_path is None:
            model_path = PROJECT_ROOT / "models" / "best_model.pkl"
        
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        self.load_model()
        
        # Define expected input schema
        self.expected_features = [
            'TransactionStartTime', 'CurrencyCode', 'ProviderId', 
            'ProductCategory', 'ChannelId', 'Amount', 'Value', 'PricingStrategy'
        ]
    
    def load_model(self) -> None:
        """Load model with error handling."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from: {self.model_path}")
            
            # Extract feature names if available
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
                logger.info(f"Model expects {len(self.feature_names)} features")
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare input data.
        
        Args:
            df: Input dataframe
            
        Returns:
            Validated dataframe
            
        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        missing_cols = set(self.expected_features) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Keep only expected columns in correct order
        df = df[self.expected_features].copy()
        
        # Type validation
        numeric_cols = ['Amount', 'Value', 'PricingStrategy']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Check for valid ranges
                if df[col].isnull().any():
                    raise ValueError(f"Invalid numeric values in {col}")
                if (df[col] < 0).any() and col != 'Amount':  # Amount can be negative
                    raise ValueError(f"Negative values not allowed in {col}")
        
        # Categorical columns
        categorical_cols = ['CurrencyCode', 'ProviderId', 'ProductCategory', 'ChannelId']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Transaction time validation
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
            if df['TransactionStartTime'].isnull().any():
                raise ValueError("Invalid TransactionStartTime format")
        
        logger.info(f"Input validated: {df.shape}")
        return df
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict risk probability for input features.
        
        Args:
            df: Input dataframe with expected features
            
        Returns:
            Series of risk probabilities (0-1)
        """
        try:
            # Validate input
            df_validated = self.validate_input(df)
            
            # Make predictions
            probabilities = self.model.predict_proba(df_validated)[:, 1]
            
            logger.info(f"Predictions generated for {len(probabilities)} samples")
            return pd.Series(probabilities, name='risk_probability')
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_single(self, data: Dict[str, Any]) -> float:
        """
        Predict risk probability for a single record.
        
        Args:
            data: Dictionary of feature values
            
        Returns:
            Risk probability (0-1)
        """
        df = pd.DataFrame([data])
        return self.predict(df).iloc[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            'model_path': str(self.model_path),
            'model_type': type(self.model).__name__ if self.model else None,
            'features_expected': self.expected_features,
            'features_available': self.feature_names,
            'model_loaded': self.model is not None
        }


# Global predictor instance
_predictor = None

def get_predictor() -> CreditRiskPredictor:
    """Get or create global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = CreditRiskPredictor()
    return _predictor


def predict(df: pd.DataFrame) -> pd.Series:
    """
    Predict risk probability for input features (legacy function).
    
    Args:
        df: Input dataframe
        
    Returns:
        Series of risk probabilities
    """
    predictor = get_predictor()
    return predictor.predict(df)


if __name__ == "__main__":
    # Test the predictor
    predictor = CreditRiskPredictor()
    print("Model info:", predictor.get_model_info())
    
    # Create test data
    test_data = {
        'TransactionStartTime': ['2024-01-15 14:30:00'],
        'CurrencyCode': ['USD'],
        'ProviderId': ['P123'],
        'ProductCategory': ['airtime'],
        'ChannelId': ['mobile'],
        'Amount': [1500.0],
        'Value': [1500.0],
        'PricingStrategy': [2]
    }
    
    test_df = pd.DataFrame(test_data)
    
    try:
        predictions = predictor.predict(test_df)
        print(f"Test prediction: {predictions.iloc[0]:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")