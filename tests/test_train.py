import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def test_evaluate_model():
    """Test model evaluation function."""
    # Create synthetic data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Evaluate
    metrics = evaluate_model(model, X, y, "Test Model")
    
    # Check metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    
    # Check metric ranges
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1


def test_model_pipeline_integration():
    """Test that the training pipeline can be imported and runs."""
    from src.data_processing import create_feature_pipeline
    from src.proxy_target import create_proxy_target
    
    # Create mock data
    data = {
        'CustomerId': [1, 1, 2, 2, 3],
        'TransactionId': [101, 102, 103, 104, 105],
        'Amount': [100.0, 200.0, 50.0, 150.0, 300.0],
        'Value': [100.0, 200.0, 50.0, 150.0, 300.0],
        'TransactionStartTime': [
            '2024-01-01 10:00:00', '2024-01-02 11:00:00',
            '2024-01-03 12:00:00', '2024-01-04 13:00:00',
            '2024-01-05 14:00:00'
        ],
        'CurrencyCode': ['USD', 'USD', 'EUR', 'EUR', 'USD'],
        'ProviderId': ['P1', 'P2', 'P1', 'P2', 'P1'],
        'ProductCategory': ['A', 'B', 'A', 'B', 'A'],
        'ChannelId': ['Web', 'Mobile', 'Web', 'Mobile', 'Web'],
        'PricingStrategy': [1, 2, 1, 2, 1]
    }
    
    df = pd.DataFrame(data)
    
    # Test proxy target creation
    target_df, metadata = create_proxy_target(df)
    assert 'is_high_risk' in target_df.columns
    assert target_df.shape[0] == len(df['CustomerId'].unique())
    
    # Test feature pipeline
    pipeline = create_feature_pipeline(use_woe=False)
    assert pipeline is not None
    
    print("Pipeline integration test passed!")


if __name__ == "__main__":
    test_evaluate_model()
    test_model_pipeline_integration()
    print("All tests passed!")