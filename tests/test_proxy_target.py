import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.proxy_target import create_proxy_target, validate_proxy_target


def test_create_proxy_target():
    """Test proxy target creation with sample data."""
    # Create sample transaction data
    data = {
        'CustomerId': [1, 1, 2, 2, 3, 3, 4, 4],
        'TransactionId': range(100, 108),
        'Value': [100, 200, 50, 150, 300, 400, 10, 20],
        'TransactionStartTime': pd.date_range('2024-01-01', periods=8, freq='D').strftime('%Y-%m-%d %H:%M:%S'),
        'CurrencyCode': ['USD'] * 8,
        'ProviderId': ['P1', 'P2'] * 4,
        'ProductCategory': ['A', 'B'] * 4,
        'ChannelId': ['Web', 'Mobile'] * 4,
        'PricingStrategy': [1, 2] * 4,
        'Amount': [100, 200, 50, 150, 300, 400, 10, 20]
    }
    
    df = pd.DataFrame(data)
    
    # Create proxy target
    target_df, metadata = create_proxy_target(df, n_clusters=2, n_runs=2)
    
    # Assertions
    assert 'is_high_risk' in target_df.columns
    assert target_df.shape[0] == len(df['CustomerId'].unique())
    assert set(target_df['is_high_risk'].unique()).issubset({0, 1})
    
    # Check metadata
    assert 'stability_percentage' in metadata
    assert 'cluster_summary' in metadata
    assert 'high_risk_cluster' in metadata
    assert 'risk_distribution' in metadata
    
    print("Proxy target creation test passed!")
    return target_df, metadata


def test_validate_proxy_target():
    """Test proxy target validation."""
    # Valid case
    valid_series = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    assert validate_proxy_target(valid_series, min_samples=2) == True
    
    # Invalid: only one class
    invalid_series = pd.Series([0, 0, 0, 0, 0, 0])
    assert validate_proxy_target(invalid_series, min_samples=2) == False
    
    # Invalid: minority class too small
    small_minority = pd.Series([0, 0, 0, 0, 0, 0, 1])
    assert validate_proxy_target(small_minority, min_samples=2) == False
    
    print("Proxy target validation test passed!")


if __name__ == "__main__":
    target_df, metadata = test_create_proxy_target()
    test_validate_proxy_target()
    print("\nAll proxy target tests passed!")
    print(f"Target distribution:\n{target_df['is_high_risk'].value_counts()}")
    print(f"Stability: {metadata['stability_percentage']:.2f}%")