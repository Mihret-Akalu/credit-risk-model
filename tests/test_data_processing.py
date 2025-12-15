import pandas as pd
from src.data_processing import preprocess_data

def test_preprocessing():
    df = pd.DataFrame({
        'Amount': [100, 200],
        'Value': [100, 200],
        'PricingStrategy': [1, 2],
        'CurrencyCode': ['UGX', 'UGX'],
        'ProviderId': ['P1', 'P2'],
        'ProductCategory': ['airtime', 'utility'],
        'ChannelId': ['C1', 'C2']
    })
    preprocessor = preprocess_data(df)
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == df.shape[0]
