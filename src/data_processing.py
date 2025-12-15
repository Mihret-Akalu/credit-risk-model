import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # Features
    cat_features = ['CurrencyCode', 'ProviderId', 'ProductCategory', 'ChannelId']
    num_features = ['Amount', 'Value', 'PricingStrategy']

    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ]
    )

    return preprocessor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def create_proxy_target(df):
    # RFM Metrics
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={'TransactionStartTime': 'Recency',
                       'TransactionId': 'Frequency',
                       'Amount': 'Monetary'})
    
    # Scale RFM
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # High-risk = lowest engagement
    high_risk_cluster = rfm.groupby('Cluster')['Monetary'].mean().idxmin()
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    
    return rfm[['is_high_risk']]
