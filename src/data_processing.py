# src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load raw transaction data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def preprocess_features(df):
    """
    Perform feature engineering:
    - Temporal features
    - Aggregates
    - Categorical encoding
    - Scaling
    """
    # Convert datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Temporal features
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    
    # Aggregate per customer
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: max(x),
        'Value': ['sum', 'mean', 'std'],
        'TransactionId': 'count'
    }).reset_index()
    rfm.columns = ['CustomerId', 'LastTransaction', 'TotalValue', 'AvgValue', 'StdValue', 'TransactionCount']
    
    # RFM metrics
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm['Recency'] = (snapshot_date - rfm['LastTransaction']).dt.days
    rfm['Frequency'] = rfm['TransactionCount']
    rfm['Monetary'] = rfm['TotalValue']
    
    # KMeans clustering for high-risk proxy
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify high-risk cluster (high recency, low frequency/monetary)
    cluster_means = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_means.sort_values('Recency', ascending=False).index[0]
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    
    # Merge back
    df = df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
    
    return df

def build_preprocessor(num_features, cat_features):
    """
    Build preprocessing pipeline for numerical and categorical features
    """
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ])
    return preprocessor

def train_test_split_data(df, target='is_high_risk', test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    """
    X = df.drop([target, 'TransactionStartTime', 'TransactionId'], axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
