import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data(path: str) -> pd.DataFrame:
    """Load CSV data"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def compute_rfm(df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """Compute RFM metrics per CustomerId"""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot = pd.to_datetime(snapshot_date)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot - x.max()).days,
        'TransactionId': 'count',
        'Value': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm

def assign_risk_cluster(rfm_df: pd.DataFrame, n_clusters=3, random_state=42) -> pd.DataFrame:
    """Assign high-risk cluster"""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency','Frequency','Monetary']])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify high-risk cluster (highest Recency, lowest Frequency & Monetary)
    cluster_summary = rfm_df.groupby('Cluster')[['Recency','Frequency','Monetary']].mean()
    high_risk_cluster = cluster_summary['Recency'].idxmax()
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    
    return rfm_df
