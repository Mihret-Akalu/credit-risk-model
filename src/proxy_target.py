import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


def create_proxy_target(df: pd.DataFrame, n_clusters: int = 3, n_runs: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """
    Create a proxy target variable using RFM clustering with stability checks.
    
    Args:
        df: Input dataframe with transaction data
        n_clusters: Number of clusters for KMeans
        n_runs: Number of runs to check clustering stability
    
    Returns:
        Tuple of (target_df, metadata) where target_df contains 'is_high_risk'
        and metadata contains clustering statistics
    """
    # Convert to datetime
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    
    # Calculate snapshot date
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)
    
    # Calculate RFM metrics
    rfm = (
        df.groupby("CustomerId")
        .agg(
            recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            frequency=("TransactionId", "count"),
            monetary=("Value", "sum"),
        )
    )
    
    # Scale features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # Stability check: run KMeans multiple times
    stability_results = []
    cluster_assignments = []
    
    for i in range(n_runs):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42 + i, n_init=10)
        labels = kmeans.fit_predict(rfm_scaled)
        cluster_assignments.append(labels)
        
        # Calculate cluster centroids
        centroids = kmeans.cluster_centers_
        stability_results.append({
            'run': i,
            'inertia': kmeans.inertia_,
            'centroids': centroids
        })
    
    # Use the most common clustering (mode) for final assignment
    cluster_matrix = np.column_stack(cluster_assignments)
    final_labels = []
    
    for row in cluster_matrix:
        unique, counts = np.unique(row, return_counts=True)
        final_labels.append(unique[np.argmax(counts)])
    
    rfm["cluster"] = final_labels
    
    # Calculate cluster stability (percentage of consistent assignments)
    consistent_count = 0
    for i in range(len(rfm)):
        if len(np.unique(cluster_matrix[i])) == 1:
            consistent_count += 1
    
    stability_percentage = consistent_count / len(rfm) * 100
    
    # Identify high-risk cluster (lowest engagement)
    # Business rule: high risk = low frequency + low monetary value
    cluster_summary = rfm.groupby("cluster").agg({
        'frequency': 'mean',
        'monetary': 'mean',
        'recency': 'mean'
    })
    
    # Create risk score (lower = higher risk)
    cluster_summary['risk_score'] = (
        cluster_summary['frequency'].rank(ascending=True) +
        cluster_summary['monetary'].rank(ascending=True) +
        cluster_summary['recency'].rank(ascending=False)
    )
    
    high_risk_cluster = cluster_summary['risk_score'].idxmin()
    
    # Label high-risk customers
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)
    
    # Prepare metadata
    metadata = {
        'stability_percentage': stability_percentage,
        'cluster_summary': cluster_summary,
        'high_risk_cluster': int(high_risk_cluster),
        'risk_distribution': rfm['is_high_risk'].value_counts().to_dict(),
        'n_runs': n_runs
    }
    
    # Print business rationale summary
    print("\n=== Proxy Target Engineering Summary ===")
    print(f"Clustering Stability: {stability_percentage:.2f}% consistent across {n_runs} runs")
    print(f"High-risk cluster: Cluster {high_risk_cluster}")
    print("\nCluster Characteristics:")
    print(cluster_summary)
    print(f"\nRisk Distribution: {metadata['risk_distribution']}")
    print("\nBusiness Rationale:")
    print("High-risk customers identified by low transaction frequency and monetary value,")
    print("combined with high recency (inactive). These behavioral patterns correlate")
    print("with higher credit risk based on alternative data analysis.")
    
    return rfm[["is_high_risk"]], metadata


def validate_proxy_target(target_series: pd.Series, min_samples: int = 100) -> bool:
    """
    Validate the proxy target meets minimum requirements.
    
    Args:
        target_series: Series containing is_high_risk labels
        min_samples: Minimum samples required in minority class
    
    Returns:
        bool: True if validation passes
    """
    counts = target_series.value_counts()
    
    if len(counts) < 2:
        print("Warning: Proxy target has only one class")
        return False
    
    minority_class = counts.idxmin()
    minority_count = counts.min()
    
    if minority_count < min_samples:
        print(f"Warning: Minority class has only {minority_count} samples (min: {min_samples})")
        return False
    
    imbalance_ratio = counts.max() / counts.min()
    if imbalance_ratio > 20:
        print(f"Warning: Extreme class imbalance (ratio: {imbalance_ratio:.1f}:1)")
    
    return True