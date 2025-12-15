import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def create_proxy_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a proxy target variable using RFM clustering.
    Returns a DataFrame indexed by CustomerId with is_high_risk label.
    """

    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            frequency=("TransactionId", "count"),
            monetary=("Value", "sum"),
        )
    )

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster (lowest frequency & monetary)
    cluster_summary = rfm.groupby("cluster")[["frequency", "monetary"]].mean()
    high_risk_cluster = cluster_summary.sum(axis=1).idxmin()

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm[["is_high_risk"]]
