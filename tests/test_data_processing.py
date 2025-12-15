from src.data_processing import compute_rfm, assign_risk_cluster
import pandas as pd

def test_rfm_output():
    df = pd.DataFrame({
        'CustomerId':[1,1,2],
        'TransactionId':[101,102,103],
        'Value':[100,200,50],
        'TransactionStartTime':['2021-01-01','2021-01-05','2021-01-03']
    })
    rfm = compute_rfm(df, "2021-01-06")
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns

def test_cluster_assignment():
    df = pd.DataFrame({
        'CustomerId':[1,2],
        'Recency':[5,10],
        'Frequency':[10,1],
        'Monetary':[1000,50]
    })
    df = assign_risk_cluster(df, n_clusters=2)
    assert 'is_high_risk' in df.columns
