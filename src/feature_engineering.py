import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class TemporalFeatures(BaseEstimator, TransformerMixin):
    """Extracts temporal features from TransactionStartTime."""
    
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        X['TransactionHour'] = X[self.datetime_col].dt.hour
        X['TransactionDay'] = X[self.datetime_col].dt.day
        X['TransactionMonth'] = X[self.datetime_col].dt.month
        X['TransactionYear'] = X[self.datetime_col].dt.year
        X['TransactionDayOfWeek'] = X[self.datetime_col].dt.dayofweek
        X['TransactionWeekOfYear'] = X[self.datetime_col].dt.isocalendar().week
        return X.drop(columns=[self.datetime_col])


class WOETransformer(BaseEstimator, TransformerMixin):
    """Weight of Evidence (WoE) transformer for categorical features."""
    
    def __init__(self, categorical_features, target_col='is_high_risk', min_samples=50):
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.min_samples = min_samples
        self.woe_dict = {}
        self.iv_dict = {}
    
    def fit(self, X, y):
        df = X[self.categorical_features].copy()
        df[self.target_col] = y
        
        for col in self.categorical_features:
            # Group infrequent categories
            counts = df[col].value_counts()
            frequent_categories = counts[counts >= self.min_samples].index
            df[col] = df[col].where(df[col].isin(frequent_categories), 'Other')
            
            # Calculate WoE
            total_good = (df[self.target_col] == 0).sum()
            total_bad = (df[self.target_col] == 1).sum()
            
            woe_map = {}
            iv_total = 0
            
            for category in df[col].unique():
                good = ((df[col] == category) & (df[self.target_col] == 0)).sum()
                bad = ((df[col] == category) & (df[self.target_col] == 1)).sum()
                
                # Avoid division by zero
                good_dist = good / total_good if total_good > 0 else 0.5 / len(df)
                bad_dist = bad / total_bad if total_bad > 0 else 0.5 / len(df)
                
                # Smoothing
                good_dist = max(good_dist, 0.0001)
                bad_dist = max(bad_dist, 0.0001)
                
                woe = np.log(bad_dist / good_dist)
                iv = (bad_dist - good_dist) * woe
                
                woe_map[category] = woe
                iv_total += iv
            
            self.woe_dict[col] = woe_map
            self.iv_dict[col] = iv_total
        
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.categorical_features:
            woe_map = self.woe_dict.get(col, {})
            X[col] = X[col].map(woe_map).fillna(0)  # 0 for unseen categories
        return X
    
    def get_iv_summary(self):
        """Returns IV summary for documentation."""
        return pd.DataFrame({
            'Feature': list(self.iv_dict.keys()),
            'IV': list(self.iv_dict.values())
        }).sort_values('IV', ascending=False)