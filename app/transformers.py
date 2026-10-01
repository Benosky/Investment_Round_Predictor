from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# 1.1. Define the Month Features Extractor Transformer
class MonthFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column, new_columns_names=['year','month','day','dayofweek']):
        self.date_column = date_column
        self.new_columns_names = new_columns_names
        
    def fit(self, X, y=None):
        return self  # No fitting logic needed
        
    def transform(self, X):
        # Prevent modifying the original DataFrame
        X_out = X.copy()
        
        # Convert to datetime and extract month number (1-12)
        date_series = pd.to_datetime(X_out[self.date_column], format='%d-%b-%y')
        date_df = pd.DataFrame({
                                    'year': date_series.dt.year,
                                    'month': date_series.dt.month,
                                    'day': date_series.dt.day,
                                    'dayofweek': date_series.dt.dayofweek
                                                })
        X_out[self.new_columns_names] = date_df
        X_out = X_out.drop(columns=[self.date_column])
        return X_out


# 1.2. Define the DropColumn Transformer
class DropColumn(BaseEstimator, TransformerMixin):
    def __init__(self, drop_column):
        self.drop_column = drop_column
        # self.new_columns_names = new_columns_names
        
    def fit(self, X, y=None):
        return self  # No fitting logic needed
        
    def transform(self, X):
        # Prevent modifying the original DataFrame
        X_out = X.copy()
        try:
            # Drop the required column
            X_out = X_out.drop(columns=[self.drop_column])
        except KeyError:
            pass
        return X_out