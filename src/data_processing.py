import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


# --- 1. Custom Transformer for Feature Engineering ---
class TransactionFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, timestamp_col='TransactionStartTime', amount_col='Amount', customer_id_col='CustomerId'):
        self.timestamp_col = timestamp_col
        self.amount_col = amount_col
        self.customer_id_col = customer_id_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Extract datetime parts
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df['hour'] = df[self.timestamp_col].dt.hour
        df['day'] = df[self.timestamp_col].dt.day
        df['month'] = df[self.timestamp_col].dt.month
        df['year'] = df[self.timestamp_col].dt.year

        # Aggregate by customer
        agg = df.groupby(self.customer_id_col).agg(
            transaction_count=(self.amount_col, 'count'),
            total_amount=(self.amount_col, 'sum'),
            avg_amount=(self.amount_col, 'mean'),
            std_amount=(self.amount_col, 'std'),
            first_transaction=('TransactionStartTime', 'min'),
            last_transaction=('TransactionStartTime', 'max')
        ).fillna(0).reset_index()

        # Merge back extracted features
        merged = df.drop_duplicates(self.customer_id_col)[[self.customer_id_col, 'hour', 'day', 'month', 'year']]
        result = pd.merge(agg, merged, on=self.customer_id_col, how='left')
        return result


# --- 2. Build Full Transformation Pipeline ---
def build_preprocessing_pipeline(numeric_features, categorical_features):
    # Numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine all
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])

    return preprocessor


# --- 3. Combine Feature Generation + Preprocessing ---
def build_full_pipeline(numeric_features, categorical_features,
                        timestamp_col='TransactionStartTime', amount_col='Amount', customer_id_col='CustomerId'):
    return Pipeline([
        ('feature_engineering', TransactionFeatureGenerator(
            timestamp_col=timestamp_col,
            amount_col=amount_col,
            customer_id_col=customer_id_col
        )),
        ('preprocessing', build_preprocessing_pipeline(numeric_features, categorical_features))
    ])


