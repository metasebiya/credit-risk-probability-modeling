import os
import pandas as pd
import numpy as np
import joblib
from src.proxy_target_engineer import ProxyTargetEngineer
from src.data_processing import build_full_pipeline

# Test the presence of is_high_risk and matching of aggregation
def test_proxy_target_labeling():
    df_raw = pd.read_csv("../data/raw/data.csv")

    proxy = ProxyTargetEngineer(
        customer_id_col='CustomerId',
        timestamp_col='TransactionStartTime',
        amount_col='Amount'
    )
    rfm_labeled = proxy.engineer_target(df_raw)
    df_labeled = proxy.merge_with_main(df_raw, rfm_labeled)

    assert 'is_high_risk' in df_labeled.columns, "Missing 'is_high_risk' label"
    assert df_labeled['CustomerId'].nunique() == rfm_labeled.shape[0], "Mismatch in customer aggregation"

# Test matching of rows of transformation and output
def test_pipeline_transform_output():
    df_labeled = pd.read_csv("data/processed/data_full_labeled.csv")

    pipeline = build_full_pipeline(
        numeric_features=['transaction_count', 'total_amount', 'avg_amount', 'std_amount'],
        categorical_features=['month', 'hour'],
        timestamp_col='last_transaction',
        amount_col='total_amount',
        customer_id_col='CustomerId'
    )
    transformed = pipeline.fit_transform(df_labeled)

    assert transformed.shape[0] == df_labeled.shape[0], "Number of transformed rows must match input"
    assert isinstance(transformed, np.ndarray), "Output of pipeline must be a NumPy array"

# Test out file directory
def test_output_files_exist():
    assert os.path.exists("../data/processed/data_labeled.csv"), "Processed feature file not found"
    assert os.path.exists("../data/processed/data_targeted.csv"), "Processed target file not found"
    assert os.path.exists("../models/feature_pipeline.pkl"), "Saved pipeline file not found"
