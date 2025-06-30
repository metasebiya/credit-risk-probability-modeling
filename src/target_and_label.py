import pandas as pd
from proxy_target_engineer import ProxyTargetEngineer
from data_processing import build_full_pipeline
import joblib

# Load raw data
df_raw = pd.read_csv("data/raw/data.csv")

# --- Step 1: Generate proxy target ---
proxy = ProxyTargetEngineer(
    customer_id_col='CustomerId',
    timestamp_col='TransactionStartTime',
    amount_col='Amount'
)
rfm_labeled = proxy.engineer_target(df_raw)
df_labeled = proxy.merge_with_main(df_raw, rfm_labeled)

# --- Step 2: Define target and raw features ---
y = df_labeled['is_high_risk']
X_raw = df_labeled.copy()

# --- Step 3: Transform features ---
pipeline = build_full_pipeline(
    numeric_features=['transaction_count', 'total_amount', 'avg_amount', 'std_amount'],
    categorical_features=['month', 'hour'],
    timestamp_col='TransactionStartTime',
    amount_col='Amount',
    customer_id_col='CustomerId'
)
X = pipeline.fit_transform(X_raw)

# --- Step 4: Save processed data ---
pd.DataFrame(X).to_csv("data/processed/data_labeled.csv", index=False)
y.to_csv("data/processed/data_targeted.csv", index=False)

# Optional: Save the pipeline
joblib.dump(pipeline, "models/feature_pipeline.pkl")
