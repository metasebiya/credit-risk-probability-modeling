import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ProxyTargetEngineer:
    def __init__(self, customer_id_col='CustomerId', timestamp_col='TransactionStartTime', amount_col='Amount', snapshot_date=None):
        self.customer_id_col = customer_id_col
        self.timestamp_col = timestamp_col
        self.amount_col = amount_col
        self.snapshot_date = snapshot_date
        self.kmeans = None
        self.high_risk_cluster = None

    def calculate_rfm(self, df):
        df = df.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        if self.snapshot_date is None:
            self.snapshot_date = df[self.timestamp_col].max() + pd.Timedelta(days=1)

        rfm = df.groupby(self.customer_id_col).agg({
            self.timestamp_col: lambda x: (self.snapshot_date - x.max()).days,
            self.amount_col: ['count', 'sum']
        })

        rfm.columns = ['recency', 'frequency', 'monetary']
        rfm = rfm.reset_index()
        return rfm

    def cluster_rfm(self, rfm, n_clusters=3, random_state=42):
        rfm_scaled = StandardScaler().fit_transform(rfm[['recency', 'frequency', 'monetary']])
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        rfm['cluster'] = self.kmeans.fit_predict(rfm_scaled)
        return rfm

    def assign_high_risk_label(self, rfm_clustered):
        summary = rfm_clustered.groupby('cluster').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).sort_values(by=['frequency', 'monetary'], ascending=[True, True])

        self.high_risk_cluster = summary.index[0]
        rfm_clustered['is_high_risk'] = (rfm_clustered['cluster'] == self.high_risk_cluster).astype(int)
        return rfm_clustered.drop(columns=['cluster'])

    def engineer_target(self, df):
        rfm = self.calculate_rfm(df)
        clustered = self.cluster_rfm(rfm)
        labeled = self.assign_high_risk_label(clustered)
        return labeled

    def merge_with_main(self, df_main, df_rfm_labeled):
        return pd.merge(df_main, df_rfm_labeled[[self.customer_id_col, 'is_high_risk']], on=self.customer_id_col, how='left')
