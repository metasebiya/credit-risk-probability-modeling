"""
data_processing.py - EDA   functions for credit risk probability modeling

This module contains functions to eda performing functions for credit risk probability modeling

Author: [Metasebiya Bizuneh]
Created: June 28, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader

class Dataprocessor:
    def __init__(self, data):
        self.df = data
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])

    def overview_data(self) -> pd.DataFrame:
        if self.df.empty:
            print("\n⚠️ The DataFrame is empty.")
            return self.df

        print("\n📊 Initial Data Description:")
        print(self.df.describe(include='all'))
        print("\n📋 Initial Columns:", self.df.columns.tolist())

        # Shape and basic info
        print(f"\n🧱 Info: {self.df.info()}")
        print(f"\n🧱 Shape: {self.df.shape}")
        print(f"📦 Total Elements: {self.df.size}")
        print(f"\n📂 Data Types:\n{self.df.dtypes}")
        print("\n🧾 Missing Values Per Column:")
        print(self.df.isnull().sum())
        print(f"\n🔍 Duplicate Rows: {self.df.duplicated().sum()}")
        return self.df

    def plot_numerical_distributions(self, bins=30):
        """Plot histograms for all numerical features."""
        num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[num_cols].hist(figsize=(15, 10), bins=bins, edgecolor='black')
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_categorical_distributions(self, max_categories=10):
        """Plot bar plots for categorical features with limited unique values."""
        cat_cols = self.df.select_dtypes(include='object').nunique()
        cat_cols = cat_cols[cat_cols <= max_categories].index

        for col in cat_cols:
            plt.figure(figsize=(8, 4))
            sns.countplot(data=self.df, x=col, order=self.df[col].value_counts().index)
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            plt.close()

    def correlation_heatmap(self):
        """Display correlation matrix for numerical features."""
        corr = self.df.select_dtypes(include=['float64', 'int64']).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()

    def missing_value_summary(self):
        """Return a table of missing values and percentages."""
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing,
            'Percentage (%)': missing_percent
        }).sort_values(by='Missing Values', ascending=False)
        return missing_df[missing_df['Missing Values'] > 0]

    def plot_outliers(self):
        """Plot boxplots for all numerical features to detect outliers."""
        num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns

        for col in num_cols:
            plt.figure(figsize=(8, 2))
            sns.boxplot(x=self.df[col], color='skyblue')
            plt.title(f"Boxplot of {col}")
            plt.tight_layout()
            plt.show()
            plt.close()

if __name__ == "__main__":
    df = "../data/raw/data.csv"
    data = DataLoader()
    all_data = data.load_data(df)
    data_preview = Dataprocessor()
    data_overview = data_preview.overview_data(all_data)
    print(data_overview)