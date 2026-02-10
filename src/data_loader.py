"""Data loading and preprocessing for Retail Prediction."""
import os
import pandas as pd
import numpy as np
import mlx.core as mx
from . import config

class RetailDataLoader:
    def __init__(self):
        self.paths = config.DATA_PATHS
        self.df = None
        self.df_clean = None
        
    def load_and_merge(self):
        print("Loading CSV files...")
        # 1. Load DataFrames
        trans_df = pd.read_csv(self.paths['transactions'])
        store_df = pd.read_csv(self.paths['store'])
        prod_df = pd.read_csv(self.paths['product'])
        causal_df = pd.read_csv(self.paths['causal'])
        
        # 2. Normalize Column Names (Lowercase)
        trans_df.columns = [c.lower().strip() for c in trans_df.columns]
        store_df.columns = [c.lower().strip() for c in store_df.columns]
        prod_df.columns = [c.lower().strip() for c in prod_df.columns]
        causal_df.columns = [c.lower().strip() for c in causal_df.columns]

        print(f"Transactions loaded: {trans_df.shape}")
        
        # 3. Rename Target to match Config (units -> units_sold)
        if 'units' in trans_df.columns:
            trans_df = trans_df.rename(columns={'units': config.TARGET_COLUMN})
        
        # 4. Calculate Price (The missing column!)
        # Avoid division by zero by replacing 0 units with 1 (rare edge case)
        print("Calculating Price column...")
        trans_df['price'] = trans_df['dollar_sales'] / trans_df[config.TARGET_COLUMN].replace(0, 1)

        # 5. Merging (The Star Schema)
        print("Merging tables...")
        # Merge Product info
        # We assume 'upc' is the key. 
        df = trans_df.merge(prod_df, on='upc', how='left')
        
        # Merge Store info
        df = df.merge(store_df, on='store', how='left')
        
        # Merge Causal (Promotions) - Key is upc, store, week
        # 'week' is in your transactions, so this works perfectly.
        df = df.merge(causal_df, on=['upc', 'store', 'week'], how='left')
        
        # Fill NaN values for promotions
        if 'feature_desc' in df.columns:
            df['feature_desc'] = df['feature_desc'].fillna('None')
        if 'display_desc' in df.columns:
            df['display_desc'] = df['display_desc'].fillna('None')
        
        self.df = df
        return df

    def preprocess(self):
        if self.df is None:
            self.load_and_merge()
            
        df = self.df.copy()
        
        # 6. Feature Engineering
        print("Generating features...")
        
        # Binary flags for promotions
        if 'display_desc' in df.columns:
            df['is_on_display'] = df['display_desc'].apply(lambda x: 0 if x == 'None' or 'Not' in str(x) else 1)
        else:
            df['is_on_display'] = 0 # Default if missing
            
        if 'feature_desc' in df.columns:
            df['is_in_mailer'] = df['feature_desc'].apply(lambda x: 0 if x == 'None' or 'Not' in str(x) else 1)
        else:
            df['is_in_mailer'] = 0
        
        # Time features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
        
        # Lag Features (Sorting is critical here)
        df = df.sort_values(['store', 'upc', 'week'])
        
        # Calculate Lags
        target = config.TARGET_COLUMN
        df['lag_7_sales'] = df.groupby(['store', 'upc'])[target].shift(1)
        df['rolling_mean_4w'] = df.groupby(['store', 'upc'])[target].transform(
            lambda x: x.shift(1).rolling(window=4).mean()
        )
        
        # Drop initial rows that have NaNs due to shifting
        df = df.dropna(subset=['lag_7_sales', 'rolling_mean_4w', 'price'])
        
        # Select final columns
        self.df_clean = df[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]]
        print(f"Final training data shape: {self.df_clean.shape}")
        return self.df_clean

    def get_mlx_arrays(self):
        """Returns Data ready for MLX training"""
        if self.df_clean is None:
            self.preprocess()
            
        # Convert to numpy first
        X_np = self.df_clean[config.FEATURE_COLUMNS].values.astype(np.float32)
        y_np = self.df_clean[config.TARGET_COLUMN].values.astype(np.float32)
        
        # Split into Train/Val
        split_idx = int(len(X_np) * config.TRAIN_RATIO)
        
        return (
            mx.array(X_np[:split_idx]), mx.array(y_np[:split_idx]), # Train
            mx.array(X_np[split_idx:]), mx.array(y_np[split_idx:])  # Val
        )