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

    def _clean_transactions(self, trans_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix/handle known transaction anomalies:
        - Negative dollar_sales (likely returns / reversals) -> DROP
        - Zero dollar_sales with positive units -> DROP
        Notes:
        - We only drop rows that cause invalid pricing logic (negative/zero price with positive units).
        - Everything else remains unchanged.
        """
        target = config.TARGET_COLUMN

        # Flags (kept for auditing/debugging; won't affect training unless included in FEATURE_COLUMNS)
        trans_df["flag_negative_sales"] = trans_df["dollar_sales"] < 0
        trans_df["flag_zero_sales_pos_units"] = (trans_df["dollar_sales"] == 0) & (trans_df[target] > 0)

        before = len(trans_df)

        # Drop rows that break the core pricing logic
        drop_mask = trans_df["flag_negative_sales"] | trans_df["flag_zero_sales_pos_units"]
        dropped_negative = int(trans_df["flag_negative_sales"].sum())
        dropped_zero_sales = int(trans_df["flag_zero_sales_pos_units"].sum())

        trans_df = trans_df.loc[~drop_mask].copy()

        after = len(trans_df)
        print("Transaction cleaning summary:")
        print(f"- Dropped negative dollar_sales rows: {dropped_negative}")
        print(f"- Dropped zero dollar_sales with positive units rows: {dropped_zero_sales}")
        print(f"- Rows before: {before:,} -> after: {after:,}")

        return trans_df
        
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

        # FIX: Clean known anomalies before computing price
        trans_df = self._clean_transactions(trans_df)
        
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
        
        # Time features (Week-of-Year Seasonality)
        df['week_of_year_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_of_year_cos'] = np.cos(2 * np.pi * df['week'] / 52)
        
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
        # Keep identifiers so downstream insight/strategy code can segment and export actionable results.
        id_cols = [c for c in ['store', 'upc', 'week'] if c in df.columns]
        self.df_clean = df[id_cols + config.FEATURE_COLUMNS + [config.TARGET_COLUMN]]
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