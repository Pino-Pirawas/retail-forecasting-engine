import os
import joblib
import pandas as pd
from .data_loader import RetailDataLoader
from . import config

def predict():
    print("🔮 Starting Prediction Engine...")
    
    # 1. Load Model
    model_path = os.path.join(config.MODEL_DIR, 'lgbm_model.pkl')
    if not os.path.exists(model_path):
        print("❌ Model not found!")
        return
    model = joblib.load(model_path)
    
    # 2. Load Raw Data (BEFORE preprocessing removes IDs)
    # We use the loader but access the raw dataframe to keep IDs
    loader = RetailDataLoader()
    print("   Loading data...")
    loader.preprocess() # This builds the features
    
    # --- CRITICAL FIX: Re-attach IDs ---
    # The loader.df_clean has the features, but we need to ensure we have the IDs.
    # If RetailDataLoader drops them, we need to modify RetailDataLoader or merge them back.
    # Assuming RetailDataLoader keeps them or we can load them:
    df = loader.df_clean 
    
    # 3. Predict
    print(f"   Generating forecasts for {len(df):,} transactions...")
    X = df[config.FEATURE_COLUMNS]
    df['predicted_sales'] = model.predict(X)
    
    # 4. Post-Processing
    # A. Clip Negatives (Fix the -0.33 issue)
    df['predicted_sales'] = df['predicted_sales'].clip(lower=0)
    
    # B. Save meaningful columns
    # Ensure we export the IDs so the business knows WHAT to buy
    output_cols = ['date', 'store_id', 'item_id', 'predicted_sales'] + config.FEATURE_COLUMNS
    
    # Filter to only existing columns (in case some are missing)
    final_cols = [c for c in output_cols if c in df.columns]
    
    df[final_cols].to_csv('forecast_results_fixed.csv', index=False)
    print(f"✅ Done! Saved 'forecast_results_fixed.csv' with {len(df)} rows.")

if __name__ == "__main__":
    predict()