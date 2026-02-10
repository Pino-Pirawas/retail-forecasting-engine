import os
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .data_loader import RetailDataLoader
from . import config

def evaluate():
    print("📊 Starting Model Evaluation...")
    
    # 1. Load Model
    model_path = os.path.join(config.MODEL_DIR, 'lgbm_model.pkl')
    if not os.path.exists(model_path):
        print("❌ Model not found. Train it first!")
        return
        
    model = joblib.load(model_path)
    
    # 2. Load Data
    loader = RetailDataLoader()
    loader.preprocess()
    df = loader.df_clean
    
    # Select Validation Data
    split_idx = int(len(df) * config.TRAIN_RATIO)
    df_val = df.iloc[split_idx:]
    
    X_val = df_val[config.FEATURE_COLUMNS]
    y_val = df_val[config.TARGET_COLUMN]
    
    # 3. Score
    print(f"   Scoring {len(df_val):,} rows...")
    preds = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    
    print("\n" + "="*30)
    print("   MODEL REPORT CARD")
    print("="*30)
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R^2:  {r2:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()