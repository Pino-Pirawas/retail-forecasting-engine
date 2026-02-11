import time
import lightgbm as lgb
import joblib
import os
from .data_loader import RetailDataLoader
from . import config

def train_lightgbm():
    print("🚀 Initializing Retail Insight Engine (LightGBM)...")
    
    # 1. Load Data
    loader = RetailDataLoader()
    print("   Loading and processing data...")
    loader.preprocess() 
    df = loader.df_clean
    
    print(f"📊 Data Loaded: {df.shape[0]:,} transactions")
    
    # 2. Prepare Training Data
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    
    # Time-based split (Last 20% is validation)
    split_idx = int(len(df) * config.TRAIN_RATIO)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
    
    print(f"   Training Set: {X_train.shape[0]:,} rows")
    print(f"   Validation Set: {X_val.shape[0]:,} rows")
    
    # 3. Train Model (TUNED PARAMETERS)
    print("\n🧠 Learning Patterns...")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,       
        'learning_rate': 0.01,      
        'num_leaves': 127,          
        'min_child_samples': 100,   
        'n_jobs': -1,               
        'verbose': -1,
        'seed': 42
    }
    
    start_time = time.time()
    
    model = lgb.LGBMRegressor(**params)
    
    # We keep the eval_set so LightGBM knows when to stop early
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=['Val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=500) # Print less often to keep logs clean
        ]
    )
    
    print(f"✅ Training Complete in {time.time() - start_time:.2f}s")

    # 4. Save the Model
    print("\n💾 Saving Model...")
    model_path = os.path.join(config.MODEL_DIR, 'lgbm_model.pkl')
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(model, model_path)
    
    print(f"   -> Model saved to: {model_path}")
    print("   -> Next Steps:")
    print("      1. Run 'python -m src.evaluate' to see the score.")
    print("      2. Run 'python -m src.visualize' to see the graphs.")
    
    return model