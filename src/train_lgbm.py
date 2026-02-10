import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .data_loader import RetailDataLoader
import joblib
import os
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
        
        # --- TUNED VALUES ---
        'n_estimators': 2000,       # Increased to ensure full learning
        'learning_rate': 0.01,      # The Sweet Spot (Rank 4)
        'num_leaves': 127,          # The Winner
        'min_child_samples': 100,   # Best for noise handling
        # --------------------
        
        'n_jobs': -1,               
        'verbose': -1,
        'seed': 42
    }
    
    start_time = time.time()
    
    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=['Val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100), # More patience for slower learning
            lgb.log_evaluation(period=200)
        ]
    )
    
    print(f"✅ Training Complete in {time.time() - start_time:.2f}s")
    
    # 4. Generate Business Insights
    print("\n" + "="*40)
    print("      BUSINESS INSIGHTS REPORT      ")
    print("="*40)
    
    # --- A. Accuracy ---
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    print(f"\n1. Model Accuracy")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - MAE:  {mae:.4f} (Avg error per prediction)")

    # --- B. Key Drivers ---
    print("\n2. What drives Sales? (Feature Importance)")
    importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print(importance.to_string(index=False))
    
    # --- C. Price Elasticity Simulation ---
    print("\n3. Price Strategy Simulation")
    print("   (Simulating a 10% price increase on a random sample of 5,000 transactions)")
    
    # Take a sample to test
    sample_indices = np.random.choice(X_val.index, 5000, replace=False)
    sample_X = X_val.loc[sample_indices].copy()
    
    # Predict with CURRENT price
    base_sales = model.predict(sample_X)
    
    # Predict with HIGHER price (+10%)
    sample_X['price'] = sample_X['price'] * 1.10
    new_sales = model.predict(sample_X)
    
    # Calculate Revenue Impact
    current_revenue = np.sum(X_val.loc[sample_indices, 'price'] * base_sales)
    new_revenue = np.sum(sample_X['price'] * new_sales)
    
    vol_change = ((np.sum(new_sales) - np.sum(base_sales)) / np.sum(base_sales)) * 100
    rev_change = ((new_revenue - current_revenue) / current_revenue) * 100
    
    print(f"   - Volume Change:  {vol_change:.2f}% (Drop in units sold)")
    print(f"   - Revenue Change: {rev_change:.2f}% (Change in total money made)")
    
    if rev_change > 0:
        print("   -> CONCLUSION: Demand is Inelastic. Raising prices might increase profit.")
    else:
        print("   -> CONCLUSION: Demand is Elastic. Raising prices will hurt revenue.")
# --- D. Training History Plot ---
    print("\n4. Generating Training Curve...")
    
    # Retrieve the training history
    results = model.evals_result_
    epochs = len(results['Val']['rmse'])
    x_axis = range(0, epochs)
    
    # Plot RMSE
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, results['Val']['rmse'], label='Validation RMSE')
    plt.title('LightGBM Training Error (RMSE) over Time')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('insight_training_curve.png')
    print("   -> Plot saved to 'insight_training_curve.png'")

# --- E. Save the Model ---
    print("\n5. Saving Model...")
    model_path = os.path.join(config.MODEL_DIR, 'lgbm_model.pkl')
    
    # Create directory if it doesn't exist
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Save the object
    joblib.dump(model, model_path)
    print(f"   -> Model saved to: {model_path}")
    print("   -> You can now use 'src/predict.py' without retraining!")
    
    return model