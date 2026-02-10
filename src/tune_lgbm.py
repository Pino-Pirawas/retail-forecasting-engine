import time
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
# We use RandomizedSearchCV for faster results with high verbosity
from sklearn.model_selection import RandomizedSearchCV
from .data_loader import RetailDataLoader
from . import config

def tune_lightgbm():
    print("🔧 Starting Hyperparameter Tuning...")
    print("   (Using Randomized Search to speed up optimization)")
    
    # 1. Load Data
    loader = RetailDataLoader()
    loader.preprocess()
    df = loader.df_clean
    
    # 2. Subsample (10%)
    print(f"   Original Data: {df.shape[0]:,} rows")
    df_sample = df.sample(frac=0.10, random_state=42)
    print(f"   Tuning Sample: {df_sample.shape[0]:,} rows")
    
    X = df_sample[config.FEATURE_COLUMNS]
    y = df_sample[config.TARGET_COLUMN]
    
    # 3. Define Parameter Distribution
    # We use the aggressive grid you defined
    param_dist = {
        'num_leaves': [127, 255],
        'learning_rate': [0.005, 0.01, 0.015],
        'min_child_samples': [100, 200, 500],
        'n_estimators': [1000, 2000] 
    }
    
    # 4. Setup Model
    lgbm = lgb.LGBMRegressor(
        objective='regression', 
        metric='rmse', 
        boosting_type='gbdt', 
        n_jobs=-1, 
        verbose=-1
    )
    
    # 5. Run Randomized Search
    # n_iter=20 means we check 20 candidates. This prevents the script from running for hours.
    print("\n⏳ Running Search...")
    print("   (You will see a progress update for every step below)")
    
    search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_dist,
        n_iter=20,               # <--- Limits to 20 iterations for speed
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=3,               # <--- High verbosity: You will see every step!
        n_jobs= 1,
        random_state=42
    )
    
    start_time = time.time()
    search.fit(X, y)
    duration = time.time() - start_time
    
    # --- SAVE RESULTS ---
    
    # A. Save the Best Params to TXT
    best_score = -search.best_score_
    
    with open("tuning_best_params.txt", "w") as f:
        f.write("=== LIGHTGBM TUNING RESULTS ===\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tuning Time: {duration:.2f} seconds\n")
        f.write(f"Best RMSE: {best_score:.4f}\n\n")
        f.write("Best Parameters:\n")
        for param, value in search.best_params_.items():
            f.write(f"   {param} = {value}\n")
            
    print(f"\n📄 Summary saved to 'tuning_best_params.txt'")
            
    # B. Save All Trials to CSV
    results_df = pd.DataFrame(search.cv_results_)
    # Keep only relevant columns
    cols = ['param_num_leaves', 'param_learning_rate', 'param_min_child_samples', 'mean_test_score', 'rank_test_score']
    results_df = results_df[cols].sort_values(by='rank_test_score')
    results_df['rmse'] = -results_df['mean_test_score'] 
    
    results_df.to_csv("tuning_log.csv", index=False)
    print(f"📊 Detailed log saved to 'tuning_log.csv'")
    
    # Print to Console as well
    print(f"\n🏆 BEST PARAMETERS FOUND:")
    for param, value in search.best_params_.items():
        print(f"   '{param}': {value}")

if __name__ == "__main__":
    tune_lightgbm()