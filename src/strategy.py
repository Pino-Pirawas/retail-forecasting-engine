import pandas as pd
import numpy as np
import joblib
import os
from .data_loader import RetailDataLoader
from . import config

def generate_strategy():
    print("🧠 Starting Strategic Analysis Engine...")
    
    # 1. Load Model & Data
    model_path = os.path.join(config.MODEL_DIR, 'lgbm_model.pkl')
    if not os.path.exists(model_path):
        print("❌ Model not found! Run training first.")
        return

    model = joblib.load(model_path)
    
    loader = RetailDataLoader()
    loader.preprocess()
    df = loader.df_clean
    
    # Focus on Validation Set (The Future)
    split_idx = int(len(df) * config.TRAIN_RATIO)
    df_val = df.iloc[split_idx:].copy()
    
    # --- INSIGHT 1: PROFIT OPTIMIZATION ---
    print("   1. Calculating Optimal Pricing Strategy...")
    
    # ASSUMPTION: Cost is 60% of Price (40% Margin)
    df_val['assumed_cost'] = df_val['price'] * 0.60
    
    # A. Predict Baseline
    df_val['pred_units_base'] = model.predict(df_val[config.FEATURE_COLUMNS])
    df_val['pred_profit_base'] = (df_val['price'] - df_val['assumed_cost']) * df_val['pred_units_base']
    
    # B. Simulation: Raise Price 10%
    df_high = df_val.copy()
    df_high['price'] = df_high['price'] * 1.10
    
    # Predict with new price
    df_high['pred_units_high'] = model.predict(df_high[config.FEATURE_COLUMNS])
    
    # --- THE FIX IS HERE ---
    # We must copy this result back to df_val so we can save it later
    df_val['pred_units_high'] = df_high['pred_units_high']
    # -----------------------

    # Calculate Profit with new price
    df_high['pred_profit_high'] = (df_high['price'] - df_high['assumed_cost']) * df_high['pred_units_high']
    
    # Compare
    df_val['profit_impact'] = df_high['pred_profit_high'] - df_val['pred_profit_base']
    
    # --- INSIGHT 2: PROMOTION EFFICIENCY ---
    print("   2. Analyzing Promotion Effectiveness...")
    
    df_display = df_val[df_val['is_on_display'] == 1].copy()
    
    if not df_display.empty:
        df_no_display = df_display.copy()
        df_no_display['is_on_display'] = 0
        
        df_display['units_with_promo'] = model.predict(df_display[config.FEATURE_COLUMNS])
        df_display['units_no_promo'] = model.predict(df_no_display[config.FEATURE_COLUMNS])
        
        df_display['promo_lift'] = df_display['units_with_promo'] - df_display['units_no_promo']
    
    # --- REPORT GENERATION ---
    print("   3. Generating Executive Actions...")
    
    # Action 1: "Raise Prices Here"
    raise_price = df_val[df_val['profit_impact'] > 0].copy()
    raise_price = raise_price.sort_values(by='profit_impact', ascending=False).head(50)
    
    # Save Strategy File
    # Now 'pred_units_high' exists in raise_price because we copied it to df_val
    strategy_cols = ['price', 'assumed_cost', 'pred_units_base', 'pred_units_high', 'profit_impact']
    
    # Add identifiers if they exist
    for col in ['store_id', 'item_id', 'date']:
        if col in raise_price.columns:
            strategy_cols.insert(0, col)
            
    raise_price[strategy_cols].to_csv('strategy_price_recommendations.csv', index=False)
    
    print("\n" + "="*50)
    print("   STRATEGIC RECOMMENDATIONS")
    print("="*50)
    print(f"   💰 Opportunity: Found {len(raise_price)} transactions where raising price +10% INCREASES Profit.")
    print(f"      Total Potential Extra Profit (Top 50): ${raise_price['profit_impact'].sum():,.2f}")
    print(f"      -> See 'strategy_price_recommendations.csv'")
    
    if not df_display.empty:
        wasted_promo = df_display[df_display['promo_lift'] < 0.1]
        print(f"   ⚠️  Inefficient Promos: Found {len(wasted_promo)} displays that drove < 0.1 extra sales.")
        print("      -> Recommendation: Stop displays for these items.")
    
    print("="*50)

if __name__ == "__main__":
    generate_strategy()