import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from . import config
from .data_loader import RetailDataLoader

def visualize():
    print("🎨 Starting Visualization Engine...")

    # 1. Load the Saved Model
    model_path = os.path.join(config.MODEL_DIR, 'lgbm_model.pkl')
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}. Run 'python main.py' first!")
        return
    
    print(f"   Loading model from {model_path}...")
    model = joblib.load(model_path)

    # 2. Load Data (We need this to simulate the price changes)
    print("   Loading data for simulation...")
    loader = RetailDataLoader()
    loader.preprocess()
    df = loader.df_clean
    
    # Split data to get the Validation set (Same as training)
    split_idx = int(len(df) * config.TRAIN_RATIO)
    X_train = df[config.FEATURE_COLUMNS].iloc[:split_idx]
    X_val = df[config.FEATURE_COLUMNS].iloc[split_idx:]
    
    print(f"   Using {len(X_val):,} validation rows for simulation.")

    # 3. Setup the Canvas
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- PLOT A: Feature Importance ---
    print("   Generating Feature Importance Plot...")
    importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=importance.head(10), 
        ax=axes[0], 
        palette='viridis', 
        hue='Feature', 
        legend=False
    )
    axes[0].set_title('Top 10 Drivers of Sales', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Importance Score')
    axes[0].grid(True, axis='x', alpha=0.3)

    # --- PLOT B: Price Elasticity Simulation ---
    print("   Generating Price Elasticity Plot...")
    
    # Define price changes from -20% to +20%
    price_changes = np.linspace(-0.20, 0.20, 21) 
    revenue_impacts = []
    
    # Use a random sample of 5,000 rows to speed up the simulation
    sample_indices = np.random.choice(X_val.index, 5000, replace=False)
    sample_X_base = X_val.loc[sample_indices].copy()
    
    # 1. Calculate Baseline Revenue (Current Price)
    base_sales_vec = model.predict(sample_X_base)
    base_sales_vec = np.maximum(base_sales_vec, 0) # Clip negatives
    base_revenue_sum = np.sum(sample_X_base['price'] * base_sales_vec)

    # 2. Simulate Scenarios
    for pct in price_changes:
        # Create a temp copy and adjust price
        tmp_X = sample_X_base.copy()
        tmp_X['price'] = tmp_X['price'] * (1 + pct)
        
        # Predict Sales
        new_sales_vec = model.predict(tmp_X)
        new_sales_vec = np.maximum(new_sales_vec, 0) # Clip negatives
        
        # Calculate New Revenue
        new_revenue_sum = np.sum(tmp_X['price'] * new_sales_vec)
        
        # Calculate % Difference
        rev_pct = ((new_revenue_sum - base_revenue_sum) / base_revenue_sum) * 100
        revenue_impacts.append(rev_pct)

    # 3. Draw the Curve
    axes[1].plot(price_changes*100, revenue_impacts, color='green', linewidth=3, marker='o')
    axes[1].set_title('projected Revenue Impact of Pricing Strategy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Price Change (%)')
    axes[1].set_ylabel('Revenue Change (%)')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].axvline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].grid(True, alpha=0.3)

    # Annotate the "Sweet Spot" (e.g., +10%)
    idx_10 = (np.abs(price_changes - 0.10)).argmin() # Find index closest to 10%
    val_10 = revenue_impacts[idx_10]
    
    axes[1].annotate(f'+{val_10:.1f}% Rev', 
                     xy=(10, val_10), 
                     xytext=(15, val_10-2),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    # Save
    plt.tight_layout()
    output_file = 'insight_report_final.png'
    plt.savefig(output_file)
    print(f"✅ Done! Saved visualization to '{output_file}'")

if __name__ == "__main__":
    visualize()