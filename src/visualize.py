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

    # --- PLOT B: Segmented Price Elasticity Simulation ---
    print("   Generating Segmented Price Elasticity Plot...")

    # Define price changes from -20% to +20%
    price_changes = np.linspace(-0.20, 0.20, 21)

    # Use validation rows, simulate LOCALLY per (store, upc)
    df_val = df.iloc[split_idx:].copy()

    # Build volume tiers based on avg demand per (store, upc)
    target = config.TARGET_COLUMN
    volume_df = (
        df_val.groupby(['store', 'upc'])[target]
        .mean()
        .reset_index()
        .rename(columns={target: 'avg_volume'})
    )

    median_volume = volume_df['avg_volume'].median()

    high_volume_keys = volume_df[volume_df['avg_volume'] >= median_volume].copy()
    low_volume_keys = volume_df[volume_df['avg_volume'] < median_volume].copy()

    def simulate_segment(segment_keys: pd.DataFrame, max_groups: int = 500):
        """
        For a segment of (store, upc) pairs:
        - choose representative row per pair (last row)
        - anchor baseline price to pair median price
        - simulate price changes around that baseline
        - return revenue % impact curve
        """
        if len(segment_keys) == 0:
            return None

        sampled = segment_keys.sample(n=min(max_groups, len(segment_keys)), random_state=42)

        base_rows = []
        base_prices = []

        for _, k in sampled.iterrows():
            g = df_val[(df_val['store'] == k['store']) & (df_val['upc'] == k['upc'])]
            if len(g) == 0:
                continue

            p0 = g['price'].median()
            row = g.iloc[-1][config.FEATURE_COLUMNS].copy()
            row['price'] = p0

            base_rows.append(row)
            base_prices.append(p0)

        if len(base_rows) == 0:
            return None

        sample_X_base = pd.DataFrame(base_rows)
        base_prices_arr = np.array(base_prices, dtype=float)

        # Baseline revenue
        base_sales_vec = model.predict(sample_X_base)
        base_sales_vec = np.maximum(base_sales_vec, 0)
        base_revenue_sum = np.sum(base_prices_arr * base_sales_vec)

        revenue_impacts = []
        for pct in price_changes:
            tmp_X = sample_X_base.copy()
            tmp_prices = base_prices_arr * (1 + pct)
            tmp_X['price'] = tmp_prices

            new_sales_vec = model.predict(tmp_X)
            new_sales_vec = np.maximum(new_sales_vec, 0)

            new_revenue_sum = np.sum(tmp_prices * new_sales_vec)
            rev_pct = ((new_revenue_sum - base_revenue_sum) / base_revenue_sum) * 100
            revenue_impacts.append(rev_pct)

        return revenue_impacts

    high_curve = simulate_segment(high_volume_keys, max_groups=500)
    low_curve = simulate_segment(low_volume_keys, max_groups=500)

    # Plot both curves
    if high_curve is not None:
        axes[1].plot(price_changes*100, high_curve, linewidth=3, marker='o', label='High Volume')
    if low_curve is not None:
        axes[1].plot(price_changes*100, low_curve, linewidth=3, marker='o', label='Low Volume')

    axes[1].set_title('Projected Revenue Impact (Segmented, SKU/Store-anchored)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Price Change (%)')
    axes[1].set_ylabel('Revenue Change (%)')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].axvline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Save
    plt.tight_layout()
    output_file = 'insight_report_final.png'
    plt.savefig(output_file)
    print(f"✅ Done! Saved visualization to '{output_file}'")

if __name__ == "__main__":
    visualize()