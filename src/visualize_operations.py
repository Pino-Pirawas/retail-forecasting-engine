import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_operations():
    print("🎨 Generating Operational Insights...")
    
    # 1. Load Forecast Data
    input_file = 'forecast_results.csv'
    if not os.path.exists(input_file):
        print("❌ Forecast file not found. Run 'python -m src.predict' first.")
        return
        
    df = pd.read_csv(input_file)
    
    # 2. Setup Canvas
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- PLOT 1: Weekly Sales Rhythm ---
    # Recover Day from Sin/Cos features
    # angle = arctan2(sin, cos)
    angles = np.arctan2(df['day_of_week_sin'], df['day_of_week_cos'])
    # Convert radians to 0-6 index
    # (This is an approximation since we don't have the original date, but it reveals the cycle)
    df['day_index'] = (angles * 7 / (2 * np.pi)).round().astype(int) % 7
    
    # Group and Plot
    daily_sales = df.groupby('day_index')['predicted_sales'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Map index to day names (0 might be Mon or Sun depending on original encoding, 
    # but the *pattern* is what matters)
    axes[0].plot(days, daily_sales.values, marker='o', linewidth=3, color='#ff7f0e')
    axes[0].set_title('Average Daily Sales Rhythm', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Avg Units Sold')
    axes[0].grid(True, alpha=0.3)
    
    # --- PLOT 2: Promotion Efficiency ---
    # Boxplot to show the "Lift" (or lack thereof)
    # Filter out extreme outliers for a cleaner view
    df_box = df[df['predicted_sales'] < 10].copy()
    
    sns.boxplot(x='is_on_display', y='predicted_sales', data=df_box, ax=axes[1], palette='Set2')
    axes[1].set_xticklabels(['No Display', 'On Display'])
    axes[1].set_title('Impact of Displays on Sales', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('')
    
    # --- PLOT 3: The Demand Curve ---
    # Scatter plot of Price vs Sales
    # Sample 5,000 points to avoid crashing the plotter
    sample = df.sample(n=5000, random_state=42)
    
    sns.scatterplot(
        x='price', 
        y='predicted_sales', 
        hue='is_on_display', 
        data=sample, 
        ax=axes[2], 
        palette='coolwarm', 
        alpha=0.6
    )
    axes[2].set_title('Price vs. Sales (Demand Curve)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Price ($)')
    axes[2].set_ylabel('Predicted Sales')
    axes[2].legend(title='Display')
    
    plt.tight_layout()
    plt.savefig('operational_insights.png')
    print("✅ Saved 'operational_insights.png'")

if __name__ == "__main__":
    visualize_operations()