import pandas as pd
import os

def summarize_forecast():
    input_file = 'forecast_results.csv'
    
    print(f"📉 Processing {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        print("   Run 'python -m src.predict' first!")
        return

    # 1. Load the Big File (It takes a second, but Python handles it easily)
    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df):,} rows.")

    # --- Sanity Check (Are the numbers huge, or just the file?) ---
    # Sometimes models predict negative numbers or millions. Let's clamp them.
    # If sales < 0, set to 0. 
    df['predicted_sales'] = df['predicted_sales'].clip(lower=0)
    
    # 2. Report 1: Daily Trend (The "Executive View")
    # Group by Date to see the total revenue/volume projection
    if 'date' in df.columns:
        daily = df.groupby('date')['predicted_sales'].sum().reset_index()
        daily.to_csv('report_daily_forecast.csv', index=False)
        print("   ✅ Created 'report_daily_forecast.csv' (Trends over time)")
    
    # 3. Report 2: Store Performance (The "Manager View")
    # Which stores will sell the most?
    if 'store_id' in df.columns:
        store = df.groupby('store_id')['predicted_sales'].sum().reset_index()
        store = store.sort_values(by='predicted_sales', ascending=False)
        store.to_csv('report_store_forecast.csv', index=False)
        print("   ✅ Created 'report_store_forecast.csv' (Sales by Store)")

    # 4. Report 3: Top Products (The "Inventory View")
    # What should we stock up on?
    # (Assuming there is a product identifier like 'item_id', 'dept_id', or 'sku')
    product_col = None
    for col in ['item_id', 'product_id', 'dept_id', 'sku']:
        if col in df.columns:
            product_col = col
            break
            
    if product_col:
        top_products = df.groupby(product_col)['predicted_sales'].sum().reset_index()
        top_products = top_products.sort_values(by='predicted_sales', ascending=False).head(50)
        top_products.to_csv('report_top_products.csv', index=False)
        print(f"   ✅ Created 'report_top_products.csv' (Top 50 items)")

    print("\n🚀 Done! You can now open these small CSVs in Excel easily.")

if __name__ == "__main__":
    summarize_forecast()