import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DATA_PATHS = {
    'transactions': os.path.join(DATA_DIR, 'dh_transactions.csv'), 
    'store': os.path.join(DATA_DIR, 'dh_store_lookup.csv'),
    'product': os.path.join(DATA_DIR, 'dh_product_lookup.csv'),
    'causal': os.path.join(DATA_DIR, 'dh_causal_lookup.csv')
}

MODEL_DIR = os.path.join(BASE_DIR, 'saved_models', 'retail_lgbm')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Features to Feed into Model
FEATURE_COLUMNS = [
    'price', 
    'is_on_display',      
    'is_in_mailer',       
    'day_of_week_sin', 
    'day_of_week_cos',
    'lag_7_sales',        
    'rolling_mean_4w'     
]
TARGET_COLUMN = 'units_sold'

# Data split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2