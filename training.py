import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from catboost import CatBoostRegressor, Pool

# Import the feature loading function from the adjacent file
from data_loader import load_data_with_features

# --- 1. DATA LOADING ---
# Data loaded now has a log-transformed 'price' and pre-processed features.
df = load_data_with_features()

# --- 2. FEATURE DEFINITION & SPLITTING ---

text_cols = ["description", "features", "title"]
cat_cols = ["location"]

X = df.drop(columns=["price", "reference"])
y = df["price"] # This 'price' column is log-transformed

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. MODEL TRAINING ---
print("Starting CatBoost model training...")

# Use safer, less aggressive parameters to combat the previous overfitting issue
model = CatBoostRegressor(
    iterations=1000,          # Reduced iterations
    learning_rate=0.05,      # Slower, safer learning rate
    depth=4,                 # Shallower trees
    loss_function='RMSE',
    eval_metric='MAE',
    verbose=50,              # Print progress every 50 iterations
    allow_writing_files=False,
    task_type="CPU" 
)

train_pool = Pool(
    data=X_train, 
    label=y_train, 
    text_features=text_cols, 
    cat_features=cat_cols
)

test_pool = Pool(
    data=X_test, 
    label=y_test, 
    text_features=text_cols, 
    cat_features=cat_cols
)

model.fit(train_pool, eval_set=test_pool)

# --- 4. MODEL SAVING (NEW STEP) ---
MODEL_PATH = 'real_estate_model.cbm'
try:
    model.save_model(MODEL_PATH)
    print(f"\n✅ Model successfully saved to {MODEL_PATH}")
except Exception as e:
    print(f"\n❌ Error saving model: {e}")

# --- 5. EVALUATION ---
preds_log = model.predict(X_test)

# Convert predictions and actual values back to the original currency scale
preds_original_scale = np.expm1(preds_log)
y_test_original_scale = np.expm1(y_test)

mae = mean_absolute_error(y_test_original_scale, preds_original_scale)
mape = mean_absolute_percentage_error(y_test_original_scale, preds_original_scale)
rmse = root_mean_squared_error(y_test_original_scale, preds_original_scale)

print("\n--- RESULTS ---")
print(f"Final Mean Absolute Error (MAE): €{mae:,.0f} (Error on original currency scale)")
print(f"Final Mean Absolute Percentage Error (MAPE): {mape:.2%}")
print(f"Final Root Mean Squared Error (RMSE): €{rmse:,.0f}")
print("---")