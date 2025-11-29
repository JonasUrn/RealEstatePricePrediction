import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from catboost import CatBoostRegressor, Pool

from data_loader import load_data_with_features

df = load_data_with_features()

text_cols = ["description", "features", "title"]
cat_cols = ["location"]

X = df.drop(columns=["price", "reference"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting CatBoost model training...")

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=4,
    loss_function="RMSE",
    eval_metric="MAE",
    verbose=50,
    allow_writing_files=False,
    task_type="CPU",
)

train_pool = Pool(data=X_train, label=y_train, text_features=text_cols, cat_features=cat_cols)
test_pool = Pool(data=X_test, label=y_test, text_features=text_cols, cat_features=cat_cols)

model.fit(train_pool, eval_set=test_pool)

MODEL_PATH = "real_estate_model.cbm"
try:
    model.save_model(MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")
except Exception as e:
    print(f"Error saving model: {e}")

preds_log = model.predict(X_test)

preds_original_scale = np.expm1(preds_log)
y_test_original_scale = np.expm1(y_test)

mae = mean_absolute_error(y_test_original_scale, preds_original_scale)
mape = mean_absolute_percentage_error(y_test_original_scale, preds_original_scale)
rmse = float(np.sqrt(mean_squared_error(y_test_original_scale, preds_original_scale)))

print("\n--- RESULTS ---")
print(f"Final Mean Absolute Error (MAE): €{mae:,.0f}")
print(f"Final Mean Absolute Percentage Error (MAPE): {mape:.2%}")
print(f"Final Root Mean Squared Error (RMSE): €{rmse:,.0f}")
print("---")