import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from catboost import CatBoostRegressor, Pool

from data_loader import load_data_with_features

df = load_data_with_features()

print(f"Dataset loaded: {len(df)} properties")
print(f"Price range: €{df['price'].min():,.0f} - €{df['price'].max():,.0f}")
print(f"Price median: €{df['price'].median():,.0f}")

print("\n=== Feature Engineering ===")

df["total_area"] = df["indoor_area"] + df["outdoor_area"]
df["has_outdoor"] = (df["outdoor_area"] > 0).astype(float)
df["outdoor_ratio"] = df["outdoor_area"] / (df["total_area"] + 1)

df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 0.5)
df["room_density"] = df["total_rooms"] / (df["indoor_area"] + 1)
df["area_per_bedroom"] = df["indoor_area"] / (df["bedrooms"] + 1)

df["price_per_sqm"] = df["price"] / (df["total_area"] + 1)

print("Encoding location features...")

df["description_length"] = df["description"].fillna("").str.len()
df["features_length"] = df["features"].fillna("").str.len()
df["title_length"] = df["title"].fillna("").str.len()

print(f"Created {len([c for c in df.columns if c not in ['reference', 'price']])} features")

text_cols = ["description", "features", "title"]
cat_cols = ["location"]

X = df.drop(columns=["price", "reference", "price_per_sqm"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nApplying location target encoding...")
loc_median = X_train.groupby("location")["total_area"].median()
global_median = X_train["total_area"].median()
X_train["loc_area_median"] = X_train["location"].map(loc_median).fillna(global_median)
X_test["loc_area_median"] = X_test["location"].map(loc_median).fillna(global_median)

location_stats = {
    "loc_area_median": loc_median.to_dict(),
    "global_median": float(global_median)
}
with open("location_stats.json", "w") as f:
    json.dump(location_stats, f)
print(f"Location statistics saved to location_stats.json")

print("\n=== Training CatBoost Model ===")

model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    loss_function="RMSE",
    eval_metric="MAE",
    verbose=100,
    allow_writing_files=False,
    task_type="CPU",
    early_stopping_rounds=50,
    random_seed=42,
)

train_pool = Pool(data=X_train, label=y_train, text_features=text_cols, cat_features=cat_cols)
test_pool = Pool(data=X_test, label=y_test, text_features=text_cols, cat_features=cat_cols)

model.fit(train_pool, eval_set=test_pool)

MODEL_PATH = "real_estate_model.cbm"
try:
    model.save_model(MODEL_PATH)
    print(f"\nModel successfully saved to {MODEL_PATH}")
except Exception as e:
    print(f"Error saving model: {e}")

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
mape = mean_absolute_percentage_error(y_test, preds)
rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
r2 = r2_score(y_test, preds)

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
print(f"Mean Absolute Error (MAE):            €{mae:,.0f}")
print(f"Root Mean Squared Error (RMSE):       €{rmse:,.0f}")
print(f"Mean Absolute Percentage Error:       {mape:.2%}")
print(f"R² Score:                             {r2:.4f}")
print("="*50)

feature_importance = model.get_feature_importance(train_pool)
feature_names = X_train.columns.tolist()
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(15)

print("\n=== Top 15 Most Important Features ===")
for idx, row in importance_df.iterrows():
    print(f"{row['feature']:30s}: {row['importance']:,.2f}")
print("="*50)