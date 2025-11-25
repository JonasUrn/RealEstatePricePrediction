import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import numpy as np


def load_description(ref):
    desc_path = os.path.join('descriptions', f'{ref}.txt')
    if os.path.exists(desc_path):
        with open(desc_path, encoding='utf-8') as f:
            return f.read()
    return ""


def load_data():
    df = pd.read_csv('properties.csv', names=[
        'reference','location','price','title','bedrooms','bathrooms',
        'indoor_area','outdoor_area','features'
    ])
    df['description'] = df['reference'].apply(load_description)

    if os.path.exists('text_features.csv'):
        print("Loading text-based features (keyword + embeddings)...")
        text_df = pd.read_csv('text_features.csv')
        df = df.merge(text_df, on='reference', how='left')
        print(f"Loaded {len(text_df.columns)-1} text features")
    else:
        print("No text features found. Run generate_text_features.py first for better results.")

    return df


def clean_price(df):
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("€", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df["price"] = df["price"].astype(float)
    df = df[df["price"] > 0]
    return df


def clean_numeric_features(df):
    numeric_cols = ["bedrooms", "bathrooms", "indoor_area", "outdoor_area"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = df[df["indoor_area"] > 0]
    return df


def remove_outliers(df):
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)]

    for col in ["indoor_area", "outdoor_area"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


def engineer_features(df):
    df["total_area"] = df["indoor_area"] + df["outdoor_area"]

    df["has_outdoor"] = (df["outdoor_area"] > 0).astype(int)

    df["outdoor_ratio"] = df["outdoor_area"] / (df["total_area"] + 1)

    df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)

    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

    df["room_density"] = df["total_rooms"] / df["indoor_area"]

    df["area_per_bedroom"] = df["indoor_area"] / (df["bedrooms"] + 1)

    return df


def prepare_features(df):
    location_dummies = pd.get_dummies(df["location"], prefix="loc")

    numeric_cols = [
        "bedrooms", "bathrooms", "indoor_area", "outdoor_area",
        "total_area", "has_outdoor", "outdoor_ratio", "bed_bath_ratio",
        "total_rooms", "room_density", "area_per_bedroom"
    ]
    X_numeric = df[numeric_cols].values

    text_feature_cols = [col for col in df.columns if col.startswith('luxury_keyword') or
                         col.startswith('modern_keyword') or col.startswith('view_keyword') or
                         col.startswith('outdoor_keyword') or col.startswith('condition_keyword') or
                         col.startswith('location_keyword') or col.startswith('text_length') or
                         col.startswith('word_count') or col.startswith('avg_word') or
                         col.startswith('exclamation') or col.startswith('embedding_')]

    has_text_features = len(text_feature_cols) > 0
    if has_text_features:
        X_text = df[text_feature_cols].fillna(0).values
        print(f"Using {len(text_feature_cols)} text features (keywords + embeddings)")
    else:
        X_text = None
        print("No text features found, training without them")

    return X_text, X_numeric, location_dummies.values, location_dummies.columns.tolist()


def normalize_data(X_numeric):
    scaler = StandardScaler()
    X_numeric_normalized = scaler.fit_transform(X_numeric)
    return X_numeric_normalized, scaler


def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_model(model, X_test, y_test, use_log):
    preds = model.predict(X_test)

    if use_log:
        preds = np.exp(preds)
        y_test = np.exp(y_test)

    mape = mean_absolute_percentage_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = np.mean(np.abs(y_test - preds))

    print("MAPE %:", round(mape * 100, 2))
    print("RMSE:", round(rmse, 2))
    print("MAE:", round(mae, 2))


def save_artifacts(model, scaler, location_cols, use_log):
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/xgb_model.joblib")
    joblib.dump(scaler, "model/scaler.joblib")
    joblib.dump({"use_log": use_log, "location_cols": location_cols}, "model/config.joblib")
    print("Model + preprocessors saved!")


def main():
    df = load_data()
    df = clean_price(df)
    df = clean_numeric_features(df)
    df = remove_outliers(df)
    df = engineer_features(df)

    X_text, X_numeric, X_location, location_cols = prepare_features(df)
    X_numeric_normalized, scaler = normalize_data(X_numeric)

    if X_text is not None:
        X = np.hstack([X_numeric_normalized, X_location, X_text])
    else:
        X = np.hstack([X_numeric_normalized, X_location])

    y = df["price"]

    use_log = True
    if use_log:
        y = np.log(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, use_log)
    save_artifacts(model, scaler, location_cols, use_log)


if __name__ == "__main__":
    main()
