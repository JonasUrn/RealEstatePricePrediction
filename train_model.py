import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
import numpy as np
from scipy.sparse import hstack


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
    return df


def remove_outliers(df):
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)]

    for col in ["bedrooms", "bathrooms", "indoor_area", "outdoor_area"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


def engineer_features(df):
    df["total_area"] = df["indoor_area"].fillna(0) + df["outdoor_area"].fillna(0)

    df["has_outdoor"] = (df["outdoor_area"] > 0).astype(int)

    df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)

    df["room_density"] = (df["bedrooms"] + df["bathrooms"]) / (df["indoor_area"] + 1)

    return df


def prepare_features(df):
    df["text"] = (
        df["title"].astype(str) + " " +
        df["features"].astype(str) + " " +
        df["description"].astype(str)
    )

    loc_encoder = LabelEncoder()
    df["location_encoded"] = loc_encoder.fit_transform(df["location"])

    numeric_cols = [
        "bedrooms", "bathrooms", "indoor_area", "outdoor_area", "location_encoded",
        "total_area", "has_outdoor", "bed_bath_ratio", "room_density"
    ]
    X_numeric = df[numeric_cols].fillna(0).values

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_text = tfidf.fit_transform(df["text"])

    return X_text, X_numeric, tfidf, loc_encoder


def normalize_data(X_numeric):
    scaler = StandardScaler()
    X_numeric_normalized = scaler.fit_transform(X_numeric)
    return X_numeric_normalized, scaler


def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, use_log):
    preds = model.predict(X_test)

    if use_log:
        preds = np.exp(preds)
        y_test = np.exp(y_test)

    mape = mean_absolute_percentage_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("📌 MAPE:", mape)
    print("📌 RMSE:", rmse)


def save_artifacts(model, tfidf, loc_encoder, scaler, use_log):
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/xgb_model.joblib")
    joblib.dump(tfidf, "model/tfidf.joblib")
    joblib.dump(loc_encoder, "model/location_encoder.joblib")
    joblib.dump(scaler, "model/scaler.joblib")
    joblib.dump({"use_log": use_log}, "model/config.joblib")
    print("Model + preprocessors saved!")


def main():
    df = load_data()
    df = clean_price(df)
    df = remove_outliers(df)
    df = engineer_features(df)

    X_text, X_numeric, tfidf, loc_encoder = prepare_features(df)
    X_numeric_normalized, scaler = normalize_data(X_numeric)

    X = hstack([X_text, X_numeric_normalized])
    y = df["price"]

    use_log = True
    if use_log:
        y = np.log(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, use_log)
    save_artifacts(model, tfidf, loc_encoder, scaler, use_log)


if __name__ == "__main__":
    main()
