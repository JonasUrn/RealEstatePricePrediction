import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import lightgbm as lgb


def load_description(ref):
    path = os.path.join("descriptions", f"{ref}.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    return ""


def load_data():
    df = pd.read_csv("properties.csv", names=[
        "reference","location","price","title","bedrooms","bathrooms",
        "indoor_area","outdoor_area","features"
    ])
    df["description"] = df["reference"].apply(load_description)
    return df


def encode_location_target(df, price_col="price"):
    loc_median = df.groupby("location")[price_col].median()
    df["loc_target_enc"] = df["location"].map(loc_median).fillna(df[price_col].median())
    loc_count = df["location"].value_counts()
    df["loc_freq"] = df["location"].map(loc_count).fillna(0).astype(float)
    df["loc_freq"] = df["loc_freq"] / len(df)
    return df


def clean_price(df):
    df["price"] = (
        df["price"].astype(str)
        .str.replace("€","")
        .str.replace(",","")
        .str.replace(" ","")
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    return df


def clean_numeric(df):
    cols = ["bedrooms","bathrooms","indoor_area","outdoor_area"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    df = df[df["indoor_area"] > 0]
    return df


def engineer(df):
    df["total_area"] = df["indoor_area"] + df["outdoor_area"]
    df["has_outdoor"] = (df["outdoor_area"] > 0).astype(int)
    df["outdoor_ratio"] = df["outdoor_area"] / (df["total_area"] + 1)
    df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["room_density"] = df["total_rooms"] / df["indoor_area"]
    df["area_per_bedroom"] = df["indoor_area"] / (df["bedrooms"] + 1)
    return df


def embed_text(df, n_components=None):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    df["title"] = df["title"].fillna("")
    df["features"] = df["features"].fillna("")
    df["description"] = df["description"].fillna("")

    full_texts = (df["title"] + " " + df["features"] + " " + df["description"]).tolist()
    embeddings = np.vstack([model.encode(str(t)) for t in full_texts])

    max_components = min(embeddings.shape[0], embeddings.shape[1])
    if n_components is None:
        n_components = min(20, max(2, embeddings.shape[0] // 10))
    n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)

    for i in range(embeddings_reduced.shape[1]):
        df[f"emb_{i}"] = embeddings_reduced[:, i]

    return df, pca



def prepare_features(df):
    num_cols = [
        "bedrooms","bathrooms","indoor_area","outdoor_area",
        "total_area","has_outdoor","outdoor_ratio","bed_bath_ratio",
        "total_rooms","room_density","area_per_bedroom",
        "loc_target_enc","loc_freq",
    ]
    num = df[num_cols].values
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    emb = df[emb_cols].values if len(emb_cols) > 0 else np.zeros((len(df), 0))
    X = np.hstack([num, emb])
    return X, num_cols, emb_cols


def normalize_numeric(X, num_cols_count):
    scaler = StandardScaler()
    X_num = X[:, :num_cols_count]
    X[:, :num_cols_count] = scaler.fit_transform(X_num)
    return X, scaler


def train_model(X_train, y_train, X_val=None, y_val=None):
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=32,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    if X_val is not None and y_val is not None:
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=100,
            )
        except TypeError:
            try:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50)],
                    verbose=100,
                )
            except TypeError:
                print("Warning: installed LightGBM does not support early stopping via sklearn API; fitting without early stopping.")
                model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    return model


def main():
    df = load_data()
    df = clean_price(df)
    df = clean_numeric(df)
    df = engineer(df)
    df = encode_location_target(df, price_col="price")

    df, pca = embed_text(df, n_components=None)

    X, num_cols, emb_cols = prepare_features(df)
    X, scaler = normalize_numeric(X, len(num_cols))

    y = df["price"]
    cap = y.quantile(0.99)
    print("Capping prices at 99th percentile:", cap)
    y_capped = y.clip(upper=cap)

    y_log = np.log1p(y_capped)

    print("Features shape:", X.shape)
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"Feature NaNs: {nan_count}, infs: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        print("Replacing NaN/inf in feature matrix with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    model = train_model(X_train, y_train, X_test, y_test)

    preds = np.expm1(model.predict(X_test))
    real = np.expm1(y_test)

    mape = mean_absolute_percentage_error(real, preds)
    rmse = np.sqrt(mean_squared_error(real, preds))
    mae = np.mean(np.abs(real - preds))

    print("MAPE %:", round(mape * 100, 2))
    print("RMSE:", round(rmse, 2))
    print("MAE:", round(mae, 2))

    try:
        feat_names = list(num_cols) + list(emb_cols)
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        else:
            fi = model.booster_.feature_importance() if hasattr(model, "booster_") else None
        if fi is not None:
            top_idx = np.argsort(fi)[::-1][:20]
            print("Top features:")
            for i in top_idx:
                name = feat_names[i] if i < len(feat_names) else f"f_{i}"
                print(f"  {name}: {fi[i]}")
    except Exception:
        pass

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/lgbm_model.joblib")
    joblib.dump(scaler, "model/scaler.joblib")
    joblib.dump(pca, "model/pca.joblib")
    joblib.dump({"num_cols": num_cols, "emb_cols": emb_cols}, "model/config.joblib")


if __name__ == "__main__":
    main()
