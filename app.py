from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


model = joblib.load("model/lgbm_model.joblib")
scaler = joblib.load("model/scaler.joblib")
config = joblib.load("model/config.joblib")

location_columns = config["loc_cols"]

embedder = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()


class PropertyInput(BaseModel):
    location: str
    title: str
    features: str
    description: str
    bedrooms: float
    bathrooms: float
    indoor_area: float
    outdoor_area: float


@app.post("/predict")
def predict_price(item: PropertyInput):

    df = pd.DataFrame([{
        "location": item.location,
        "title": item.title,
        "features": item.features,
        "description": item.description,
        "bedrooms": item.bedrooms,
        "bathrooms": item.bathrooms,
        "indoor_area": item.indoor_area,
        "outdoor_area": item.outdoor_area
    }])

    df["total_area"] = df["indoor_area"] + df["outdoor_area"]
    df["has_outdoor"] = (df["outdoor_area"] > 0).astype(int)
    df["outdoor_ratio"] = df["outdoor_area"] / (df["total_area"] + 1)
    df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["room_density"] = df["total_rooms"] / df["indoor_area"]
    df["area_per_bedroom"] = df["indoor_area"] / (df["bedrooms"] + 1)

    numeric_cols = [
        "bedrooms","bathrooms","indoor_area","outdoor_area",
        "total_area","has_outdoor","outdoor_ratio","bed_bath_ratio",
        "total_rooms","room_density","area_per_bedroom"
    ]

    X_numeric = df[numeric_cols].values
    X_numeric = scaler.transform(X_numeric)

    loc_df = pd.get_dummies(df["location"], prefix="loc")
    for col in location_columns:
        if col not in loc_df:
            loc_df[col] = 0
    loc_df = loc_df[location_columns]
    X_loc = loc_df.values

    full_text = df["description"].iloc[0]
    emb = embedder.encode(full_text)
    X_emb = emb.reshape(1, -1)

    X = np.hstack([X_numeric, X_loc, X_emb])

    pred_log = model.predict(X)[0]
    pred = float(np.exp(pred_log))

    return {"predicted_price": pred}
