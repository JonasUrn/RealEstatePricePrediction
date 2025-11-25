from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from scipy.sparse import hstack

model = joblib.load("model/xgb_model.joblib")
tfidf = joblib.load("model/tfidf.joblib")
loc_encoder = joblib.load("model/location_encoder.joblib")

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

    try:
        loc_val = loc_encoder.transform([item.location])[0]
    except:
        loc_val = 0

    full_text = f"{item.title} {item.features} {item.description}"
    X_text = tfidf.transform([full_text])

    numeric = np.array([
        item.bedrooms,
        item.bathrooms,
        item.indoor_area,
        item.outdoor_area,
        loc_val
    ]).reshape(1, -1)

    X = hstack([X_text, numeric])

    prediction = model.predict(X)[0]

    return {"predicted_price": float(prediction)}
