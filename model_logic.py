# model_logic.py

import io
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from pydantic import BaseModel as PydanticBaseModel
from torchvision import models, transforms

from catboost import CatBoostRegressor
from google import genai
from google.genai import types


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CLIENT = None
if GEMINI_API_KEY:
    try:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as _init_exc:
        print(f"Warning: failed to initialize Gemini client: {_init_exc}")
        GEMINI_CLIENT = None
else:
    print("Warning: GEMINI_API_KEY environment variable not set; Gemini client disabled.")


class PricePredictionSchema(PydanticBaseModel):
    predicted_price_eur: float
    confidence_level: float
    justification: str


BASE_COLS = [
    "location",
    "title",
    "bedrooms",
    "bathrooms",
    "indoor_area",
    "outdoor_area",
    "features",
    "description",
]

IMAGE_FEAT_COLS = [f"img_feat_{i}" for i in range(512)]
FINAL_FEATURE_ORDER = BASE_COLS + IMAGE_FEAT_COLS


_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
_feature_extractor = torch.nn.Sequential(*list(_resnet.children())[:-1])
_feature_extractor.eval()

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

PREDICTIVE_MODEL = CatBoostRegressor()
PREDICTIVE_MODEL.load_model("real_estate_model.cbm")


def process_image(img_stream: io.BytesIO) -> np.ndarray | None:
    try:
        img = Image.open(img_stream).convert("RGB")
        img_tensor = preprocess(img)
        with torch.no_grad():
            feature_vector = _feature_extractor(img_tensor.unsqueeze(0)).squeeze().cpu().numpy()
        return feature_vector
    except Exception:
        return None


def get_image_features(image_streams_or_urls: List[Any], max_images: int = 3) -> np.ndarray:
    features: List[np.ndarray] = []
    for item in image_streams_or_urls[:max_images]:
        img_stream = None
        if isinstance(item, io.BytesIO):
            img_stream = item
        elif isinstance(item, str) and item.startswith(("http", "https")):
            try:
                response = requests.get(item, timeout=3)
                response.raise_for_status()
                img_stream = io.BytesIO(response.content)
            except Exception:
                continue
        if img_stream:
            feature_vector = process_image(img_stream)
            if feature_vector is not None:
                features.append(feature_vector)
    if features:
        return np.mean(features, axis=0)
    return np.zeros(512)


def encode_location_target(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    loc_median = df.groupby("location")[price_col].median()
    df["loc_target_enc"] = df["location"].map(loc_median).fillna(df[price_col].median())
    loc_count = df["location"].value_counts()
    df["loc_freq"] = df["location"].map(loc_count).fillna(0).astype(float)
    df["loc_freq"] = df["loc_freq"] / len(df)
    return df


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, List[str], List[str]]:
    num_cols = [
        "bedrooms",
        "bathrooms",
        "indoor_area",
        "outdoor_area",
        "total_area",
        "has_outdoor",
        "outdoor_ratio",
        "bed_bath_ratio",
        "total_rooms",
        "room_density",
        "area_per_bedroom",
        "loc_target_enc",
        "loc_freq",
    ]
    num = df[num_cols].values
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    emb = df[emb_cols].values if len(emb_cols) > 0 else np.zeros((len(df), 0))
    X = np.hstack([num, emb])
    return X, num_cols, emb_cols


def normalize_numeric(X: np.ndarray, num_cols_count: int) -> tuple[np.ndarray, Any]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_num = X[:, :num_cols_count]
    X[:, :num_cols_count] = scaler.fit_transform(X_num)
    return X, scaler


def make_prediction(data: Dict[str, Any], image_inputs: List[Any]) -> float:
    image_features = get_image_features(image_inputs)
    X = pd.DataFrame([data])
    for i, feat in enumerate(image_features):
        X[f"img_feat_{i}"] = feat
    X["features"] = X["features"].fillna("").astype(str)
    X["description"] = X["description"].fillna("").astype(str)
    X["title"] = X["title"].fillna("").astype(str)
    X["location"] = X["location"].astype(str)
    X["indoor_area"] = np.log1p(X["indoor_area"].fillna(0))
    X["outdoor_area"] = np.log1p(X["outdoor_area"].fillna(0))
    X_predict = X[FINAL_FEATURE_ORDER]
    log_price_pred = PREDICTIVE_MODEL.predict(X_predict)
    predicted_price = np.expm1(log_price_pred[0])
    return max(0, predicted_price)


async def predict_price_with_gemini(data: Dict[str, Any]) -> PricePredictionSchema:
    class PropertyFeatures(PydanticBaseModel):
        location: str
        bedrooms: float
        bathrooms: float
        indoor_area: float
        title: str
        features: str
        description: str

    property_data = PropertyFeatures(**data)

    prompt = f"""
You are an expert real estate appraiser. Based on the following property data, 
predict the market price in Euros (EUR). Use your general knowledge and real-time 
search if necessary (through the search tool).

PROPERTY DATA:
- Location: {property_data.location}
- Bedrooms: {property_data.bedrooms}
- Bathrooms: {property_data.bathrooms}
- Indoor Area: {property_data.indoor_area} sqm
- Title: {property_data.title}
- Features: {property_data.features}
- Description: {property_data.description}

Provide the predicted price strictly in the required JSON format.
The predicted_price_eur must be a single float number.
"""

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PricePredictionSchema,
                tools=[{"google_search": {}}],
            ),
        )
        json_string = response.text.strip()
        prediction_result = PricePredictionSchema.model_validate_json(json_string)
        return prediction_result
    except Exception as exc:
        raise RuntimeError("Gemini API prediction failed during execution or parsing.") from exc