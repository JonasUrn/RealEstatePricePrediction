import json
import re
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
from catboost import Pool

class PropertyFeatures(PydanticBaseModel):
    location: str
    bedrooms: float
    bathrooms: float
    indoor_area: float
    title: str
    features: str
    description: str

class PricePredictionSchema(PydanticBaseModel):
    predicted_price_eur: float
    confidence_level: float
    justification: str

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CLIENT = None
print(f"[DEBUG] GEMINI_API_KEY set: {bool(GEMINI_API_KEY)}")
if GEMINI_API_KEY:
    try:
        print(f"[DEBUG] Initializing Gemini client with key: {GEMINI_API_KEY[:10]}...")
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        print("[DEBUG] Gemini client initialized successfully")
    except Exception as _init_exc:
        print(f"[ERROR] Failed to initialize Gemini client: {type(_init_exc).__name__}: {_init_exc}")
        import traceback
        traceback.print_exc()
        GEMINI_CLIENT = None
else:
    print("[WARNING] GEMINI_API_KEY environment variable not set; Gemini client disabled.")


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

def make_prediction(data: Dict[str, Any], image_inputs: List[Any]) -> float:
    image_features = get_image_features(image_inputs)
    X = pd.DataFrame([data])
    for i, feat in enumerate(image_features):
        X[f"img_feat_{i}"] = feat

    X["features"] = X["features"].fillna("").astype(str)
    X["description"] = X["description"].fillna("").astype(str)
    X["title"] = X["title"].fillna("").astype(str)
    X["location"] = X["location"].astype(str)

    X["indoor_area"] = X["indoor_area"].fillna(0)
    X["outdoor_area"] = X["outdoor_area"].fillna(0)
    X["bedrooms"] = X["bedrooms"].fillna(0)
    X["bathrooms"] = X["bathrooms"].fillna(0)

    X["total_area"] = X["indoor_area"] + X["outdoor_area"]
    X["has_outdoor"] = (X["outdoor_area"] > 0).astype(float)
    X["outdoor_ratio"] = X["outdoor_area"] / (X["total_area"] + 1)

    X["total_rooms"] = X["bedrooms"] + X["bathrooms"]
    X["bed_bath_ratio"] = X["bedrooms"] / (X["bathrooms"] + 0.5)
    X["room_density"] = X["total_rooms"] / (X["indoor_area"] + 1)
    X["area_per_bedroom"] = X["indoor_area"] / (X["bedrooms"] + 1)

    X["description_length"] = X["description"].str.len()
    X["features_length"] = X["features"].str.len()
    X["title_length"] = X["title"].str.len()

    try:
        with open("location_stats.json", "r") as f:
            location_stats = json.load(f)
        loc_area_median = location_stats["loc_area_median"]
        global_median = location_stats["global_median"]
        X["loc_area_median"] = X["location"].map(loc_area_median).fillna(global_median)
    except FileNotFoundError:
        print("[WARNING] location_stats.json not found, using default value")
        X["loc_area_median"] = X["total_area"]

    text_cols = ["description", "features", "title"]
    cat_cols = ["location"]
    prediction_pool = Pool(data=X, text_features=text_cols, cat_features=cat_cols)

    predicted_price = PREDICTIVE_MODEL.predict(prediction_pool)[0]
    return max(0, predicted_price)


async def predict_price_with_gemini(data: Dict[str, Any]) -> PricePredictionSchema:
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

                Return a JSON object with these exact fields:
                - predicted_price_eur: float (the predicted price in EUR)
                - confidence_level: float (0.0 to 1.0)
                - justification: string (brief explanation of the estimate)
            """

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        response_text = response.text.strip()
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response_text}")
        
        json_string = json_match.group()
        parsed = json.loads(json_string)
        
        prediction_result = PricePredictionSchema(
            predicted_price_eur=float(parsed.get("predicted_price_eur", 0)),
            confidence_level=float(parsed.get("confidence_level", 0.5)),
            justification=str(parsed.get("justification", "")),
        )
        return prediction_result
    except Exception as exc:
        print(f"[ERROR] Gemini API failed: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Gemini API prediction failed: {exc}") from exc