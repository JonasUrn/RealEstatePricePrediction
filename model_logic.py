# model_logic.py

# --------------------------------------------------------------------------
# --- Setup (Global Initialization - Runs once when the FastAPI server starts) ---
import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import requests
import io
from catboost import CatBoostRegressor

# Load PyTorch model for feature extraction
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

# Standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the trained CatBoost model
PREDICTIVE_MODEL = CatBoostRegressor()
PREDICTIVE_MODEL.load_model('real_estate_model.cbm')

# --------------------------------------------------------------------------
# --- FEATURE ORDER LIST (CRITICAL FOR ALIGNMENT) ---
# This list ensures the input DataFrame column order matches the model's training order.

# 1. Base Tabular/Text Features (Order as they appear in X_train)
BASE_COLS = [
    'location', 'title', 'bedrooms', 'bathrooms', 
    'indoor_area', 'outdoor_area', 'features', 'description'
]

# 2. Image Feature Columns (512 dimensions)
IMAGE_FEAT_COLS = [f'img_feat_{i}' for i in range(512)]

# 3. Final Required Order
FINAL_FEATURE_ORDER = BASE_COLS + IMAGE_FEAT_COLS
# --------------------------------------------------------------------------


# --- Feature Extraction Functions ---

def process_image(img_stream):
    """Processes a single image stream/file object to extract 512 features."""
    try:
        img = Image.open(img_stream).convert('RGB')
        img_tensor = preprocess(img)
        with torch.no_grad():
            feature_vector = model(img_tensor.unsqueeze(0)).squeeze().cpu().numpy()
        return feature_vector
    except Exception:
        return None

def get_image_features(image_streams_or_urls: list, max_images=3):
    """
    Handles a list of image streams/URLs, extracts features, and averages them.
    """
    features = []
    
    # Process up to max_images
    for item in image_streams_or_urls[:max_images]:
        img_stream = None
        
        # 1. Handle File Stream (BytesIO)
        if isinstance(item, io.BytesIO):
            img_stream = item
        
        # 2. Handle URL
        elif isinstance(item, str) and item.startswith(('http', 'https')):
            try:
                # Use a small timeout for external requests
                response = requests.get(item, timeout=3)
                response.raise_for_status() 
                img_stream = io.BytesIO(response.content)
            except Exception:
                continue # Skip bad URL or download failure
        
        if img_stream:
            feature_vector = process_image(img_stream)
            if feature_vector is not None:
                features.append(feature_vector)

    if features:
        return np.mean(features, axis=0)
    else:
        # Must return a zero vector of the correct size (512) for the DataFrame
        return np.zeros(512) 


# --- Main Prediction Function ---

def make_prediction(data: dict, image_inputs: list):
    """Applies all preprocessing and runs the prediction."""
    
    # 1. Feature Extraction
    image_features = get_image_features(image_inputs)

    # 2. Construct Base DataFrame
    # Create the base DataFrame from the input dict
    X = pd.DataFrame([data])
    
    # 3. Add Image Features
    for i, feat in enumerate(image_features):
        X[f'img_feat_{i}'] = feat
        
    # 4. Apply Training Preprocessing (Must match data_loader.py logic)
    
    # Fill NaN for text/categorical columns (should already be done by Pydantic but safe)
    X['features'] = X['features'].fillna("").astype(str)
    X['description'] = X['description'].fillna("").astype(str)
    X['title'] = X['title'].fillna("").astype(str)
    X["location"] = X["location"].astype(str)
    
    # Log Transformation for Area Features (CRUCIAL)
    # The training data was log-transformed.
    X['indoor_area'] = np.log1p(X['indoor_area'].fillna(0))
    X['outdoor_area'] = np.log1p(X['outdoor_area'].fillna(0))

    # 5. Enforce Final Feature Order (THE FIX)
    # Select and reorder columns to match FINAL_FEATURE_ORDER exactly.
    try:
        X_predict = X[FINAL_FEATURE_ORDER]
    except KeyError as e:
        raise ValueError(f"Missing required feature column: {e}. Check input data structure.")
    
    # 6. Predict
    
    # Predict the log price
    log_price_pred = PREDICTIVE_MODEL.predict(X_predict)
    
    # Invert the log transformation (np.expm1 is the inverse of np.log1p)
    predicted_price = np.expm1(log_price_pred[0])
    
    return max(0, predicted_price) # Ensure price is not negative