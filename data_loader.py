import pandas as pd
import numpy as np
import os
import torch
from torchvision import models, transforms
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# --- 1. COMPUTER VISION SETUP ---
# Load a pre-trained model for feature extraction
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

# Standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 2. I/O AND FEATURE EXTRACTION FUNCTIONS ---

def load_text_description(ref):
    """Loads text from individual description files."""
    path = os.path.join("descriptions", f"{ref}.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""

def extract_image_features(ref, max_images=3):
    """Loads a maximum of 3 images, extracts features, and averages them."""
    folder_path = os.path.join("images", str(ref))
    features = []
    
    if os.path.exists(folder_path):
        # Limit image loading to the first 'max_images' files found
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_images]
        
        for filename in image_files:
            try:
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img)
                
                with torch.no_grad():
                    feature_vector = model(img_tensor.unsqueeze(0)).squeeze().cpu().numpy()
                    
                features.append(feature_vector)
                
            except Exception:
                continue
    
    if features:
        # Average the 512-dimension features
        return np.mean(features, axis=0)
    else:
        return np.zeros(512)

# --- 3. HELPER FUNCTION FOR IQR OUTLIER CLIPPING ---

def remove_iqr_outliers(series, factor=1.5):
    """Clips extreme values using the IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # Clip values to the calculated bounds
    return np.clip(series, lower_bound, upper_bound)

# --- 4. MAIN LOADING FUNCTION ---

def load_data_with_features():
    """Loads main CSV, cleans price, extracts features, removes outliers, and normalizes."""
    df = pd.read_csv("properties.csv", names=[
        "reference","location","price","title","bedrooms","bathrooms",
        "indoor_area","outdoor_area","features"
    ])

    # df = df.head(100)  # For testing purposes, limit to first 1000 rows
    
    # Initial Price Cleaning and Drop
    df['price'] = df['price'].astype(str).str.replace('€', '').str.replace(',', '')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(subset=['price'], inplace=True)
    
    refs = df["reference"].tolist()
    
    print(f"Starting parallel feature extraction for {len(df)} properties...")
    
    # Parallel Feature Extraction (Text and Image)
    with ThreadPoolExecutor(max_workers=8) as executor:
        text_results = list(executor.map(load_text_description, refs))
        image_features_list = list(executor.map(extract_image_features, refs))

    df["description"] = text_results
    
    # Integrate Image Features
    image_features_df = pd.DataFrame(image_features_list)
    image_features_df.columns = [f'img_feat_{i}' for i in range(image_features_df.shape[1])]
    image_features_df.index = df.index
    df = pd.concat([df, image_features_df], axis=1)

    # Fill NaNs for numerical and text features (0 for numbers, "" for text)
    df.fillna({"features": "", "description": "", "title": "", 
               "bedrooms": 0, "bathrooms": 0, "indoor_area": 0, "outdoor_area": 0}, 
              inplace=True)
    
    # --- Outlier Removal (New Step) ---
    numerical_cols_for_outlier = ["price", "indoor_area", "outdoor_area"]
    
    print("Applying IQR-based outlier clipping...")
    for col in numerical_cols_for_outlier:
        # Only clip on values > 0
        positive_mask = df[col] > 0
        df.loc[positive_mask, col] = remove_iqr_outliers(df.loc[positive_mask, col])

    # --- Normalization (New Step) ---
    normalization_cols = ["price", "indoor_area", "outdoor_area"]
    print("Applying Log(1+X) transformation...")
    for col in normalization_cols:
        # np.log1p is log(1+x), safe for 0 or positive values
        df[col] = np.log1p(df[col]) 

    # Prepare final categorical columns
    df["location"] = df["location"].astype(str)
    df["bedrooms"] = df["bedrooms"].astype(float)
    df["bathrooms"] = df["bathrooms"].astype(float)

    print("Data loading and preprocessing complete.")
    return df