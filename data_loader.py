import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
_model = torch.nn.Sequential(*list(_resnet.children())[:-1])
_model.eval()

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_text_description(ref: str) -> str:
    path = os.path.join("descriptions", f"{ref}.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""


def extract_image_features(ref: str, max_images: int = 3) -> np.ndarray:
    folder_path = os.path.join("images", str(ref))
    features: List[np.ndarray] = []
    if os.path.exists(folder_path):
        image_files = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:max_images]
        for filename in image_files:
            try:
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(img)
                with torch.no_grad():
                    feature_vector = _model(img_tensor.unsqueeze(0)).squeeze().cpu().numpy()
                features.append(feature_vector)
            except Exception:
                continue
    if features:
        return np.mean(features, axis=0)
    return np.zeros(512)


def remove_iqr_outliers(series: pd.Series, factor: float = 1.5) -> np.ndarray:
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return np.clip(series, lower_bound, upper_bound)


def load_data_with_features() -> pd.DataFrame:
    df = pd.read_csv(
        "properties.csv",
        names=[
            "reference",
            "location",
            "price",
            "title",
            "bedrooms",
            "bathrooms",
            "indoor_area",
            "outdoor_area",
            "features",
        ],
    )

    df["price"] = df["price"].astype(str).str.replace("€", "", regex=False).str.replace(",", "", regex=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df.dropna(subset=["price"], inplace=True)

    refs = df["reference"].tolist()

    print(f"Starting parallel feature extraction for {len(df)} properties...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        text_results = list(executor.map(load_text_description, refs))
        image_features_list = list(executor.map(extract_image_features, refs))

    df["description"] = text_results
    image_features_df = pd.DataFrame(image_features_list)
    image_features_df.columns = [f"img_feat_{i}" for i in range(image_features_df.shape[1])]
    image_features_df.index = df.index
    df = pd.concat([df, image_features_df], axis=1)

    df.fillna(
        {
            "features": "",
            "description": "",
            "title": "",
            "bedrooms": 0,
            "bathrooms": 0,
            "indoor_area": 0,
            "outdoor_area": 0,
        },
        inplace=True,
    )

    numerical_cols_for_outlier = ["price", "indoor_area", "outdoor_area"]
    print("Applying IQR-based outlier clipping...")
    for col in numerical_cols_for_outlier:
        positive_mask = df[col] > 0
        df.loc[positive_mask, col] = remove_iqr_outliers(df.loc[positive_mask, col])

    df["location"] = df["location"].astype(str)
    df["bedrooms"] = df["bedrooms"].astype(float)
    df["bathrooms"] = df["bathrooms"].astype(float)

    print("Data loading and preprocessing complete.")
    return df