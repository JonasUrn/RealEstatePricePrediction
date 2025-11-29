from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import json
import io
import uvicorn
import requests

# Import the prediction logic and the loaded model
from model_logic import make_prediction

app = FastAPI(title="Real Estate Price Predictor")

# Define the structure for the non-image data (the tabular/text features)
class PropertyFeatures(BaseModel):
    location: str
    price: float = 0 # Price is a dummy/optional field as it's the target
    title: str
    bedrooms: float
    bathrooms: float
    indoor_area: float
    outdoor_area: float
    features: str
    description: str

@app.post("/api/predict")
async def predict(
    # Tabular/Text Data: Sent as JSON string in a form field
    features_json: str = Form(..., description="JSON string of the property features (location, bedrooms, etc.)"),
    # Image Files: Multiple files can be uploaded
    image_files: list[UploadFile] = File(None, description="List of image files to upload (Max 3)"),
    # Image URLs: Sent as a list of strings in a form field
    image_urls_json: str = Form("[]", description="JSON string array of image URLs (e.g., [\"url1\", \"url2\"])")
):
    try:
        # 1. Parse JSON inputs
        data_dict = json.loads(features_json)
        # Use Pydantic to validate and clean the data
        property_data = PropertyFeatures(**data_dict).model_dump()
        
        url_list = json.loads(image_urls_json)
        
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format for features or URLs: {e}")

    # 2. Collect ALL image inputs (Files as BytesIO, URLs as strings)
    image_inputs = []
    
    # A. Handle Uploaded Files
    if image_files:
        for file in image_files:
            # Read file bytes into an in-memory stream (BytesIO)
            image_inputs.append(io.BytesIO(await file.read()))
    
    # B. Handle URLs
    image_inputs.extend(url_list)

    # 3. Run Prediction
    try:
        predicted_price = make_prediction(property_data, image_inputs)
        
        return {
            "predicted_price": f"€{predicted_price:,.2f}",
            "raw_price": round(predicted_price, 2)
        }
    except Exception as e:
        # Log the exception for debugging
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed due to internal processing error.")
    