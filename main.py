from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import json
import io

from model_logic import make_prediction, predict_price_with_gemini

app = FastAPI(title="Real Estate Price Predictor")


class PropertyFeatures(BaseModel):
    location: str
    price: float = 0.0
    title: str
    bedrooms: float
    bathrooms: float
    indoor_area: float
    outdoor_area: float
    features: str
    description: str


@app.post("/api/predict")
async def predict(
    features_json: str = Form(...),
    image_files: list[UploadFile] = File(None),
    image_urls_json: str = Form("[]"),
):
    try:
        data_dict = json.loads(features_json)
        property_data = PropertyFeatures(**data_dict).model_dump()
        url_list = json.loads(image_urls_json)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format for features or URLs: {exc}")

    image_inputs = []
    if image_files:
        for file in image_files:
            image_inputs.append(io.BytesIO(await file.read()))
    image_inputs.extend(url_list)

    try:
        predicted_price = make_prediction(property_data, image_inputs)
        return {"predicted_price": f"€{predicted_price:,.2f}", "raw_price": round(predicted_price, 2)}
    except Exception as exc:
        print(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail="Prediction failed due to internal processing error.")


@app.post("/api/predict_new")
async def predict_new(features_json: str = Form(...)):
    try:
        data_dict = json.loads(features_json)
        PropertyFeatures(**data_dict)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format for features: {exc}")

    try:
        prediction_result = await predict_price_with_gemini(data_dict)
        predicted_price = prediction_result.predicted_price_eur
        return {
            "model": "Gemini (LLM/Search-Based)",
            "predicted_price": f"€{predicted_price:,.2f}",
            "raw_price": round(predicted_price, 2),
            "confidence_level": prediction_result.confidence_level,
            "justification": prediction_result.justification,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Gemini API Service Error: {exc}")
    except Exception as exc:
        print(f"Unexpected prediction error: {exc}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")