from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import logging

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response, Request
import time
from pydantic import BaseModel, Field, validator

# Counts number of prediction requests
REQUEST_COUNT = Counter("prediction_requests_total", "Total number of prediction requests")
# Measures prediction latency in seconds
PREDICTION_DURATION = Histogram("prediction_duration_seconds", "Prediction latency (seconds)")



class HousingFeatures(BaseModel):
    MedInc: float = Field(..., gt=0, description="Median income, must be positive")
    HouseAge: float = Field(..., ge=0, le=100, description="Age of house, 0-100 years")
    AveRooms: float = Field(..., gt=0, description="Average number of rooms; must be >0")
    AveBedrms: float = Field(..., gt=0, description="Average number of bedrooms; must be >0")
    Population: float = Field(..., gt=0, description="Population must be positive")
    AveOccup: float = Field(..., gt=0, description="Average occupancy must be positive")
    Latitude: float = Field(..., ge=32.0, le=42.0, description="Latitude for California range (32-42)")
    Longitude: float = Field(..., ge=-124.5, le=-114.0, description="Longitude for California range (-124.5 to -114)")

    @validator("AveBedrms")
    def bedrooms_less_than_rooms(cls, v, values):
        rooms = values.get("AveRooms")
        if rooms is not None and v > rooms:
            raise ValueError("Average bedrooms cannot exceed average rooms")
        return v


# Optional: configure logging
logging.basicConfig(filename="logs/api.log", level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = FastAPI(
    title="California Housing Price Predictor",
    description="API for serving predictions with the best ML model registered via MLflow.",
    version="1.0",
)

# Load model from MLflow registry (Production version)
model = mlflow.pyfunc.load_model("models:/CaliforniaHousingModel/Production")

# Define expected input schema
class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(features: HousingFeatures):
    input_df = pd.DataFrame([features.dict()])
    pred = model.predict(input_df)
    prediction = float(pred[0])
    # Basic request/response logging
    logging.info(f"Request: {features.dict()} | Prediction: {prediction}")
    return {"prediction": prediction}

@app.get("/")
def root():
    return {"message": "California Housing Prediction API is running!"}

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
