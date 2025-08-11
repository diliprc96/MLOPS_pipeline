from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import logging

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response, Request
import time

# Counts number of prediction requests
REQUEST_COUNT = Counter("prediction_requests_total", "Total number of prediction requests")
# Measures prediction latency in seconds
PREDICTION_DURATION = Histogram("prediction_duration_seconds", "Prediction latency (seconds)")


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
