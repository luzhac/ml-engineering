# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LightGBM Model Inference API",
    description="API for making predictions using a trained LightGBM model",
    version="1.0.0"
)

# Global variable to store the model
model = None


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature values as key-value pairs")

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "feature1": 1.5,
                    "feature2": "category_a",
                    "feature3": 10,
                    "feature4": True
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {
                        "feature1": 1.5,
                        "feature2": "category_a",
                        "feature3": 10,
                        "feature4": True
                    },
                    {
                        "feature1": 2.3,
                        "feature2": "category_b",
                        "feature3": 20,
                        "feature4": False
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    prediction: float
    prediction_proba: Optional[List[float]] = None
    timestamp: datetime
    model_version: str = "1.0.0"


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    prediction_probas: Optional[List[List[float]]] = None
    timestamp: datetime
    model_version: str = "1.0.0"
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: datetime


# Load model function
def load_model(model_path: str):
    """Load LightGBM model from text file"""
    global model
    try:
        model = lgb.Booster(model_file=model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    # Update this path to your model file
    MODEL_PATH = "models/lgb_model.txt"  # Change this to your model path

    if not load_model(MODEL_PATH):
        logger.error("Failed to load model on startup")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is healthy and model is loaded"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        timestamp=datetime.now()
    )


# Model info endpoint
@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return {
            "num_features": model.num_feature(),
            "num_trees": model.num_trees(),
            "feature_names": model.feature_name(),
            "feature_importance": dict(zip(
                model.feature_name(),
                model.feature_importance(importance_type='gain').tolist()
            ))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert features to DataFrame
        df = pd.DataFrame([request.features])

        # Make prediction
        prediction = model.predict(df, num_iteration=model.best_iteration)

        # Get prediction probabilities for classification
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict(df, num_iteration=model.best_iteration, predict_type='probability')
                prediction_proba = proba.tolist() if proba.ndim > 1 else [proba.item()]
            except:
                pass

        return PredictionResponse(
            prediction=float(prediction[0]),
            prediction_proba=prediction_proba,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)

        # Make predictions
        predictions = model.predict(df, num_iteration=model.best_iteration)

        # Get prediction probabilities for classification
        prediction_probas = None
        if hasattr(model, 'predict_proba'):
            try:
                probas = model.predict(df, num_iteration=model.best_iteration, predict_type='probability')
                if probas.ndim > 1:
                    prediction_probas = probas.tolist()
                else:
                    prediction_probas = [[p] for p in probas]
            except:
                pass

        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            prediction_probas=prediction_probas,
            timestamp=datetime.now(),
            count=len(predictions)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


# Reload model endpoint
@app.post("/model/reload")
async def reload_model(model_path: str):
    """Reload the model from a new path"""
    if load_model(model_path):
        return {"message": "Model reloaded successfully", "path": model_path}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )