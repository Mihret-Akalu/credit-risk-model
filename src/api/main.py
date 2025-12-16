from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

from .pydantic_models import CustomerData, BatchPredictionRequest
from src.predict import CreditRiskPredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk probability using alternative data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup."""
    global predictor
    try:
        predictor = CreditRiskPredictor()
        logger.info("CreditRiskPredictor initialized successfully")
        logger.info(f"Model info: {predictor.get_model_info()}")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        raise


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "model-info": "/model-info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if predictor and predictor.model is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}


@app.get("/model-info")
async def model_info():
    """Get model information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    info = predictor.get_model_info()
    return 