"""
Prediction API

FastAPI service for real-time delay predictions.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Request body for prediction."""
    stop_id: str
    route_id: str
    hour: Optional[int] = None  # Defaults to current hour
    day_of_week: Optional[int] = None  # Defaults to current day
    
class PredictionResponse(BaseModel):
    """Response from prediction."""
    stop_id: str
    route_id: str
    risk_level: str  # 'low', 'moderate', 'high'
    probability: float
    predicted_delay_min: float
    confidence: str
    model_version: str
    features_used: dict
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str]

# ============================================================================
# Model Management
# ============================================================================

class ModelManager:
    """Manage model loading and caching."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize model manager.
        
        Args:
            model_path: Path to model file. If None, uses MODEL_PATH env var.
        """
        self.model_path = model_path or os.getenv('MODEL_PATH', 'artifacts/models/production/model.pkl')
        self.trainer = None
        self.model_version = None
        
    def load(self) -> bool:
        """
        Load model from disk.
        
        Returns:
            True if successful.
        """
        try:
            from src.models.train import ModelTrainer
            
            if not Path(self.model_path).exists():
                logger.error(f"Model not found: {self.model_path}")
                return False
            
            self.trainer = ModelTrainer.load(self.model_path)
            self.model_version = Path(self.model_path).stem
            
            logger.info(f"Loaded model: {self.model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.trainer is not None
    
    def predict(self, features: np.ndarray) -> tuple:
        """
        Make prediction.
        
        Args:
            features: Feature vector.
            
        Returns:
            Tuple of (prediction, probability).
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        return self.trainer.predict(features.reshape(1, -1))

# Global model manager
model_manager = ModelManager()

# ============================================================================
# Feature Fetching
# ============================================================================

class FeatureFetcher:
    """Fetch features for prediction."""
    
    def __init__(self, data_dir: str = 'data/processed'):
        """
        Initialize feature fetcher.
        
        Args:
            data_dir: Directory with processed data.
        """
        self.data_dir = Path(data_dir)
        self.cache = {}
        
    def get_route_stats(self, route_id: str) -> dict:
        """Get historical stats for route."""
        # In production, this would query BigQuery
        # For now, return defaults or cached values
        return {
            'route_avg_delay': 0.5,
            'route_std_delay': 2.0,
            'route_delay_rate': 0.08
        }
    
    def get_stop_stats(self, stop_id: str) -> dict:
        """Get historical stats for stop."""
        return {
            'stop_avg_delay': 0.6,
            'stop_delay_rate': 0.10
        }
    
    def get_route_stop_stats(self, route_id: str, stop_id: str) -> dict:
        """Get stats for route+stop combination."""
        return {
            'route_stop_avg_delay': 0.7,
            'route_stop_delay_rate': 0.12
        }
    
    def fetch_features(
        self,
        stop_id: str,
        route_id: str,
        hour: Optional[int] = None,
        day_of_week: Optional[int] = None
    ) -> dict:
        """
        Fetch all features for prediction.
        
        Args:
            stop_id: Stop identifier.
            route_id: Route identifier.
            hour: Hour of day (0-23).
            day_of_week: Day of week (0-6).
            
        Returns:
            Dictionary of features.
        """
        from datetime import datetime
        
        # Use current time if not provided
        now = datetime.now()
        hour = hour if hour is not None else now.hour
        day_of_week = day_of_week if day_of_week is not None else now.weekday()
        
        # Build feature dict
        features = {
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_rush_hour': 1 if hour in [7, 8, 9, 16, 17, 18] else 0,
            'stop_sequence': 5,  # Default, would come from GTFS static
        }
        
        # Add historical stats
        features.update(self.get_route_stats(route_id))
        features.update(self.get_stop_stats(stop_id))
        features.update(self.get_route_stop_stats(route_id, stop_id))
        
        return features

feature_fetcher = FeatureFetcher()

# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    # Startup
    logger.info("Starting up...")
    success = model_manager.load()
    if success:
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model not loaded - predictions will fail")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="TransLink Delay Predictor API",
    description="Real-time transit delay predictions",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "service": "TransLink Delay Predictor API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded() else "unhealthy",
        model_loaded=model_manager.is_loaded(),
        model_version=model_manager.model_version
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a delay prediction.
    
    Args:
        request: Prediction request with stop_id, route_id, etc.
        
    Returns:
        Prediction response with risk level and probability.
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Fetch features
        features_dict = feature_fetcher.fetch_features(
            stop_id=request.stop_id,
            route_id=request.route_id,
            hour=request.hour,
            day_of_week=request.day_of_week
        )
        
        # Convert to array (must match training order)
        feature_names = model_manager.trainer.feature_names
        features_array = np.array([features_dict.get(f, 0) for f in feature_names])
        
        # Predict
        pred, proba = model_manager.predict(features_array)
        probability = float(proba[0])
        
        # Determine risk level
        if probability > 0.7:
            risk_level = "high"
            confidence = "High confidence"
        elif probability > 0.4:
            risk_level = "moderate"
            confidence = "Medium confidence"
        else:
            risk_level = "low"
            confidence = "High confidence"
        
        # Estimate delay (rough: probability * max_expected_delay)
        predicted_delay = probability * 15  # Max 15 min delay expected
        
        from datetime import datetime
        
        return PredictionResponse(
            stop_id=request.stop_id,
            route_id=request.route_id,
            risk_level=risk_level,
            probability=round(probability, 4),
            predicted_delay_min=round(predicted_delay, 1),
            confidence=confidence,
            model_version=model_manager.model_version or "unknown",
            features_used=features_dict,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/routes", response_model=List[str])
async def get_routes():
    """Get list of available routes."""
    # In production, query from database
    return ["099", "009", "010", "025", "041"]  # Example routes

@app.get("/stops/{route_id}", response_model=List[dict])
async def get_stops(route_id: str):
    """Get stops for a route."""
    # In production, query from GTFS static
    return [
        {"stop_id": "1234", "stop_name": "Main St & Broadway"},
        {"stop_id": "1235", "stop_name": "Broadway & Cambie"}
    ]

# ============================================================================
# Main
# ============================================================================

def main():
    """Run API server."""
    import uvicorn
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    uvicorn.run(
        'src.api.main:app',
        host=host,
        port=port,
        reload=True  # Disable in production
    )

if __name__ == '__main__':
    main()
