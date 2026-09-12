import os
import joblib
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
from src.drift_monitor import run_monitoring_pipeline, REPORT_DIR
from src.retrain import execute_retraining_pipeline

# Initialize FastAPI app
app = FastAPI(
    title="E-Commerce Demand Forecasting API",
    description="Asynchronous MLOps microservice for dynamic demand forecasting.",
    version="1.0.0"
)

MODEL_PATH = os.path.join("models", "demand_model.pkl")

# Pydantic schema for incoming API request payloads
class DemandPredictionInput(BaseModel):
    avg_price: float = Field(..., gt=0, description="Average product price in USD")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month of year (1-12)")
    day: int = Field(..., ge=1, le=31, description="Day of month (1-31)")
    lag_1: float = Field(..., ge=0, description="Demand from 1 day prior")
    lag_7: float = Field(..., ge=0, description="Demand from 7 days prior")
    rolling_mean_7: float = Field(..., ge=0, description="7-day rolling average demand")

# Pydantic schema for outgoing API response
class DemandPredictionOutput(BaseModel):
    predicted_units_sold: float
    status: str


def load_model():
    """Loads saved model artifact from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Run 'python -m src.train' first.")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint for container orchestrators."""
    if os.path.exists(MODEL_PATH):
        return {"status": "healthy", "model_loaded": "true"}
    return {"status": "degraded", "model_loaded": "false"}


@app.post("/predict", response_model=DemandPredictionOutput)
def predict(payload: DemandPredictionInput) -> DemandPredictionOutput:
    """Predicts daily product demand based on feature telemetry payload."""
    try:
        model = load_model()
        
        # Format payload into DataFrame matching XGBoost feature order
        feature_order = ["avg_price", "day_of_week", "month", "day", "lag_1", "lag_7", "rolling_mean_7"]
        input_data = pd.DataFrame([payload.model_dump()])[feature_order]

        # Execute prediction
        prediction = float(model.predict(input_data)[0])
        prediction_rounded = max(0.0, round(prediction, 2))

        return DemandPredictionOutput(
            predicted_units_sold=prediction_rounded,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift-report", response_class=HTMLResponse)
def get_drift_report():
    """Triggers drift monitoring check and serves the interactive HTML report."""
    try:
        run_monitoring_pipeline()
        report_file = os.path.join(REPORT_DIR, "drift_report.html")
        if os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
        raise HTTPException(status_code=500, detail="Report generation failed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
def trigger_retrain(force: bool = False):
    """Triggers automated model retraining pipeline."""
    try:
        success = execute_retraining_pipeline(drift_threshold_exceeded=force)
        if success:
            return {"status": "success", "message": "Model retrained and updated successfully."}
        return {"status": "skipped", "message": "Retraining skipped. No drift threshold exceeded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))