import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib

from src.drift_monitor import run_monitoring_pipeline, REPORT_DIR
from src.retrain import execute_retraining_pipeline

app = FastAPI(title="E-Commerce Demand Forecasting API")

MODEL_PATH = os.path.join("models", "demand_model.pkl")


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


class DemandPayload(BaseModel):
    avg_price: float = Field(..., gt=0, description="Average price must be greater than 0")
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    lag_1: float = Field(..., ge=0)
    lag_7: float = Field(..., ge=0)
    rolling_mean_7: float = Field(..., ge=0)


@app.get("/health")
def health_check():
    model = load_model()
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict_demand(payload: DemandPayload):
    model = load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifact not found.")

    features = [[
        payload.avg_price,
        payload.day_of_week,
        payload.month,
        payload.day,
        payload.lag_1,
        payload.lag_7,
        payload.rolling_mean_7
    ]]

    prediction = model.predict(features)[0]
    return {
        "status": "success",
        "predicted_units_sold": float(prediction)
    }


@app.get("/drift-report", response_class=HTMLResponse)
def get_drift_report():
    """Triggers drift monitoring check and serves the interactive HTML report."""
    try:
        run_monitoring_pipeline()
        report_file = os.path.join(REPORT_DIR, "drift_report.html")
        if os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
        raise HTTPException(status_code=500, detail="Drift report file creation failed.")
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