import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from src.features import FEATURE_COLUMNS
from src.model_store import ModelStore

from src.drift_monitor import run_monitoring_pipeline
from src.retrain import execute_retraining_pipeline
from src.config import Settings

app = FastAPI(title="E-Commerce Demand Forecasting API")

def load_model():
    settings = Settings.from_env()
    store = ModelStore(settings.runtime_dir / "models")
    production = store.load_production()

    if production is None:
        return None

    model, metadata = production
    return model


class DemandPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

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
        raise HTTPException(
            status_code=503,
            detail="A compatible trained model is not available.",
        )

    features = [[
        getattr(payload, column)
        for column in FEATURE_COLUMNS
    ]]

    prediction = model.predict(features)[0]
    return {
        "status": "success",
        "predicted_units_sold": float(prediction)
    }


@app.get("/drift-report", response_class=HTMLResponse)
def get_drift_report():
    """Triggers drift monitoring check and serves the interactive HTML report."""
    run_monitoring_pipeline()
    
    report_file = Settings.from_env().report_dir / "drift_report.html"
    if not os.path.exists(report_file):
         raise HTTPException(status_code=404, detail="Drift report not found.")
            
    with open(report_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/retrain")
def trigger_retrain(force: bool = False):
    """Request candidate evaluation and return its actual outcome."""
    return execute_retraining_pipeline(
        drift_threshold_exceeded=force,
    )
