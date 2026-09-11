"""FastAPI routes for AIOps operations.

ponytail: Endpoints map directly to the ML baseline pipelines. 
No intermediate abstraction layers; just schema validation and Pandas DataFrame conversion.
"""
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    HealthResponse,
    PredictIncidentRequest,
    RcaRequest,
    TelemetryBatchRequest,
)
from app.data.preprocess import preprocess_telemetry
from app.models.anomaly_isolation_forest import IsolationForestDetector
from app.models.incident_predictor import IncidentPredictorBaseline
from app.models.rca_engine import rank_root_causes
from app.models.rca_explainer import generate_rca_explanation
from app.models.severity_classifier import SeverityClassifierBaseline
from app.mlops.monitoring import evaluate_model_health

router = APIRouter(prefix="/api/v1")

# Lazy-loaded model singletons
_MODELS: dict[str, Any] = {}

def get_model(model_cls: Any, filename: str) -> Any:
    """Lazy load a specialized model artifact."""
    if filename not in _MODELS:
        p = Path(f"data/models/{filename}")
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Model artifact {filename} not found.")
        _MODELS[filename] = model_cls.load(p)
    return _MODELS[filename]

def to_df(records: list[Any]) -> pd.DataFrame:
    if not records:
        raise HTTPException(status_code=400, detail="Empty records list")
    df = pd.DataFrame([r.model_dump() for r in records])
    return df

@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok", "service": "omniroute-aiops"}

@router.get("/telemetry/latest")
def get_latest_telemetry(limit: int = 500):
    """Retrieve the most recent telemetry records from the data store."""
    p = Path("data/synthetic/telemetry.parquet")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Telemetry data not found.")
    
    df = pd.read_parquet(p)
    # Sort by timestamp descending and take the limit
    df = df.sort_values("timestamp", ascending=False).head(limit)
    return df.to_dict(orient="records")

@router.get("/mlops/status")
def get_mlops_status():
    """Retrieve lightweight artifact inventory."""
    status: dict[str, Any] = {"artifacts": {}}
    model_dir = Path("data/models")
    if model_dir.exists():
        for p in model_dir.glob("*.joblib"):
            status["artifacts"][p.name] = {"size_bytes": p.stat().st_size}
    return status

@router.get("/mlops/health")
def get_ml_health():
    """Run holistic health, data quality, and drift monitoring."""
    p = Path("data/synthetic/telemetry.parquet")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Reference telemetry not found.")
    
    df = pd.read_parquet(p)
    # Take a window of data for health check
    current_df = df.tail(1000)
    reference_df = df.head(1000) # Use the beginning as reference for drift check
    
    detector = get_model(IsolationForestDetector, "isolation_forest_latest.joblib")
    
    health_report = evaluate_model_health(
        model=detector,
        current_df=current_df,
        reference_df=reference_df,
        artifact_path=Path("data/models/isolation_forest_latest.joblib")
    )
    return health_report.to_dict()

@router.post("/telemetry/summary")
def get_telemetry_summary(req: TelemetryBatchRequest):
    df = to_df(req.records)
    
    summary = {
        "record_count": len(df),
        "services": df["service"].unique().tolist(),
        "time_range": {
            "start": df["timestamp"].min(),
            "end": df["timestamp"].max(),
        }
    }
    return summary

@router.post("/anomalies/detect")
def detect_anomalies(req: TelemetryBatchRequest):
    df = to_df(req.records)
    clean_df, _ = preprocess_telemetry(df)
    detector = get_model(IsolationForestDetector, "isolation_forest_latest.joblib")
    
    anomalies_df = detector.predict(clean_df)
    return {"anomalies": anomalies_df.to_dict(orient="records")}

@router.post("/incidents/predict")
def predict_incidents(req: PredictIncidentRequest):
    df = to_df(req.records)
    clean_df, _ = preprocess_telemetry(df)
    predictor = get_model(IncidentPredictorBaseline, "incident_predictor_latest.joblib")
    
    preds_df = predictor.predict_dataframe(clean_df)
    return {"predictions": preds_df.to_dict(orient="records")}

@router.post("/incidents/severity")
def predict_severity(req: PredictIncidentRequest):
    df = to_df(req.records)
    clean_df, _ = preprocess_telemetry(df)
    classifier = get_model(SeverityClassifierBaseline, "severity_classifier_latest.joblib")
    
    preds_df = classifier.predict_dataframe(clean_df)
    return {"severities": preds_df.to_dict(orient="records")}

@router.post("/rca/rank")
def get_rca_ranking(req: RcaRequest):
    df = to_df(req.records)
    clean_df, _ = preprocess_telemetry(df)
    
    detector = get_model(IsolationForestDetector, "isolation_forest_latest.joblib")
    anomalies_df = detector.predict(clean_df)
    
    incident_dict = {
        "incident_id": "api_incident",
        "start_time": req.incident_start_time,
        "detection_time": req.incident_end_time,
        "end_time": req.incident_end_time,
    }
    
    rca_res = rank_root_causes(clean_df, anomalies_df, incident_dict)
    return rca_res.to_dict()

@router.post("/rca/explain")
def get_rca_explanation(req: RcaRequest):
    df = to_df(req.records)
    clean_df, _ = preprocess_telemetry(df)
    
    detector = get_model(IsolationForestDetector, "isolation_forest_latest.joblib")
    anomalies_df = detector.predict(clean_df)
    
    incident_dict = {
        "incident_id": "api_incident",
        "start_time": req.incident_start_time,
        "detection_time": req.incident_end_time,
        "end_time": req.incident_end_time,
    }
    
    expl = generate_rca_explanation(clean_df, anomalies_df, incident_dict)
    return expl.to_dict()
