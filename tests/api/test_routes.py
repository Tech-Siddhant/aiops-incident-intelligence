"""Unit tests for the FastAPI intelligence routes."""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.data.synthetic import SyntheticConfig, generate_synthetic_telemetry
from app.models.anomaly_isolation_forest import IsolationForestConfig, IsolationForestDetector
from app.models.incident_predictor import IncidentPredictionConfig, IncidentPredictorBaseline
from app.models.severity_classifier import SeverityClassificationConfig, SeverityClassifierBaseline


@pytest.fixture(scope="module")
def client():
    return TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_models(tmp_path_factory):
    # Ensure baseline models are trained and present in data/models
    from pathlib import Path
    Path("data/models").mkdir(parents=True, exist_ok=True)
    
    cfg = SyntheticConfig(duration_seconds=1800, sampling_interval_seconds=10, seed=42)
    df, _ = generate_synthetic_telemetry(cfg)
    
    # 1. Isolation Forest
    if_detector = IsolationForestDetector(config=IsolationForestConfig(n_estimators=10))
    if_detector.fit(df)
    if_detector.save("data/models/isolation_forest_latest.joblib")
    
    # 2. Incident Predictor
    pred = IncidentPredictorBaseline(config=IncidentPredictionConfig())
    # Synthetic ground truth
    import numpy as np
    y_pred = np.zeros(len(df), dtype=int)
    pred.fit(df, y_pred)
    pred.save("data/models/incident_predictor_latest.joblib")
    
    # 3. Severity Classifier
    sev = SeverityClassifierBaseline(config=SeverityClassificationConfig())
    y_sev = ["low"] * len(df)
    sev.fit(df, y_sev)
    sev.save("data/models/severity_classifier_latest.joblib")

@pytest.fixture
def sample_records():
    cfg = SyntheticConfig(duration_seconds=1800, sampling_interval_seconds=10, seed=42)
    df, _ = generate_synthetic_telemetry(cfg)
    return df.to_dict(orient="records")

def test_health_check(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "omniroute-aiops"

def test_mlops_status(client):
    res = client.get("/api/v1/mlops/status")
    assert res.status_code == 200
    data = res.json()
    assert "artifacts" in data
    assert "isolation_forest_latest.joblib" in data["artifacts"]

def test_telemetry_summary(client, sample_records):
    payload = {"records": sample_records}
    res = client.post("/api/v1/telemetry/summary", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["record_count"] == len(sample_records)
    assert "services" in data
def test_detect_anomalies(client, sample_records):
    payload = {"records": sample_records}
    res = client.post("/api/v1/anomalies/detect", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "anomalies" in data
    assert len(data["anomalies"]) == len(sample_records)
    assert "is_anomaly" in data["anomalies"][0]

def test_predict_incidents(client, sample_records):
    payload = {"records": sample_records}
    res = client.post("/api/v1/incidents/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predictions" in data
    assert len(data["predictions"]) == len(sample_records)

def test_predict_severity(client, sample_records):
    payload = {"records": sample_records}
    res = client.post("/api/v1/incidents/severity", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "severities" in data
    assert len(data["severities"]) == len(sample_records)

def test_rca_endpoints(client, sample_records):
    payload = {
        "records": sample_records,
        "incident_start_time": sample_records[0]["timestamp"],
        "incident_end_time": sample_records[-1]["timestamp"],
    }
    
    # Test RCA Ranking
    res = client.post("/api/v1/rca/rank", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "top_1" in data
    assert "ranked_candidates" in data

    # Test RCA Explanation
    res = client.post("/api/v1/rca/explain", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "probable_root_cause" in data
    assert "summary" in data

def test_invalid_payload_error(client):
    # Empty records
    res = client.post("/api/v1/anomalies/detect", json={"records": []})
    assert res.status_code == 400
    
    # Missing fields
    res = client.post("/api/v1/anomalies/detect", json={"foo": "bar"})
    assert res.status_code == 422

def test_missing_model_error(client, sample_records, monkeypatch):
    from app.api import routes
    routes._MODELS.clear()
    monkeypatch.setattr(routes, "_MODELS", {})
    
    # Point to nonexistent artifact
    with pytest.raises(Exception):
        routes.get_model(IsolationForestDetector, "nonexistent_model.joblib")

def test_telemetry_latest(client):
    res = client.get("/api/v1/telemetry/latest")
    assert res.status_code in [200, 404]

def test_mlops_health(client):
    res = client.get("/api/v1/mlops/health")
    assert res.status_code in [200, 404]

