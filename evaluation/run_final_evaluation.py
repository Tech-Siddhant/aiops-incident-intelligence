"""Final comprehensive evaluation and benchmark runner for AIOps Incident Intelligence.

Evaluates all pipeline layers using actual synthetic data and trained models:
1. Anomaly Detection (Statistical Baseline vs Isolation Forest)
2. Incident Prediction (Train / Val / Test splits, PR-AUC, ROC-AUC, Lead Time)
3. Severity Classification (Macro F1, Per-class metrics, Confusion Matrix)
4. Root Cause Analysis (Top-1, Top-3, MRR, Explanations)
5. System Performance (API latencies, Model inference times, Artifact sizes, Peak RAM)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app
from app.config import MODELS_DIR
from app.data.features import extract_features
from app.data.preprocess import get_incident_labels, preprocess_telemetry
from app.data.synthetic import SyntheticConfig, generate_synthetic_telemetry
from app.models.anomaly_baseline import BaselineConfig, detect_anomalies
from app.models.anomaly_isolation_forest import (
    IsolationForestConfig,
    IsolationForestDetector,
    detect_anomalies_iforest,
)
from app.models.incident_predictor import (
    IncidentPredictionConfig,
    IncidentPredictorBaseline,
    chronological_train_val_test_split,
    create_horizon_labels,
)
from app.models.rca_engine import DEFAULT_TOPOLOGY, RCAEngine, rank_root_causes
from app.models.rca_explainer import generate_rca_explanation
from app.models.severity_classifier import (
    SEVERITY_CLASSES,
    SeverityClassificationConfig,
    SeverityClassifierBaseline,
    calculate_severity_metrics,
    extract_severity_labels,
)
from evaluation.anomaly_evaluation import calculate_metrics, run_anomaly_benchmark
from evaluation.incident_evaluation import (
    calculate_prediction_lead_time,
    run_incident_evaluation,
)
from evaluation.rca_evaluation import evaluate_rca_rankings


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_benchmark_b_dataset() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Generate 6-hour continuous multi-episode stream (Benchmark B)."""
    start = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    episodes = [
        SyntheticConfig(
            start_time=_iso_utc(start + timedelta(hours=0)),
            duration_seconds=3600,
            failure_scenario="database_connection_saturation",
            failure_start_seconds=1200,
            failure_duration_seconds=600,
            seed=42,
        ),
        SyntheticConfig(
            start_time=_iso_utc(start + timedelta(hours=1)),
            duration_seconds=3600,
            failure_scenario=None,
            seed=43,
        ),
        SyntheticConfig(
            start_time=_iso_utc(start + timedelta(hours=2)),
            duration_seconds=3600,
            failure_scenario="database_connection_saturation",
            failure_start_seconds=1500,
            failure_duration_seconds=900,
            seed=44,
        ),
        SyntheticConfig(
            start_time=_iso_utc(start + timedelta(hours=3)),
            duration_seconds=3600,
            failure_scenario=None,
            seed=45,
        ),
        SyntheticConfig(
            start_time=_iso_utc(start + timedelta(hours=4)),
            duration_seconds=3600,
            failure_scenario="database_connection_saturation",
            failure_start_seconds=1000,
            failure_duration_seconds=800,
            seed=46,
        ),
        SyntheticConfig(
            start_time=_iso_utc(start + timedelta(hours=5)),
            duration_seconds=3600,
            failure_scenario="database_connection_saturation",
            failure_start_seconds=1800,
            failure_duration_seconds=600,
            seed=47,
        ),
    ]

    dfs, incs = [], []
    for i, cfg in enumerate(episodes):
        df_ep, inc_ep = generate_synthetic_telemetry(cfg)
        for inc in inc_ep:
            inc["incident_id"] = f"INC-00{i+1}"
        dfs.append(df_ep)
        incs.extend(inc_ep)

    full_df = pd.concat(dfs, ignore_index=True).sort_values(["timestamp", "service"]).reset_index(drop=True)
    return full_df, incs


def evaluate_anomaly_layer() -> dict[str, Any]:
    """1. Evaluate Anomaly Detection (Baseline vs Isolation Forest)."""
    cfg = SyntheticConfig(
        duration_seconds=900,
        sampling_interval_seconds=10,
        failure_scenario="database_connection_saturation",
        failure_start_seconds=300,
        failure_duration_seconds=300,
        seed=42,
    )
    raw_df, incs = generate_synthetic_telemetry(cfg)
    clean_df, _ = preprocess_telemetry(raw_df)

    res = run_anomaly_benchmark(clean_df, incs)
    return {
        "statistical_baseline": res["statistical_baseline"].to_dict(),
        "isolation_forest": res["isolation_forest"].to_dict(),
    }


def evaluate_prediction_and_severity_layers() -> dict[str, Any]:
    """2 & 3. Evaluate Incident Prediction & Severity Classification on Benchmark B."""
    raw_df, incs = generate_benchmark_b_dataset()
    clean_df, _ = preprocess_telemetry(raw_df)
    feat_df = extract_features(clean_df)

    # Incident prediction evaluation
    res = run_incident_evaluation(
        feat_df,
        incs,
        prediction_config=IncidentPredictionConfig(random_state=42),
        severity_config=SeverityClassificationConfig(random_state=42),
        val_ratio=0.20,
        test_ratio=0.20,
    )
    return res.to_dict()


def evaluate_rca_layer() -> dict[str, Any]:
    """4. Evaluate Root Cause Analysis on synthetic cascading incident."""
    cfg = SyntheticConfig(
        duration_seconds=1800,
        sampling_interval_seconds=10,
        failure_scenario="database_connection_saturation",
        failure_start_seconds=900,
        failure_duration_seconds=600,
        seed=42,
    )
    raw_df, incs = generate_synthetic_telemetry(cfg)
    clean_df, _ = preprocess_telemetry(raw_df)

    # Isolation Forest trained on pre-incident nominal stream
    train_df = clean_df[clean_df["timestamp"] < "2026-01-01T00:15:00Z"]
    anom_df = detect_anomalies_iforest(
        clean_df,
        config=IsolationForestConfig(contamination=0.05, random_state=42),
        train_df=train_df,
    )

    results = []
    explanations = []
    engine = RCAEngine()

    for inc in incs:
        rca_res = engine.rank(clean_df, anomalies_df=anom_df, incident=inc)
        results.append((rca_res, inc))
        expl = generate_rca_explanation(clean_df, anomalies_df=anom_df, incident=inc)
        explanations.append(expl.to_dict())

    rca_metrics = evaluate_rca_rankings(results)
    top_candidates = results[0][0].ranked_candidates if results else []

    return {
        "metrics": rca_metrics.to_dict(),
        "top_ranked_candidates": [
            {
                "service": c.service,
                "score": round(c.score, 4),
                "rank": c.rank,
                "score_breakdown": {k: round(v, 4) for k, v in c.score_breakdown.items()},
                "explanation": c.explanation,
            }
            for c in top_candidates
        ],
        "top_explanation": explanations[0] if explanations else None,
    }


def evaluate_system_and_performance() -> dict[str, Any]:
    """5. Measure real system benchmarks: API latencies, inference times, artifact sizes, RAM."""
    # Ensure models are trained and saved
    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = SyntheticConfig(duration_seconds=1800, sampling_interval_seconds=10, seed=42)
    raw_df, incs = generate_synthetic_telemetry(cfg)
    clean_df, _ = preprocess_telemetry(raw_df)
    feat_df = extract_features(clean_df)
    incident = incs[0]

    # Save models
    if_det = IsolationForestDetector(config=IsolationForestConfig(n_estimators=50, contamination=0.05, random_state=42))
    if_det.fit(clean_df)
    if_path = models_dir / "isolation_forest_latest.joblib"
    if_det.save(if_path)

    pred_model = IncidentPredictorBaseline(config=IncidentPredictionConfig(random_state=42))
    y_pred = np.zeros(len(clean_df), dtype=int)
    for inc in incs:
        mask = (clean_df["timestamp"] >= inc["start_time"]) & (clean_df["timestamp"] <= inc["end_time"])
        y_pred[mask] = 1
    pred_model.fit(clean_df, y_pred)
    pred_path = models_dir / "incident_predictor_latest.joblib"
    pred_model.save(pred_path)

    sev_model = SeverityClassifierBaseline(config=SeverityClassificationConfig(random_state=42))
    y_sev = np.array(["low"] * len(clean_df), dtype=object)
    for inc in incs:
        mask = (clean_df["timestamp"] >= inc["start_time"]) & (clean_df["timestamp"] <= inc["end_time"])
        y_sev[mask] = inc["severity"]
    sev_model.fit(clean_df, y_sev)
    sev_path = models_dir / "severity_classifier_latest.joblib"
    sev_model.save(sev_path)

    # 1. Model artifact sizes
    artifact_sizes = {
        "isolation_forest_kb": round(if_path.stat().st_size / 1024, 2),
        "incident_predictor_kb": round(pred_path.stat().st_size / 1024, 2),
        "severity_classifier_kb": round(sev_path.stat().st_size / 1024, 2),
        "total_models_size_kb": round(
            (if_path.stat().st_size + pred_path.stat().st_size + sev_path.stat().st_size) / 1024, 2
        ),
    }

    # 2. Inference latency and memory overhead
    tracemalloc.start()
    t0 = time.perf_counter()
    if_det.predict(clean_df)
    if_latency_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    pred_model.predict(clean_df)
    pred_latency_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    sev_model.predict(clean_df)
    sev_latency_ms = (time.perf_counter() - t0) * 1000

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_samples = len(clean_df)
    inference_metrics = {
        "batch_size_records": n_samples,
        "isolation_forest_batch_ms": round(if_latency_ms, 2),
        "isolation_forest_per_sample_us": round((if_latency_ms / n_samples) * 1000, 2),
        "incident_predictor_batch_ms": round(pred_latency_ms, 2),
        "incident_predictor_per_sample_us": round((pred_latency_ms / n_samples) * 1000, 2),
        "severity_classifier_batch_ms": round(sev_latency_ms, 2),
        "severity_classifier_per_sample_us": round((sev_latency_ms / n_samples) * 1000, 2),
        "peak_inference_memory_kb": round(peak_mem / 1024, 2),
    }

    # 3. API Route Latencies
    client = TestClient(app)
    records = raw_df.to_dict(orient="records")

    api_latencies: dict[str, float] = {}

    # GET /api/v1/health
    t0 = time.perf_counter()
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    api_latencies["get_health_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # GET /api/v1/mlops/status
    t0 = time.perf_counter()
    res = client.get("/api/v1/mlops/status")
    assert res.status_code == 200
    api_latencies["get_mlops_status_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # POST /api/v1/telemetry/summary
    t0 = time.perf_counter()
    res = client.post("/api/v1/telemetry/summary", json={"records": records})
    assert res.status_code == 200
    api_latencies["post_telemetry_summary_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # POST /api/v1/anomalies/detect
    t0 = time.perf_counter()
    res = client.post("/api/v1/anomalies/detect", json={"records": records})
    assert res.status_code == 200
    api_latencies["post_anomalies_detect_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # POST /api/v1/incidents/predict
    t0 = time.perf_counter()
    res = client.post("/api/v1/incidents/predict", json={"records": records})
    assert res.status_code == 200
    api_latencies["post_incidents_predict_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # POST /api/v1/incidents/severity
    t0 = time.perf_counter()
    res = client.post("/api/v1/incidents/severity", json={"records": records})
    assert res.status_code == 200
    api_latencies["post_incidents_severity_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # POST /api/v1/rca/rank
    rca_payload = {
        "records": records,
        "incident_start_time": incident["start_time"],
        "incident_end_time": incident["end_time"],
    }
    t0 = time.perf_counter()
    res = client.post("/api/v1/rca/rank", json=rca_payload)
    assert res.status_code == 200
    api_latencies["post_rca_rank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # POST /api/v1/rca/explain
    t0 = time.perf_counter()
    res = client.post("/api/v1/rca/explain", json=rca_payload)
    assert res.status_code == 200
    api_latencies["post_rca_explain_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "artifact_sizes": artifact_sizes,
        "inference_metrics": inference_metrics,
        "api_latencies": api_latencies,
    }


def run_full_final_evaluation() -> dict[str, Any]:
    print("=" * 60)
    print("RUNNING FINAL BENCHMARK EVALUATION (PHASE 8.1)")
    print("=" * 60)

    print("\n[1/5] Evaluating Anomaly Detection Layer...")
    anomaly_results = evaluate_anomaly_layer()

    print("\n[2/5 & 3/5] Evaluating Incident Prediction & Severity Classification...")
    pred_sev_results = evaluate_prediction_and_severity_layers()

    print("\n[4/5] Evaluating Root Cause Analysis (RCA) Layer...")
    rca_results = evaluate_rca_layer()

    print("\n[5/5] Measuring System Performance & Resource Telemetry...")
    system_results = evaluate_system_and_performance()

    final_report = {
        "evaluation_timestamp": _iso_utc(datetime.now(timezone.utc)),
        "anomaly_detection": anomaly_results,
        "incident_prediction_and_severity": pred_sev_results,
        "root_cause_analysis": rca_results,
        "system_performance": system_results,
    }

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE - SUMMARY RESULTS:")
    print("=" * 60)
    print(json.dumps(final_report, indent=2))
    return final_report


if __name__ == "__main__":
    run_full_final_evaluation()
