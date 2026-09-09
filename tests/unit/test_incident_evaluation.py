"""Unit tests for Phase 4 unified incident prediction and severity evaluation."""
import numpy as np
import pandas as pd
import pytest

from app.data.synthetic import SyntheticConfig, generate_synthetic_telemetry
from evaluation.incident_evaluation import (
    IncidentEvaluationResult,
    calculate_prediction_lead_time,
    run_incident_evaluation,
)


def test_calculate_prediction_lead_time():
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=6, freq="10s", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "service": ["database"] * 6,
        "is_predicted_incident": [False, True, True, False, False, False],
    })
    incidents = [{
        "incident_id": "INC-1",
        "start_time": "2026-01-01T00:00:40Z",
        "end_time": "2026-01-01T00:00:50Z",
        "root_cause_service": "database",
        "affected_services": ["database"],
    }]

    # Earliest true positive prediction before 00:00:40 is at 00:00:10 (Row 1).
    # Lead time = 40s - 10s = 30s.
    lead_time = calculate_prediction_lead_time(df, incidents, horizon_seconds=60)
    assert lead_time == 30.0


def test_calculate_prediction_lead_time_no_alerts():
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=5, freq="10s", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "service": ["database"] * 5,
        "is_predicted_incident": [False] * 5,
    })
    incidents = [{
        "start_time": "2026-01-01T00:00:30Z",
        "end_time": "2026-01-01T00:00:40Z",
        "root_cause_service": "database",
    }]
    lead_time = calculate_prediction_lead_time(df, incidents, horizon_seconds=60)
    assert lead_time is None


def test_run_incident_evaluation_synthetic():
    cfg = SyntheticConfig(
        duration_seconds=3600,
        sampling_interval_seconds=10,
        failure_scenario="database_connection_saturation",
        failure_start_seconds=1200,
        failure_duration_seconds=600,
        seed=42,
    )
    raw_df, incs = generate_synthetic_telemetry(cfg)

    result = run_incident_evaluation(raw_df, incs, val_ratio=0.15, test_ratio=0.15)
    assert isinstance(result, IncidentEvaluationResult)
    assert 0.0 <= result.prediction_train.f1_score <= 1.0
    assert 0.0 <= result.prediction_val.f1_score <= 1.0
    assert 0.0 <= result.prediction_test.f1_score <= 1.0
    assert result.prediction_test.total_samples > 0
    assert result.prediction_false_positives >= 0
    assert result.prediction_false_negatives >= 0
    assert result.severity_metrics.total_samples > 0

    res_dict = result.to_dict()
    assert "prediction_test" in res_dict
    assert "severity_metrics" in res_dict
    assert "top_predictive_features" in res_dict
