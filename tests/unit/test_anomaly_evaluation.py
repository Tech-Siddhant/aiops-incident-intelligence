"""Unit tests for anomaly detection evaluation suite."""
import numpy as np
import pandas as pd
import pytest

from app.data.preprocess import preprocess_telemetry
from app.data.synthetic import SyntheticConfig, generate_synthetic_telemetry
from evaluation.anomaly_evaluation import (
    AnomalyMetrics,
    calculate_metrics,
    run_anomaly_benchmark,
)


def test_calculate_metrics_perfect_score():
    preds = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="10s", tz="UTC"),
        "service": ["database"] * 4,
        "is_anomaly": [False, True, True, False],
    })
    labels = pd.DataFrame({
        "is_incident": [False, True, True, False],
    })
    metrics = calculate_metrics(preds, labels)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.true_positives == 2
    assert metrics.true_negatives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


def test_calculate_metrics_detection_delay():
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=6, freq="10s", tz="UTC")
    preds = pd.DataFrame({
        "timestamp": timestamps,
        "service": ["database"] * 6,
        "is_anomaly": [False, False, False, True, True, True],
    })
    labels = pd.DataFrame({
        "is_incident": [False, False, True, True, True, True],
    })
    incidents = [{
        "incident_id": "INC-1",
        "start_time": "2026-01-01T00:00:20Z",
        "end_time": "2026-01-01T00:00:50Z",
        "root_cause_service": "database",
        "affected_services": ["database"],
    }]
    metrics = calculate_metrics(preds, labels, incidents=incidents)
    # Incident started at 00:00:20, first anomaly at 00:00:30 -> delay = 10.0s
    assert metrics.detection_delay_seconds == 10.0
    assert metrics.true_positives == 3
    assert metrics.false_negatives == 1


def test_calculate_metrics_length_mismatch():
    preds = pd.DataFrame({"is_anomaly": [True, False]})
    labels = pd.DataFrame({"is_incident": [True]})
    with pytest.raises(ValueError):
        calculate_metrics(preds, labels)


def test_run_anomaly_benchmark_on_synthetic_failure():
    cfg = SyntheticConfig(
        duration_seconds=600,
        sampling_interval_seconds=10,
        failure_scenario="database_connection_saturation",
        failure_start_seconds=200,
        failure_duration_seconds=200,
        seed=42,
    )
    raw_df, incs = generate_synthetic_telemetry(cfg)
    clean_df, _ = preprocess_telemetry(raw_df)

    results = run_anomaly_benchmark(clean_df, incs)
    assert "statistical_baseline" in results
    assert "isolation_forest" in results

    base_m = results["statistical_baseline"]
    if_m = results["isolation_forest"]

    assert isinstance(base_m, AnomalyMetrics)
    assert isinstance(if_m, AnomalyMetrics)
    assert base_m.total_samples == len(clean_df)
    assert if_m.total_samples == len(clean_df)
    assert base_m.detection_delay_seconds is not None
    assert if_m.detection_delay_seconds is not None
