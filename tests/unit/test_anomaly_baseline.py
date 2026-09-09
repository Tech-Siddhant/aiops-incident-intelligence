"""Unit tests for statistical baseline anomaly detector."""
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from app.data.preprocess import preprocess_telemetry
from app.data.synthetic import SyntheticConfig, generate_synthetic_telemetry
from app.models.anomaly_baseline import (
    BaselineConfig,
    StatisticalBaselineDetector,
    detect_anomalies,
)


@pytest.fixture
def clean_normal_data() -> pd.DataFrame:
    """Synthetic dataset with no failure scenario."""
    cfg = SyntheticConfig(
        duration_seconds=600,
        sampling_interval_seconds=10,
        failure_scenario=None,
        seed=42,
    )
    raw_df, _ = generate_synthetic_telemetry(cfg)
    clean_df, _ = preprocess_telemetry(raw_df)
    return clean_df


@pytest.fixture
def synthetic_incident_data() -> tuple[pd.DataFrame, list[dict]]:
    """Synthetic dataset with database connection saturation failure at t=300s."""
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
    return clean_df, incs


def test_detector_output_schema(clean_normal_data: pd.DataFrame):
    results = detect_anomalies(clean_normal_data)
    assert set(results.columns) == {"timestamp", "service", "anomaly_score", "is_anomaly", "anomalous_metrics"}
    assert len(results) == len(clean_normal_data)
    assert pd.api.types.is_datetime64_any_dtype(results["timestamp"])
    assert pd.api.types.is_bool_dtype(results["is_anomaly"])
    assert pd.api.types.is_float_dtype(results["anomaly_score"])


def test_no_future_leakage(clean_normal_data: pd.DataFrame):
    """Perturbing future rows must not change past anomaly scores."""
    res_orig = detect_anomalies(clean_normal_data)

    mutated = clean_normal_data.copy()
    split_time = mutated["timestamp"].iloc[len(mutated) // 2]
    future_mask = mutated["timestamp"] >= split_time
    mutated.loc[future_mask, "latency_ms"] = mutated.loc[future_mask, "latency_ms"] * 100.0

    res_mutated = detect_anomalies(mutated)

    past_orig = res_orig[res_orig["timestamp"] < split_time].reset_index(drop=True)
    past_mut = res_mutated[res_mutated["timestamp"] < split_time].reset_index(drop=True)

    pd.testing.assert_frame_equal(past_orig, past_mut)


def test_insufficient_history_handled():
    """Initial points (< min_periods) must default to score 0.0 and is_anomaly=False."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="10s", tz="UTC"),
        "service": ["database"] * 4,
        "latency_ms": [1000.0, 2000.0, 3000.0, 4000.0],
        "error_rate": [0.5, 0.5, 0.5, 0.5],
        "cpu_usage_pct": [90.0] * 4,
        "memory_usage_pct": [90.0] * 4,
        "disk_usage_pct": [50.0] * 4,
        "network_in_mbps": [10.0] * 4,
        "network_out_mbps": [10.0] * 4,
        "request_rate_rps": [100.0] * 4,
        "active_connections": [100] * 4,
        "connection_utilization": [0.9] * 4,
    })
    cfg = BaselineConfig(window_size=10, min_periods=5, z_threshold=2.0)
    res = detect_anomalies(df, config=cfg)

    # First 4 rows have < 5 periods of prior history
    assert (res["anomaly_score"] == 0.0).all()
    assert (~res["is_anomaly"]).all()


def test_deterministic_output(clean_normal_data: pd.DataFrame):
    res1 = detect_anomalies(clean_normal_data)
    res2 = detect_anomalies(clean_normal_data)
    pd.testing.assert_frame_equal(res1, res2)


def test_empty_dataframe():
    empty_df = pd.DataFrame(columns=["timestamp", "service", "latency_ms"])
    res = detect_anomalies(empty_df)
    assert res.empty
    assert "is_anomaly" in res.columns


def test_clean_normal_data_low_false_positives(clean_normal_data: pd.DataFrame):
    """Under normal operating conditions with z_threshold=3.0, false alarm rate should be very low (<5%)."""
    res = detect_anomalies(clean_normal_data, config=BaselineConfig(z_threshold=3.0))
    fp_rate = res["is_anomaly"].mean()
    assert fp_rate < 0.05


def test_detection_on_synthetic_failure_scenario(synthetic_incident_data):
    """Verify that during database saturation failure, anomalies are detected in database and cascade downstream."""
    df, incs = synthetic_incident_data
    res = detect_anomalies(df, config=BaselineConfig(window_size=30, z_threshold=3.0))

    inc = incs[0]
    start_ts = pd.to_datetime(inc["start_time"], utc=True)
    end_ts = pd.to_datetime(inc["end_time"], utc=True)

    incident_window = res[(res["timestamp"] >= start_ts) & (res["timestamp"] <= end_ts)]

    # 1. Database (root cause) must have strong anomaly signals
    db_anomalies = incident_window[incident_window["service"] == "database"]
    assert db_anomalies["is_anomaly"].sum() > 0
    assert db_anomalies["anomaly_score"].max() > 10.0

    # 2. Orders service and api_gateway must also show downstream cascading anomalies
    orders_anomalies = incident_window[incident_window["service"] == "orders_service"]
    gateway_anomalies = incident_window[incident_window["service"] == "api_gateway"]
    assert orders_anomalies["is_anomaly"].sum() > 0
    assert gateway_anomalies["is_anomaly"].sum() > 0

    # 3. Auth service should have very low/no anomalies
    auth_anomalies = incident_window[incident_window["service"] == "auth_service"]
    assert auth_anomalies["is_anomaly"].sum() == 0
