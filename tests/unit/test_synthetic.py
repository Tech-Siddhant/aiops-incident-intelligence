"""Unit tests for synthetic telemetry generator."""
import json
from pathlib import Path
import pandas as pd
import pytest
from app.data.synthetic import SERVICES, SyntheticConfig, generate_synthetic_telemetry, save_synthetic_data

REQUIRED_COLUMNS = [
    "timestamp", "service", "cpu_usage_pct", "memory_usage_pct",
    "disk_usage_pct", "network_in_mbps", "network_out_mbps",
    "request_rate_rps", "latency_ms", "error_rate",
    "active_connections", "connection_utilization",
]


def test_config_validation():
    with pytest.raises(ValueError, match="duration_seconds must be > 0"):
        SyntheticConfig(duration_seconds=0).validate()
    with pytest.raises(ValueError, match="sampling_interval_seconds must be > 0"):
        SyntheticConfig(sampling_interval_seconds=-1).validate()
    with pytest.raises(ValueError, match="cannot exceed duration_seconds"):
        SyntheticConfig(duration_seconds=10, sampling_interval_seconds=20).validate()
    with pytest.raises(ValueError, match="Unknown failure_scenario"):
        SyntheticConfig(failure_scenario="invalid_scenario").validate()
    with pytest.raises(ValueError, match="failure_start_seconds must be < duration_seconds"):
        SyntheticConfig(duration_seconds=100, failure_start_seconds=150).validate()
    with pytest.raises(ValueError, match="failure_duration_seconds must be > 0"):
        SyntheticConfig(failure_duration_seconds=0).validate()


def test_all_services_and_columns():
    config = SyntheticConfig(duration_seconds=120, sampling_interval_seconds=10, seed=1, failure_scenario=None)
    df, incidents = generate_synthetic_telemetry(config)
    assert set(df.columns) == set(REQUIRED_COLUMNS)
    assert set(df["service"].unique()) == set(SERVICES)
    assert len(df) == (120 // 10) * len(SERVICES)
    assert df.isnull().sum().sum() == 0
    assert incidents == []


def test_value_ranges():
    config = SyntheticConfig(duration_seconds=600, sampling_interval_seconds=10,
                              failure_start_seconds=100, failure_duration_seconds=300, seed=42)
    df, _ = generate_synthetic_telemetry(config)
    assert (df["cpu_usage_pct"].between(0.0, 100.0)).all()
    assert (df["memory_usage_pct"].between(0.0, 100.0)).all()
    assert (df["disk_usage_pct"].between(0.0, 100.0)).all()
    assert (df["network_in_mbps"] >= 0.0).all()
    assert (df["network_out_mbps"] >= 0.0).all()
    assert (df["request_rate_rps"] >= 0.0).all()
    assert (df["latency_ms"] >= 0.0).all()
    assert (df["error_rate"].between(0.0, 1.0)).all()
    assert (df["active_connections"] >= 0).all()
    assert (df["connection_utilization"].between(0.0, 1.0)).all()


def test_failure_propagation_behavior():
    config = SyntheticConfig(duration_seconds=600, sampling_interval_seconds=10,
                              failure_start_seconds=200, failure_duration_seconds=300, seed=123)
    df, _ = generate_synthetic_telemetry(config)
    db_df    = df[df["service"] == "database"].reset_index(drop=True)
    ord_df   = df[df["service"] == "orders_service"].reset_index(drop=True)
    gw_df    = df[df["service"] == "api_gateway"].reset_index(drop=True)
    auth_df  = df[df["service"] == "auth_service"].reset_index(drop=True)

    assert db_df.iloc[25:45]["connection_utilization"].mean() > db_df.iloc[:10]["connection_utilization"].mean() + 0.3
    assert db_df.iloc[25:45]["latency_ms"].mean() > db_df.iloc[:10]["latency_ms"].mean() * 5
    assert ord_df.iloc[27:47]["latency_ms"].mean() > ord_df.iloc[:10]["latency_ms"].mean() * 5
    assert gw_df.iloc[29:49]["error_rate"].mean() > gw_df.iloc[:10]["error_rate"].mean() * 5
    assert abs(auth_df.iloc[25:45]["latency_ms"].mean() - auth_df.iloc[:10]["latency_ms"].mean()) < 5.0


def test_reproducibility():
    config1 = SyntheticConfig(seed=999, duration_seconds=600, failure_start_seconds=120)
    config2 = SyntheticConfig(seed=999, duration_seconds=600, failure_start_seconds=120)
    df1, inc1 = generate_synthetic_telemetry(config1)
    df2, inc2 = generate_synthetic_telemetry(config2)
    pd.testing.assert_frame_equal(df1, df2)
    assert inc1 == inc2


def test_save_synthetic_data(tmp_path: Path):
    config = SyntheticConfig(duration_seconds=120, sampling_interval_seconds=10, failure_scenario=None)
    df, incidents = generate_synthetic_telemetry(config)
    parquet_file, json_file = save_synthetic_data(df, incidents, output_dir=tmp_path)
    assert parquet_file.exists()
    assert json_file.exists()
    loaded_df = pd.read_parquet(parquet_file)
    assert len(loaded_df) == len(df)
    with open(json_file, encoding="utf-8") as f: loaded_incidents = json.load(f)
    assert loaded_incidents == incidents


def test_failure_scenario_ground_truth():
    config = SyntheticConfig(duration_seconds=600, sampling_interval_seconds=10,
                              failure_start_seconds=120, failure_duration_seconds=200, seed=42)
    _, incidents = generate_synthetic_telemetry(config)
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc["incident_id"] == "INC-001"
    assert inc["incident_type"] == "database_connection_saturation"
    assert inc["root_cause_service"] == "database"
    assert inc["start_time"] == "2026-01-01T00:02:00Z"
    assert inc["end_time"] == "2026-01-01T00:05:20Z"
    assert inc["start_time"] <= inc["detection_time"] <= inc["end_time"]
