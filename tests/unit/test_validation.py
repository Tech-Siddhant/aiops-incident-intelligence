"""Unit tests for the synthetic data validation layer."""
import pandas as pd
import pytest
from app.data.validation import validate_telemetry, ValidationResult
from app.data.synthetic import REQUIRED_COLUMNS

def get_valid_df() -> pd.DataFrame:
    data = []
    for t in ["2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z"]:
        for s in ["api_gateway", "auth_service", "orders_service", "database"]:
            data.append({
                "timestamp": t,
                "service": s,
                "cpu_usage_pct": 20.0,
                "memory_usage_pct": 40.0,
                "disk_usage_pct": 30.0,
                "network_in_mbps": 10.0,
                "network_out_mbps": 15.0,
                "request_rate_rps": 100.0,
                "latency_ms": 20.0,
                "error_rate": 0.05,
                "active_connections": 50,
                "connection_utilization": 0.3,
            })
    return pd.DataFrame(data)

def get_valid_incident():
    return {
        "incident_id": "INC-001",
        "incident_type": "database_connection_saturation",
        "root_cause_service": "database",
        "start_time": "2026-01-01T00:00:00Z",
        "detection_time": "2026-01-01T00:00:05Z",
        "end_time": "2026-01-01T00:00:15Z",
        "severity": "critical",
        "description": "test"
    }

def test_valid_dataset_passes():
    df = get_valid_df()
    res = validate_telemetry(df, [get_valid_incident()])
    assert res.passed
    assert not res.errors

def test_missing_column_fails():
    df = get_valid_df()
    df.drop(columns=["cpu_usage_pct"], inplace=True)
    res = validate_telemetry(df)
    assert not res.passed
    assert any("Missing required columns" in e for e in res.errors)

def test_unexpected_service_fails():
    df = get_valid_df()
    df.loc[0, "service"] = "unknown_service"
    res = validate_telemetry(df)
    assert not res.passed
    assert any("Unknown services present" in e for e in res.errors)

def test_invalid_metric_range_fails():
    df = get_valid_df()
    df.loc[0, "cpu_usage_pct"] = 150.0
    res = validate_telemetry(df)
    assert not res.passed
    assert any("cpu_usage_pct out of bounds" in e for e in res.errors)
    
    df = get_valid_df()
    df.loc[0, "latency_ms"] = -10.0
    res = validate_telemetry(df)
    assert not res.passed
    assert any("latency_ms contains negative values" in e for e in res.errors)

def test_null_values_fails():
    df = get_valid_df()
    df.loc[0, "memory_usage_pct"] = None
    res = validate_telemetry(df)
    assert not res.passed
    assert any("Null values present" in e for e in res.errors)

def test_duplicate_rows_fails():
    df = pd.concat([get_valid_df(), get_valid_df().iloc[[0]]], ignore_index=True)
    res = validate_telemetry(df)
    assert not res.passed
    assert any("Duplicate rows found" in e for e in res.errors)

def test_invalid_timestamps_fails():
    df = get_valid_df()
    df.loc[6, "timestamp"] = "2025-01-01T00:00:00Z"
    res = validate_telemetry(df)
    assert not res.passed
    assert any("not monotonically strictly increasing" in e for e in res.errors)

def test_invalid_incident_timing_fails():
    df = get_valid_df()
    inc = get_valid_incident()
    inc["detection_time"] = "2025-01-01T00:00:00Z" # Before start_time
    res = validate_telemetry(df, [inc])
    assert not res.passed
    assert any("timestamps invalid ordering" in e for e in res.errors)

def test_unknown_root_cause_service_fails():
    df = get_valid_df()
    inc = get_valid_incident()
    inc["root_cause_service"] = "cache"
    res = validate_telemetry(df, [inc])
    assert not res.passed
    assert any("unrecognized root_cause_service" in e for e in res.errors)

def test_warnings_distinguished():
    df = get_valid_df()
    df.loc[0, "latency_ms"] = 6000.0  # Over 5000ms triggers warning
    res = validate_telemetry(df)
    assert res.passed
    assert len(res.warnings) == 1
    assert "Extremely high latency" in res.warnings[0]
