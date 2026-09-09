"""Validation layer for synthetic telemetry data."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd
from app.data.synthetic import SERVICES, VALID_SCENARIOS, REQUIRED_COLUMNS

@dataclass
class ValidationResult:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)

def validate_telemetry(df: pd.DataFrame, incidents: list[dict[str, Any]] | None = None) -> ValidationResult:
    """Validate telemetry DataFrame and optional incident ground truth."""
    res = ValidationResult()

    # 1. Required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        res.error(f"Missing required columns: {missing}")
        return res  # Stop if schema is completely broken

    # 2. Expected data types
    if not (pd.api.types.is_string_dtype(df["service"]) or pd.api.types.is_object_dtype(df["service"])):
        res.error("Column 'service' should be string/object")
    
    # 3. Missing/null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        res.error(f"Null values present:\n{null_counts[null_counts > 0]}")

    # 4. Duplicate rows
    if df.duplicated(subset=["timestamp", "service"]).any():
        res.error("Duplicate rows found for same timestamp and service")

    # 5. Timestamp validity and 6. ordering
    try:
        ts_df = pd.to_datetime(df["timestamp"], format="ISO8601")
        for srv in df["service"].unique():
            srv_ts = ts_df[df["service"] == srv].reset_index(drop=True)
            if not srv_ts.is_monotonic_increasing:
                res.error(f"Timestamps for {srv} are not monotonically strictly increasing")
    except Exception as e:
        res.error(f"Timestamp parsing failed: {e}")

    # 7. Expected service names
    unknown = set(df["service"]) - set(SERVICES)
    if unknown:
        res.error(f"Unknown services present: {unknown}")

    # 8-11. Numeric metric ranges and bounds
    for col in ["cpu_usage_pct", "memory_usage_pct", "disk_usage_pct", "connection_utilization"]:
        if not df[col].between(0.0, 100.0).all() and col != "connection_utilization":
            if df[col].min() < 0.0 or df[col].max() > 100.0:
                 res.error(f"{col} out of bounds [0.0, 100.0]")
    if not df["connection_utilization"].between(0.0, 1.0).all():
        res.error("connection_utilization out of bounds [0.0, 1.0]")
        
    for col in ["network_in_mbps", "network_out_mbps", "request_rate_rps", "latency_ms", "active_connections"]:
        if (df[col] < 0).any():
            res.error(f"{col} contains negative values")

    if not df["error_rate"].between(0.0, 1.0).all():
        res.error("error_rate out of bounds [0.0, 1.0]")

    # 12. Ground-truth incident consistency
    if incidents:
        for i, inc in enumerate(incidents):
            req_keys = {"incident_id", "incident_type", "root_cause_service", "start_time", "detection_time", "end_time"}
            if missing_keys := req_keys - set(inc.keys()):
                res.error(f"Incident {i} missing keys: {missing_keys}")
                continue
            
            if inc["incident_type"] not in VALID_SCENARIOS:
                res.error(f"Incident {i} has unrecognized type {inc['incident_type']}")
            if inc["root_cause_service"] not in SERVICES:
                res.error(f"Incident {i} has unrecognized root_cause_service {inc['root_cause_service']}")
            if not (inc["start_time"] <= inc["detection_time"] <= inc["end_time"]):
                res.error(f"Incident {i} timestamps invalid ordering")
            
            tf_min = df["timestamp"].min()
            tf_max = df["timestamp"].max()
            if inc["start_time"] > tf_max or inc["end_time"] < tf_min:
                res.error(f"Incident {i} falls fully outside telemetry window")

    # Informational warning check example
    if df["latency_ms"].max() > 5000:
        res.warning("Extremely high latency encountered (>5000ms), might be unrealistic")

    return res
