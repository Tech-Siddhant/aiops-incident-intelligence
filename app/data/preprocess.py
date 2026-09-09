"""Telemetry preprocessing and normalization pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.data.synthetic import REQUIRED_COLUMNS, SERVICES

NUMERIC_COLUMNS = [
    "cpu_usage_pct",
    "memory_usage_pct",
    "disk_usage_pct",
    "network_in_mbps",
    "network_out_mbps",
    "request_rate_rps",
    "latency_ms",
    "error_rate",
    "active_connections",
    "connection_utilization",
]


@dataclass
class PreprocessingReport:
    """Summary of data preprocessing transformations and filtered records."""
    input_rows: int = 0
    output_rows: int = 0
    null_rows_dropped: int = 0
    duplicate_rows_dropped: int = 0
    invalid_rows_dropped: int = 0
    unknown_service_rows_dropped: int = 0

    @property
    def total_dropped(self) -> int:
        return self.input_rows - self.output_rows


def preprocess_telemetry(df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Normalize, clean, and validate a telemetry DataFrame deterministically.

    Args:
        df: Raw telemetry DataFrame.

    Returns:
        Tuple of (clean_df, report).

    Raises:
        ValueError: If required columns are missing from the input.
    """
    report = PreprocessingReport(input_rows=len(df))

    if df.empty:
        empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        report.output_rows = 0
        return empty_df, report

    # 1. Normalize column names
    clean = df.copy()
    clean.columns = [str(c).strip().lower() for c in clean.columns]

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in clean.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input: {missing_cols}")

    # 2. Normalize and filter service identifiers
    clean["service"] = clean["service"].astype(str).str.strip().str.lower()
    valid_services = set(SERVICES)
    known_svc_mask = clean["service"].isin(valid_services)
    report.unknown_service_rows_dropped = int((~known_svc_mask).sum())
    clean = clean[known_svc_mask]

    # 3. Normalize timestamps to UTC datetime
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")

    # 4. Handle null / NaN values across required columns
    null_mask = clean[REQUIRED_COLUMNS].isna().any(axis=1)
    report.null_rows_dropped = int(null_mask.sum())
    clean = clean[~null_mask]

    # 5. Remove duplicate (timestamp, service) records
    dup_mask = clean.duplicated(subset=["timestamp", "service"], keep="first")
    report.duplicate_rows_dropped = int(dup_mask.sum())
    clean = clean[~dup_mask]

    # 6. Validate value bounds and remove corrupted records
    valid_bounds = (
        clean["cpu_usage_pct"].between(0.0, 100.0)
        & clean["memory_usage_pct"].between(0.0, 100.0)
        & clean["disk_usage_pct"].between(0.0, 100.0)
        & clean["connection_utilization"].between(0.0, 1.0)
        & clean["error_rate"].between(0.0, 1.0)
        & (clean["network_in_mbps"] >= 0.0)
        & (clean["network_out_mbps"] >= 0.0)
        & (clean["request_rate_rps"] >= 0.0)
        & (clean["latency_ms"] >= 0.0)
        & (clean["active_connections"] >= 0)
    )
    report.invalid_rows_dropped = int((~valid_bounds).sum())
    clean = clean[valid_bounds]

    # 7. Normalize data types
    for col in NUMERIC_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean["active_connections"] = clean["active_connections"].astype(int)

    # 8. Sort deterministically by timestamp then service
    clean = clean[REQUIRED_COLUMNS].sort_values(["timestamp", "service"]).reset_index(drop=True)
    report.output_rows = len(clean)

    return clean, report


def get_incident_labels(
    df: pd.DataFrame,
    incidents: list[dict[str, Any]],
) -> pd.DataFrame:
    """Generate separate ground-truth incident labels aligned with a telemetry DataFrame.

    Prevents data leakage by isolating targets from telemetry feature columns.

    Args:
        df: Preprocessed telemetry DataFrame with parsed UTC datetime timestamps.
        incidents: List of ground-truth incident dictionaries.

    Returns:
        DataFrame containing target columns ('is_incident', 'is_root_cause', 'incident_id', 'incident_type').
    """
    ts = pd.to_datetime(df["timestamp"], utc=True)
    services = df["service"].astype(str)

    is_incident = np.zeros(len(df), dtype=bool)
    is_root_cause = np.zeros(len(df), dtype=bool)
    incident_id: list[str | None] = [None] * len(df)
    incident_type: list[str | None] = [None] * len(df)

    for inc in incidents:
        inc_start = pd.to_datetime(inc["start_time"], utc=True)
        inc_end = pd.to_datetime(inc["end_time"], utc=True)
        affected_services = set(inc.get("affected_services", [inc.get("root_cause_service")]))
        root_svc = inc.get("root_cause_service")

        time_window_mask = (ts >= inc_start) & (ts <= inc_end)

        for i in np.where(time_window_mask)[0]:
            svc = services.iloc[i]
            if svc in affected_services:
                is_incident[i] = True
                incident_id[i] = inc.get("incident_id")
                incident_type[i] = inc.get("incident_type")
            if svc == root_svc:
                is_root_cause[i] = True

    return pd.DataFrame({
        "is_incident": is_incident,
        "is_root_cause": is_root_cause,
        "incident_id": incident_id,
        "incident_type": incident_type,
    }, index=df.index)


def save_processed_telemetry(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save preprocessed telemetry DataFrame to Parquet.

    Args:
        df: Preprocessed telemetry DataFrame.
        output_path: Target path.

    Returns:
        Path of the saved file.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False, engine="pyarrow")
    return p
