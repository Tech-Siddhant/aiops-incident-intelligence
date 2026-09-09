"""Telemetry data loading and validation module."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import INCIDENTS_JSON, SYNTHETIC_DATA_DIR, TELEMETRY_PARQUET
from app.data.validation import validate_telemetry


def load_telemetry(
    path: str | Path | None = None,
    validate: bool = True,
    parse_timestamps: bool = False,
) -> pd.DataFrame:
    """Load telemetry data from a Parquet or CSV file.

    Args:
        path: Path to the telemetry file. Defaults to SYNTHETIC_DATA_DIR / TELEMETRY_PARQUET.
        validate: Whether to run data contract validation on the loaded dataset.
        parse_timestamps: Whether to parse the timestamp column to datetime64[ns, UTC].

    Returns:
        pd.DataFrame containing telemetry records.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is unsupported or schema validation fails.
    """
    file_path = Path(path) if path is not None else SYNTHETIC_DATA_DIR / TELEMETRY_PARQUET

    if not file_path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(file_path, engine="pyarrow")
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Expected .parquet or .csv")

    if validate:
        val_res = validate_telemetry(df)
        if not val_res.passed:
            err_msg = "; ".join(val_res.errors)
            raise ValueError(f"Telemetry schema validation failed: {err_msg}")

    if parse_timestamps and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df


def load_incidents(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load incident ground truth records from a JSON file.

    Args:
        path: Path to the incident JSON file. Defaults to SYNTHETIC_DATA_DIR / INCIDENTS_JSON.

    Returns:
        List of incident record dictionaries.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON structure is invalid.
    """
    file_path = Path(path) if path is not None else SYNTHETIC_DATA_DIR / INCIDENTS_JSON

    if not file_path.exists():
        raise FileNotFoundError(f"Incidents file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of incidents in {file_path}, got {type(data).__name__}")

    return data
