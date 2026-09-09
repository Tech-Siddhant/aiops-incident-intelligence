"""Unit tests for the telemetry loader module."""
import json
from pathlib import Path

import pandas as pd
import pytest

from app.data.loader import load_incidents, load_telemetry
from app.data.synthetic import REQUIRED_COLUMNS, generate_synthetic_telemetry, SyntheticConfig


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """Generate a valid small parquet dataset."""
    cfg = SyntheticConfig(duration_seconds=60, sampling_interval_seconds=10, failure_scenario=None)
    df, _ = generate_synthetic_telemetry(cfg)
    p = tmp_path / "telemetry.parquet"
    df.to_parquet(p, index=False, engine="pyarrow")
    return p


@pytest.fixture
def sample_csv(tmp_path: Path, sample_parquet: Path) -> Path:
    """Convert the parquet fixture to CSV."""
    df = pd.read_parquet(sample_parquet)
    p = tmp_path / "telemetry.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def sample_incidents(tmp_path: Path) -> Path:
    """Write a minimal valid incidents JSON."""
    incs = [{"incident_id": "INC-T", "incident_type": "database_connection_saturation",
             "root_cause_service": "database", "start_time": "2026-01-01T00:00:00Z",
             "detection_time": "2026-01-01T00:00:10Z", "end_time": "2026-01-01T00:00:50Z",
             "severity": "critical", "description": "test"}]
    p = tmp_path / "incidents.json"
    p.write_text(json.dumps(incs), encoding="utf-8")
    return p


def test_load_parquet(sample_parquet: Path):
    df = load_telemetry(sample_parquet)
    assert isinstance(df, pd.DataFrame)
    assert set(REQUIRED_COLUMNS).issubset(df.columns)
    assert len(df) > 0


def test_load_csv(sample_csv: Path):
    df = load_telemetry(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert set(REQUIRED_COLUMNS).issubset(df.columns)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_telemetry("/nonexistent/path/telemetry.parquet")


def test_unsupported_format(tmp_path: Path):
    p = tmp_path / "data.xlsx"
    p.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_telemetry(p)


def test_invalid_schema(tmp_path: Path):
    """DataFrame missing required columns should raise ValueError."""
    df = pd.DataFrame({"timestamp": ["2026-01-01T00:00:00Z"], "service": ["x"]})
    p = tmp_path / "bad.parquet"
    df.to_parquet(p, index=False, engine="pyarrow")
    with pytest.raises(ValueError, match="schema validation failed"):
        load_telemetry(p)


def test_skip_validation(tmp_path: Path):
    """With validate=False, broken schema should load without error."""
    df = pd.DataFrame({"timestamp": ["2026-01-01T00:00:00Z"], "service": ["x"]})
    p = tmp_path / "bad.parquet"
    df.to_parquet(p, index=False, engine="pyarrow")
    loaded = load_telemetry(p, validate=False)
    assert len(loaded) == 1


def test_parse_timestamps(sample_parquet: Path):
    df = load_telemetry(sample_parquet, parse_timestamps=True)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_timestamps_stay_string_by_default(sample_parquet: Path):
    df = load_telemetry(sample_parquet)
    assert pd.api.types.is_string_dtype(df["timestamp"]) or pd.api.types.is_object_dtype(df["timestamp"])


def test_deterministic_load(sample_parquet: Path):
    df1 = load_telemetry(sample_parquet)
    df2 = load_telemetry(sample_parquet)
    pd.testing.assert_frame_equal(df1, df2)


def test_load_incidents(sample_incidents: Path):
    incs = load_incidents(sample_incidents)
    assert isinstance(incs, list)
    assert len(incs) == 1
    assert incs[0]["incident_id"] == "INC-T"


def test_load_incidents_missing_file():
    with pytest.raises(FileNotFoundError):
        load_incidents("/nonexistent/incidents.json")


def test_load_incidents_invalid_json(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a list"):
        load_incidents(p)
