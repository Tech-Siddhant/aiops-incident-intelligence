"""Unit tests for configuration module."""
from pathlib import Path
from app.config import (
    ROOT_DIR,
    DATA_DIR,
    SYNTHETIC_DATA_DIR,
    DOCS_DIR,
    TELEMETRY_PARQUET,
    INCIDENTS_JSON,
    DEFAULT_SEED,
    DEFAULT_START_TIME,
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    DEFAULT_DURATION_SECONDS,
    DEFAULT_FAILURE_SCENARIO,
)


def test_paths_exist_or_valid():
    """Verify paths are Path instances and point within repository."""
    assert isinstance(ROOT_DIR, Path)
    assert isinstance(DATA_DIR, Path)
    assert isinstance(SYNTHETIC_DATA_DIR, Path)
    assert isinstance(DOCS_DIR, Path)
    assert DOCS_DIR.exists()


def test_runtime_defaults():
    """Verify deterministic runtime default constants."""
    assert DEFAULT_SEED == 42
    assert DEFAULT_START_TIME == "2026-01-01T00:00:00Z"
    assert DEFAULT_SAMPLING_INTERVAL_SECONDS == 10
    assert DEFAULT_DURATION_SECONDS == 3600
    assert DEFAULT_FAILURE_SCENARIO == "database_connection_saturation"
    assert TELEMETRY_PARQUET == "telemetry.parquet"
    assert INCIDENTS_JSON == "incidents.json"
