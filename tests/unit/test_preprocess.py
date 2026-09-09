"""Unit tests for telemetry preprocessing and normalization pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.loader import load_telemetry
from app.data.preprocess import (
    NUMERIC_COLUMNS,
    PreprocessingReport,
    get_incident_labels,
    preprocess_telemetry,
    save_processed_telemetry,
)
from app.data.synthetic import REQUIRED_COLUMNS, SERVICES, SyntheticConfig, generate_synthetic_telemetry


@pytest.fixture
def clean_telemetry() -> tuple[pd.DataFrame, list[dict]]:
    cfg = SyntheticConfig(
        duration_seconds=120,
        sampling_interval_seconds=10,
        failure_scenario="database_connection_saturation",
        failure_start_seconds=30,
        failure_duration_seconds=40,
        seed=42,
    )
    return generate_synthetic_telemetry(cfg)


def test_preprocess_clean_data(clean_telemetry):
    raw_df, _ = clean_telemetry
    clean_df, report = preprocess_telemetry(raw_df)

    assert len(clean_df) == len(raw_df)
    assert report.input_rows == len(raw_df)
    assert report.output_rows == len(raw_df)
    assert report.total_dropped == 0
    assert set(clean_df.columns) == set(REQUIRED_COLUMNS)
    assert pd.api.types.is_datetime64_any_dtype(clean_df["timestamp"])
    assert clean_df["active_connections"].dtype == int


def test_preprocess_missing_columns_raises():
    bad_df = pd.DataFrame({"timestamp": ["2026-01-01T00:00:00Z"], "service": ["database"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess_telemetry(bad_df)


def test_preprocess_handles_empty_df():
    empty_df = pd.DataFrame()
    clean_df, report = preprocess_telemetry(empty_df)
    assert len(clean_df) == 0
    assert report.input_rows == 0
    assert report.output_rows == 0


def test_preprocess_drops_null_records(clean_telemetry):
    raw_df, _ = clean_telemetry
    corrupted = raw_df.copy()
    corrupted.loc[0, "cpu_usage_pct"] = np.nan
    corrupted.loc[2, "latency_ms"] = None

    clean_df, report = preprocess_telemetry(corrupted)
    assert report.null_rows_dropped == 2
    assert len(clean_df) == len(raw_df) - 2


def test_preprocess_drops_unknown_services(clean_telemetry):
    raw_df, _ = clean_telemetry
    corrupted = raw_df.copy()
    corrupted.loc[0, "service"] = "rogue_service"
    corrupted.loc[1, "service"] = "payment_v2"

    clean_df, report = preprocess_telemetry(corrupted)
    assert report.unknown_service_rows_dropped == 2
    assert set(clean_df["service"].unique()).issubset(set(SERVICES))


def test_preprocess_drops_duplicate_records(clean_telemetry):
    raw_df, _ = clean_telemetry
    dup_row = raw_df.iloc[[0]].copy()
    corrupted = pd.concat([raw_df, dup_row], ignore_index=True)

    clean_df, report = preprocess_telemetry(corrupted)
    assert report.duplicate_rows_dropped == 1
    assert len(clean_df) == len(raw_df)


def test_preprocess_drops_out_of_bounds_records(clean_telemetry):
    raw_df, _ = clean_telemetry
    corrupted = raw_df.copy()
    corrupted.loc[0, "cpu_usage_pct"] = 150.0   # > 100
    corrupted.loc[1, "latency_ms"] = -5.0        # < 0
    corrupted.loc[2, "error_rate"] = 1.5         # > 1.0
    corrupted.loc[3, "connection_utilization"] = -0.1  # < 0

    clean_df, report = preprocess_telemetry(corrupted)
    assert report.invalid_rows_dropped == 4
    assert len(clean_df) == len(raw_df) - 4


def test_preprocess_deterministic(clean_telemetry):
    raw_df, _ = clean_telemetry
    df1, rep1 = preprocess_telemetry(raw_df)
    df2, rep2 = preprocess_telemetry(raw_df)

    pd.testing.assert_frame_equal(df1, df2)
    assert rep1 == rep2


def test_incident_labels_leakage_free(clean_telemetry):
    raw_df, incidents = clean_telemetry
    clean_df, _ = preprocess_telemetry(raw_df)

    labels = get_incident_labels(clean_df, incidents)

    assert len(labels) == len(clean_df)
    assert set(labels.columns) == {"is_incident", "is_root_cause", "incident_id", "incident_type"}
    assert labels["is_incident"].sum() > 0
    assert labels["is_root_cause"].sum() > 0

    # Ensure clean_df itself is unchanged and free of label columns
    assert "is_incident" not in clean_df.columns
    assert "is_root_cause" not in clean_df.columns


def test_save_processed_telemetry(tmp_path: Path, clean_telemetry):
    raw_df, _ = clean_telemetry
    clean_df, _ = preprocess_telemetry(raw_df)
    out_file = tmp_path / "processed" / "telemetry_clean.parquet"

    saved = save_processed_telemetry(clean_df, out_file)
    assert saved.exists()

    loaded = pd.read_parquet(saved)
    assert len(loaded) == len(clean_df)
