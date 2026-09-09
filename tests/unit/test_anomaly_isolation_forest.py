"""Unit tests for Isolation Forest anomaly detector."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.preprocess import preprocess_telemetry
from app.data.synthetic import SyntheticConfig, generate_synthetic_telemetry
from app.models.anomaly_isolation_forest import (
    IsolationForestConfig,
    IsolationForestDetector,
    detect_anomalies_iforest,
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


def test_iforest_output_schema(clean_normal_data: pd.DataFrame):
    detector = IsolationForestDetector(IsolationForestConfig(random_state=42))
    results = detector.fit_predict(clean_normal_data)
    assert set(results.columns) == {"timestamp", "service", "anomaly_score", "is_anomaly"}
    assert len(results) == len(clean_normal_data)
    assert pd.api.types.is_datetime64_any_dtype(results["timestamp"])
    assert pd.api.types.is_bool_dtype(results["is_anomaly"])
    assert pd.api.types.is_float_dtype(results["anomaly_score"])


def test_iforest_deterministic_output(clean_normal_data: pd.DataFrame):
    res1 = detect_anomalies_iforest(clean_normal_data, IsolationForestConfig(random_state=42))
    res2 = detect_anomalies_iforest(clean_normal_data, IsolationForestConfig(random_state=42))
    pd.testing.assert_frame_equal(res1, res2)


def test_iforest_missing_and_invalid_inputs():
    detector = IsolationForestDetector()

    # Unfitted predict raises error
    with pytest.raises(RuntimeError):
        detector.predict(pd.DataFrame({"timestamp": [pd.Timestamp.now()], "service": ["db"]}))

    # Empty fit raises error
    with pytest.raises(ValueError):
        detector.fit(pd.DataFrame())

    # Empty predict on fitted detector returns empty frame
    fitted_detector = IsolationForestDetector()
    train_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="10s", tz="UTC"),
        "service": ["database"] * 10,
        "latency_ms": [10.0] * 10,
        "error_rate": [0.0] * 10,
        "cpu_usage_pct": [20.0] * 10,
        "memory_usage_pct": [30.0] * 10,
        "disk_usage_pct": [40.0] * 10,
        "network_in_mbps": [5.0] * 10,
        "network_out_mbps": [5.0] * 10,
        "request_rate_rps": [50.0] * 10,
        "active_connections": [10] * 10,
        "connection_utilization": [0.1] * 10,
    })
    fitted_detector.fit(train_df)
    empty_res = fitted_detector.predict(pd.DataFrame())
    assert empty_res.empty
    assert "is_anomaly" in empty_res.columns

    # Handling NaNs and Infs in test data
    corrupted_df = train_df.copy()
    corrupted_df.loc[0, "latency_ms"] = np.nan
    corrupted_df.loc[1, "cpu_usage_pct"] = np.inf
    corrupted_df.loc[2, "memory_usage_pct"] = -np.inf
    res_corrupted = fitted_detector.predict(corrupted_df)
    assert len(res_corrupted) == len(corrupted_df)
    assert not res_corrupted["anomaly_score"].isna().any()


def test_iforest_no_future_leakage_when_fit_on_train(clean_normal_data: pd.DataFrame):
    """When trained on past data, modifying future test data does not alter past predictions."""
    split_time = clean_normal_data["timestamp"].iloc[len(clean_normal_data) // 2]
    train_df = clean_normal_data[clean_normal_data["timestamp"] < split_time]
    test_df_orig = clean_normal_data[clean_normal_data["timestamp"] >= split_time].copy()

    detector = IsolationForestDetector(IsolationForestConfig(random_state=42))
    detector.fit(train_df)

    res_orig = detector.predict(test_df_orig)

    # Corrupt second half of test dataset
    test_df_mut = test_df_orig.copy()
    mid_ts = test_df_mut["timestamp"].iloc[len(test_df_mut) // 2]
    test_df_mut.loc[test_df_mut["timestamp"] >= mid_ts, "latency_ms"] *= 100.0

    res_mut = detector.predict(test_df_mut)

    # Points before mid_ts should be identical
    orig_first_half = res_orig[res_orig["timestamp"] < mid_ts].reset_index(drop=True)
    mut_first_half = res_mut[res_mut["timestamp"] < mid_ts].reset_index(drop=True)

    pd.testing.assert_frame_equal(orig_first_half, mut_first_half)


def test_iforest_serialization(clean_normal_data: pd.DataFrame):
    detector = IsolationForestDetector(IsolationForestConfig(random_state=42))
    detector.fit(clean_normal_data)
    res1 = detector.predict(clean_normal_data)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "iforest_model.joblib"
        detector.save(model_path)
        loaded_detector = IsolationForestDetector.load(model_path)
        res2 = loaded_detector.predict(clean_normal_data)

        pd.testing.assert_frame_equal(res1, res2)


def test_iforest_detection_on_synthetic_failure(synthetic_incident_data):
    """Verify Isolation Forest flags anomalies during synthetic failure scenario."""
    df, incs = synthetic_incident_data
    # Train on normal baseline period (first 250s)
    train_df = df[df["timestamp"] < pd.to_datetime("2026-01-01 00:04:10", utc=True)]

    detector = IsolationForestDetector(IsolationForestConfig(contamination=0.1, random_state=42))
    detector.fit(train_df)
    results = detector.predict(df)

    inc = incs[0]
    start_ts = pd.to_datetime(inc["start_time"], utc=True)
    end_ts = pd.to_datetime(inc["end_time"], utc=True)

    during_incident = results[(results["timestamp"] >= start_ts) & (results["timestamp"] <= end_ts)]
    db_during = during_incident[during_incident["service"] == "database"]

    assert db_during["is_anomaly"].sum() > 0
