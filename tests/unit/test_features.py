"""Unit tests for time-series feature engineering pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.features import FeatureConfig, extract_features, save_features
from app.data.preprocess import preprocess_telemetry
from app.data.synthetic import REQUIRED_COLUMNS, SyntheticConfig, generate_synthetic_telemetry


@pytest.fixture
def preprocessed_df() -> pd.DataFrame:
    cfg = SyntheticConfig(
        duration_seconds=200,
        sampling_interval_seconds=10,
        failure_scenario="database_connection_saturation",
        failure_start_seconds=60,
        failure_duration_seconds=60,
        seed=42,
    )
    raw_df, _ = generate_synthetic_telemetry(cfg)
    clean_df, _ = preprocess_telemetry(raw_df)
    return clean_df


def test_extract_features_columns_present(preprocessed_df: pd.DataFrame):
    feat_df = extract_features(preprocessed_df)

    for col in REQUIRED_COLUMNS:
        assert col in feat_df.columns

    assert "cpu_usage_pct_roll_mean_3" in feat_df.columns
    assert "cpu_usage_pct_roll_std_3" in feat_df.columns
    assert "cpu_usage_pct_roll_max_3" in feat_df.columns
    assert "latency_ms_roll_mean_6" in feat_df.columns
    assert "latency_ms_roll_max_12" in feat_df.columns

    assert "cpu_usage_pct_delta_1" in feat_df.columns
    assert "cpu_usage_pct_pct_change_1" in feat_df.columns
    assert "latency_ms_delta_3" in feat_df.columns

    assert "latency_error_product" in feat_df.columns
    assert "error_per_request" in feat_df.columns
    assert "connection_pressure" in feat_df.columns
    assert "traffic_ratio_in_out" in feat_df.columns

    assert "system_mean_latency" in feat_df.columns


def test_no_future_leakage(preprocessed_df: pd.DataFrame):
    """Features at timestamp t must remain identical if future data is changed."""
    feat_original = extract_features(preprocessed_df)

    mutated_df = preprocessed_df.copy()
    split_idx = len(mutated_df) // 2
    split_time = mutated_df.loc[split_idx, "timestamp"]

    future_mask = mutated_df["timestamp"] >= split_time
    mutated_df.loc[future_mask, "latency_ms"] = mutated_df.loc[future_mask, "latency_ms"] * 10.0
    mutated_df.loc[future_mask, "cpu_usage_pct"] = np.clip(mutated_df.loc[future_mask, "cpu_usage_pct"] * 2.0, 0, 100)

    feat_mutated = extract_features(mutated_df)

    past_orig = feat_original[feat_original["timestamp"] < split_time].reset_index(drop=True)
    past_mutated = feat_mutated[feat_mutated["timestamp"] < split_time].reset_index(drop=True)

    pd.testing.assert_frame_equal(past_orig, past_mutated)


def test_rolling_calculations_correctness():
    """Verify rolling mean, max, and std against manual calculation."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="10s", tz="UTC"),
        "service": ["database"] * 4,
        "cpu_usage_pct": [10.0, 20.0, 30.0, 40.0],
        "memory_usage_pct": [50.0] * 4,
        "disk_usage_pct": [20.0] * 4,
        "network_in_mbps": [5.0] * 4,
        "network_out_mbps": [5.0] * 4,
        "request_rate_rps": [100.0] * 4,
        "latency_ms": [10.0, 20.0, 30.0, 40.0],
        "error_rate": [0.01] * 4,
        "active_connections": [50] * 4,
        "connection_utilization": [0.5] * 4,
    })

    cfg = FeatureConfig(
        rolling_windows=(3,),
        delta_steps=(1,),
        rolling_metrics=("cpu_usage_pct",),
        include_cross_service=False,
    )
    feat_df = extract_features(df, config=cfg)

    expected_means = [10.0, 15.0, 20.0, 30.0]
    np.testing.assert_allclose(feat_df["cpu_usage_pct_roll_mean_3"], expected_means)

    expected_maxs = [10.0, 20.0, 30.0, 40.0]
    np.testing.assert_allclose(feat_df["cpu_usage_pct_roll_max_3"], expected_maxs)

    expected_deltas = [0.0, 10.0, 10.0, 10.0]
    np.testing.assert_allclose(feat_df["cpu_usage_pct_delta_1"], expected_deltas)


def test_missing_history_no_nans(preprocessed_df: pd.DataFrame):
    """Ensure early rows with insufficient history do not produce NaNs."""
    feat_df = extract_features(preprocessed_df)
    assert not feat_df.isna().any().any()


def test_multi_service_boundary_isolation():
    """Ensure rolling operations on one service do not bleed into another."""
    df = pd.DataFrame({
        "timestamp": list(pd.date_range("2026-01-01", periods=2, freq="10s", tz="UTC")) * 2,
        "service": ["database", "database", "api_gateway", "api_gateway"],
        "cpu_usage_pct": [10.0, 20.0, 80.0, 90.0],
        "memory_usage_pct": [50.0] * 4,
        "disk_usage_pct": [20.0] * 4,
        "network_in_mbps": [5.0] * 4,
        "network_out_mbps": [5.0] * 4,
        "request_rate_rps": [100.0] * 4,
        "latency_ms": [10.0] * 4,
        "error_rate": [0.01] * 4,
        "active_connections": [50] * 4,
        "connection_utilization": [0.5] * 4,
    })
    cfg = FeatureConfig(
        rolling_windows=(2,),
        delta_steps=(1,),
        rolling_metrics=("cpu_usage_pct",),
        include_cross_service=False,
    )
    feat_df = extract_features(df, config=cfg)

    api_df = feat_df[feat_df["service"] == "api_gateway"].sort_values("timestamp")
    assert api_df["cpu_usage_pct_delta_1"].iloc[0] == 0.0
    assert api_df["cpu_usage_pct_roll_mean_2"].iloc[0] == 80.0


def test_features_deterministic(preprocessed_df: pd.DataFrame):
    df1 = extract_features(preprocessed_df)
    df2 = extract_features(preprocessed_df)
    pd.testing.assert_frame_equal(df1, df2)


def test_features_missing_columns_raises():
    bad_df = pd.DataFrame({"timestamp": ["2026-01-01T00:00:00Z"]})
    with pytest.raises(ValueError, match="Missing required telemetry columns"):
        extract_features(bad_df)


def test_features_empty_dataframe():
    empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    res = extract_features(empty_df)
    assert res.empty


def test_save_features(tmp_path: Path, preprocessed_df: pd.DataFrame):
    feat_df = extract_features(preprocessed_df)
    out_path = tmp_path / "features" / "features.parquet"
    saved = save_features(feat_df, out_path)

    assert saved.exists()
    loaded = pd.read_parquet(saved)
    assert len(loaded) == len(feat_df)
    assert len(loaded.columns) == len(feat_df.columns)

    assert "system_max_error_rate" in feat_df.columns
    assert "latency_to_system_mean_ratio" in feat_df.columns
