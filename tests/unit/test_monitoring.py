"""Unit tests for lightweight MLOps monitoring and health evaluation."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.mlops.drift import DriftConfig
from app.mlops.monitoring import (
    DataQualityReport,
    ModelHealthReport,
    check_data_quality,
    evaluate_model_health,
    profile_inference,
)
from app.models.anomaly_isolation_forest import IsolationForestConfig, IsolationForestDetector


def _sample_telemetry_df(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="10s", tz="UTC").astype(str),
            "service": ["api_gateway"] * n,
            "latency_ms": np.random.normal(50.0, 5.0, n),
            "error_rate": np.zeros(n),
            "cpu_usage_pct": np.random.uniform(20.0, 40.0, n),
            "memory_usage_pct": np.random.uniform(30.0, 50.0, n),
            "disk_usage_pct": np.random.uniform(10.0, 20.0, n),
            "network_in_mbps": np.random.uniform(10.0, 50.0, n),
            "network_out_mbps": np.random.uniform(10.0, 50.0, n),
            "request_rate_rps": np.random.uniform(100.0, 200.0, n),
            "active_connections": np.random.randint(10, 50, n),
            "connection_utilization": np.random.uniform(0.1, 0.5, n),
        }
    )


def test_check_data_quality_valid():
    df = _sample_telemetry_df()
    dq = check_data_quality(df)
    assert dq.is_valid
    assert len(dq.missing_columns) == 0
    assert len(dq.null_counts) == 0
    assert len(dq.infinite_counts) == 0


def test_check_data_quality_invalid():
    df = _sample_telemetry_df()
    df.loc[0, "latency_ms"] = np.nan
    df.loc[1, "cpu_usage_pct"] = np.inf
    df = df.drop(columns=["disk_usage_pct"])

    dq = check_data_quality(df)
    assert not dq.is_valid
    assert "disk_usage_pct" in dq.missing_columns
    assert "latency_ms" in dq.null_counts
    assert "cpu_usage_pct" in dq.infinite_counts


def test_profile_inference_and_resources():
    df = _sample_telemetry_df(50)
    detector = IsolationForestDetector(config=IsolationForestConfig(n_estimators=10))
    detector.fit(df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        art_path = Path(tmp_dir) / "model.joblib"
        detector.save(art_path)

        preds, res_usage, pred_dist = profile_inference(detector, df, artifact_path=art_path)

        assert len(preds) == 50
        assert res_usage.latency_ms >= 0.0
        assert res_usage.peak_memory_bytes > 0
        assert res_usage.artifact_size_bytes is not None
        assert res_usage.artifact_size_bytes > 0
        assert pred_dist.total_predictions == 50
        assert 0.0 <= pred_dist.positive_rate <= 1.0


def test_evaluate_model_health_normal():
    ref_df = _sample_telemetry_df(100)
    curr_df = _sample_telemetry_df(100)

    detector = IsolationForestDetector(config=IsolationForestConfig(n_estimators=10))
    detector.fit(ref_df)

    health = evaluate_model_health(
        model=detector,
        current_df=curr_df,
        reference_df=ref_df,
        model_name="isolation_forest",
        model_version="1.0.0",
        evaluation_metrics={"f1_score": 0.77},
        latency_sla_ms=500.0,
    )

    assert health.health_status == "NORMAL"
    assert health.data_quality.is_valid
    assert health.drift_report is not None
    assert not health.drift_report.has_drift
    assert health.evaluation_metrics["f1_score"] == 0.77


def test_evaluate_model_health_critical_on_data_quality():
    ref_df = _sample_telemetry_df(50)
    bad_curr = _sample_telemetry_df(50).drop(columns=["latency_ms"])

    detector = MagicMock()
    detector.predict.return_value = np.zeros(50, dtype=bool)

    health = evaluate_model_health(
        model=detector,
        current_df=bad_curr,
        reference_df=ref_df,
    )

    assert health.health_status == "CRITICAL"
    assert not health.data_quality.is_valid


def test_evaluate_model_health_warning_on_drift():
    ref_df = _sample_telemetry_df(200)
    curr_df = _sample_telemetry_df(200)
    # Intentionally shift one feature slightly to trigger warning
    curr_df["cpu_usage_pct"] += 3.0

    detector = MagicMock()
    detector.predict.return_value = np.zeros(200, dtype=bool)

    cfg = DriftConfig(
        psi_warning_threshold=0.01,
        psi_drift_threshold=1.0,
        ks_stat_warning_threshold=0.05,
        ks_stat_drift_threshold=1.0,
    )

    health = evaluate_model_health(
        model=detector,
        current_df=curr_df,
        reference_df=ref_df,
        drift_config=cfg,
    )

    assert health.health_status == "WARNING"
