"""Unit tests for lightweight drift detection."""
import numpy as np
import pandas as pd
import pytest

from app.mlops.drift import (
    DriftConfig,
    calculate_psi,
    detect_data_drift,
    detect_performance_degradation,
    evaluate_feature_drift,
)


def test_calculate_psi_identical_distribution():
    """Verify PSI is ~0 for identical distributions."""
    np.random.seed(42)
    ref = np.random.normal(loc=50.0, scale=5.0, size=500)
    curr = ref.copy()
    psi = calculate_psi(ref, curr)
    assert psi == pytest.approx(0.0, abs=1e-3)


def test_calculate_psi_shifted_distribution():
    """Verify PSI is high for distinct distributions."""
    np.random.seed(42)
    ref = np.random.normal(loc=50.0, scale=5.0, size=500)
    curr = np.random.normal(loc=70.0, scale=5.0, size=500)
    psi = calculate_psi(ref, curr)
    assert psi > 0.2  # Significant shift


def test_constant_features():
    """Verify constant identical vs constant shifted features."""
    ref_same = pd.Series([10.0] * 100)
    curr_same = pd.Series([10.0] * 100)
    res_same = evaluate_feature_drift(ref_same, curr_same, feature_name="constant_metric")
    assert res_same.drift_status == "NO_DRIFT"
    assert not res_same.is_drift

    curr_diff = pd.Series([20.0] * 100)
    res_diff = evaluate_feature_drift(ref_same, curr_diff, feature_name="constant_metric")
    assert res_diff.drift_status == "DRIFT"
    assert res_diff.is_drift


def test_missing_features_in_dataframe():
    """Verify missing columns in current/reference are flagged as drift/error."""
    ref_df = pd.DataFrame({"cpu": [10.0, 20.0, 30.0], "memory": [50.0, 60.0, 70.0]})
    curr_df = pd.DataFrame({"cpu": [10.0, 20.0, 30.0]})  # memory missing

    report = detect_data_drift(ref_df, curr_df)
    assert "memory" in report.drifted_features
    assert report.feature_results["memory"].drift_status == "MISSING"
    assert report.has_drift


def test_threshold_behavior_and_warning():
    """Verify warning threshold vs drift threshold triggering."""
    np.random.seed(42)
    ref = np.random.normal(loc=50.0, scale=5.0, size=1000)
    # Slight shift triggering warning
    curr = np.random.normal(loc=52.5, scale=5.0, size=1000)

    cfg = DriftConfig(
        psi_warning_threshold=0.05,
        psi_drift_threshold=0.5,
        ks_stat_warning_threshold=0.1,
        ks_stat_drift_threshold=0.5,
    )
    res = evaluate_feature_drift(ref, curr, feature_name="latency", config=cfg)
    assert res.drift_status == "WARNING"
    assert res.is_warning
    assert not res.is_drift


def test_detect_data_drift_multi_column():
    """Verify full DataFrame drift reporting."""
    np.random.seed(42)
    ref_df = pd.DataFrame(
        {
            "cpu": np.random.normal(50, 5, 200),
            "latency": np.random.normal(100, 10, 200),
            "errors": [0.0] * 200,
        }
    )
    curr_df = pd.DataFrame(
        {
            "cpu": np.random.normal(50, 5, 200),  # No drift
            "latency": np.random.normal(150, 10, 200),  # Heavy drift
            "errors": [0.0] * 200,  # Constant identical
        }
    )

    report = detect_data_drift(ref_df, curr_df)
    assert not report.feature_results["cpu"].is_drift
    assert report.feature_results["latency"].is_drift
    assert not report.feature_results["errors"].is_drift
    assert "latency" in report.drifted_features
    assert report.has_drift
    assert report.to_dict()["reference_count"] == 200


def test_detect_performance_degradation():
    """Verify model performance degradation detection against thresholds."""
    baseline = {"f1_score": 0.85, "precision": 0.80, "recall": 0.90}

    # Healthy current
    curr_ok = {"f1_score": 0.84, "precision": 0.81, "recall": 0.88}
    res_ok = detect_performance_degradation(baseline, curr_ok, max_allowed_drop=0.05)
    assert not res_ok.is_degraded
    assert len(res_ok.degraded_metrics) == 0

    # Degraded current (f1 dropped from 0.85 to 0.70 => relative drop > 17%)
    curr_bad = {"f1_score": 0.70, "precision": 0.65, "recall": 0.89}
    res_bad = detect_performance_degradation(baseline, curr_bad, max_allowed_drop=0.10)
    assert res_bad.is_degraded
    assert "f1_score" in res_bad.degraded_metrics
    assert "precision" in res_bad.degraded_metrics
    assert "recall" not in res_bad.degraded_metrics
