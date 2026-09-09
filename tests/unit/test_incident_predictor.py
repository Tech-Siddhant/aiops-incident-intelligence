"""Unit tests for incident prediction baseline model."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.data.features import extract_features
from app.data.preprocess import preprocess_telemetry
from app.data.synthetic import SyntheticConfig, generate_synthetic_telemetry
from app.models.incident_predictor import (
    IncidentPredictionConfig,
    IncidentPredictorBaseline,
    calculate_prediction_metrics,
    chronological_train_val_test_split,
    create_horizon_labels,
)


def test_create_horizon_labels():
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=6, freq="10min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "service": ["database"] * 6,
    })
    incidents = [{
        "incident_id": "INC-1",
        "start_time": "2026-01-01T00:30:00Z",
        "end_time": "2026-01-01T00:50:00Z",
        "root_cause_service": "database",
        "affected_services": ["database"],
    }]

    # Horizon of 20 minutes (1200 seconds):
    # At 00:00: 00:00 + 20min = 00:20 < 00:30 -> False
    # At 00:10: 00:10 + 20min = 00:30 >= 00:30 -> True
    # At 00:20: 00:20 + 20min = 00:40 >= 00:30 -> True
    # At 00:30: active -> True
    # At 00:40: active -> True
    # At 00:50: active -> True
    labels = create_horizon_labels(df, incidents, horizon_seconds=1200, lead_time_only=False)
    assert labels.tolist() == [False, True, True, True, True, True]

    # Lead time only (strictly before incident start):
    lead_labels = create_horizon_labels(df, incidents, horizon_seconds=1200, lead_time_only=True)
    assert lead_labels.tolist() == [False, True, True, False, False, False]


def test_chronological_train_val_test_split():
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=10, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "service": ["api_gateway"] * 10,
        "val": range(10),
    })
    y = pd.Series([0] * 5 + [1] * 5)

    train_df, val_df, test_df, y_tr, y_va, y_te = chronological_train_val_test_split(
        df, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )

    assert len(train_df) == 6
    assert len(val_df) == 2
    assert len(test_df) == 2

    # Verify temporal ordering: max(train) < min(val) and max(val) < min(test)
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()

    assert y_tr is not None and len(y_tr) == 6
    assert y_va is not None and len(y_va) == 2
    assert y_te is not None and len(y_te) == 2


def test_calculate_prediction_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_prob = [0.1, 0.6, 0.8, 0.9]

    metrics = calculate_prediction_metrics(y_true, y_pred, y_prob)
    assert metrics.precision == pytest.approx(2 / 3, 0.01)
    assert metrics.recall == 1.0
    assert metrics.total_samples == 4
    assert metrics.positive_samples == 2
    assert metrics.negative_samples == 2
    assert 0.0 <= metrics.pr_auc <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_model_fit_predict_save_load(tmp_path: Path):
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="10s", tz="UTC"),
        "service": ["database"] * n,
        "latency_ms": rng.normal(10, 2, n),
        "cpu_usage_pct": rng.normal(40, 5, n),
        "error_rate": rng.uniform(0, 0.05, n),
    })
    y = (df["latency_ms"] > 11).astype(int)

    model = IncidentPredictorBaseline(IncidentPredictionConfig(random_state=42))
    model.fit(df, y)

    probs = model.predict_proba(df)
    preds = model.predict(df)
    pred_df = model.predict_dataframe(df)

    assert len(probs) == n
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert len(preds) == n
    assert "incident_probability" in pred_df.columns
    assert "is_predicted_incident" in pred_df.columns

    # Save and load round-trip
    model_path = tmp_path / "baseline_model.joblib"
    model.save(model_path)
    loaded_model = IncidentPredictorBaseline.load(model_path)

    loaded_probs = loaded_model.predict_proba(df)
    np.testing.assert_allclose(probs, loaded_probs)


def test_single_class_edge_case():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="10s", tz="UTC"),
        "service": ["database"] * 10,
        "latency_ms": [10.0] * 10,
    })
    y = np.zeros(10, dtype=int)

    model = IncidentPredictorBaseline()
    model.fit(df, y)
    probs = model.predict_proba(df)
    assert len(probs) == 10
    metrics = model.evaluate(df, y)
    assert metrics.total_samples == 10


def test_full_pipeline_on_synthetic_data():
    cfg = SyntheticConfig(
        duration_seconds=1800,
        sampling_interval_seconds=10,
        failure_scenario="database_connection_saturation",
        failure_start_seconds=600,
        failure_duration_seconds=600,
        seed=42,
    )
    raw_df, incs = generate_synthetic_telemetry(cfg)
    clean_df, _ = preprocess_telemetry(raw_df)
    feat_df = extract_features(clean_df)

    y = create_horizon_labels(feat_df, incs, horizon_seconds=600)

    train_df, val_df, test_df, y_train, y_val, y_test = chronological_train_val_test_split(
        feat_df, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )

    model = IncidentPredictorBaseline(IncidentPredictionConfig(random_state=42))
    model.fit(train_df, y_train)

    val_metrics = model.evaluate(val_df, y_val)
    test_metrics = model.evaluate(test_df, y_test)

    assert isinstance(val_metrics.f1_score, float)
    assert isinstance(test_metrics.f1_score, float)
