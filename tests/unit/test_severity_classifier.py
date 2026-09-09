"""Unit tests for incident severity classification baseline."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.data.features import extract_features
from app.data.preprocess import preprocess_telemetry
from app.data.synthetic import SyntheticConfig, generate_synthetic_telemetry
from app.models.severity_classifier import (
    SEVERITY_CLASSES,
    SeverityClassificationConfig,
    SeverityClassifierBaseline,
    calculate_severity_metrics,
    extract_severity_labels,
)


def test_extract_severity_labels():
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=5, freq="10s", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "service": ["database", "database", "database", "orders_service", "auth_service"],
    })
    incidents = [{
        "incident_id": "INC-1",
        "start_time": "2026-01-01T00:00:10Z",
        "end_time": "2026-01-01T00:00:30Z",
        "severity": "critical",
        "root_cause_service": "database",
        "affected_services": ["database", "orders_service"],
    }]

    labels = extract_severity_labels(df, incidents)
    # Row 0: 00:00:00 -> None
    # Row 1: 00:00:10, database -> critical
    # Row 2: 00:00:20, database -> critical
    # Row 3: 00:00:30, orders_service -> critical
    # Row 4: 00:00:40, auth_service -> None
    result = [None if pd.isna(x) else x for x in labels]
    assert result == [None, "critical", "critical", "critical", None]


def test_calculate_severity_metrics():
    y_true = ["low", "low", "medium", "high", "critical"]
    y_pred = ["low", "medium", "medium", "high", "critical"]

    metrics = calculate_severity_metrics(y_true, y_pred, labels=["low", "medium", "high", "critical"])
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert metrics.total_samples == 5
    assert "critical" in metrics.per_class_f1
    assert metrics.per_class_f1["critical"] == 1.0
    assert metrics.per_class_recall["low"] == 0.5
    assert len(metrics.confusion_matrix) == 4


def test_severity_classifier_multiclass_and_persistence(tmp_path: Path):
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="10s", tz="UTC"),
        "service": ["database"] * n,
        "latency_ms": np.concatenate([
            rng.normal(10, 2, 30),
            rng.normal(50, 5, 30),
            rng.normal(200, 20, 30),
            rng.normal(800, 50, 30),
        ]),
        "error_rate": np.concatenate([
            rng.uniform(0.001, 0.005, 30),
            rng.uniform(0.01, 0.05, 30),
            rng.uniform(0.1, 0.2, 30),
            rng.uniform(0.4, 0.8, 30),
        ]),
    })
    y = ["low"] * 30 + ["medium"] * 30 + ["high"] * 30 + ["critical"] * 30

    clf = SeverityClassifierBaseline(SeverityClassificationConfig(random_state=42))
    clf.fit(df, y)

    preds = clf.predict(df)
    probs = clf.predict_proba(df)
    pred_df = clf.predict_dataframe(df)

    assert len(preds) == n
    assert probs.shape == (n, 4)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert "predicted_severity" in pred_df.columns
    assert "prob_critical" in pred_df.columns

    metrics = clf.evaluate(df, y)
    assert metrics.macro_f1 > 0.85

    # Test serialization
    model_path = tmp_path / "severity_clf.joblib"
    clf.save(model_path)
    loaded = SeverityClassifierBaseline.load(model_path)
    loaded_preds = loaded.predict(df)
    assert (preds == loaded_preds).all()


def test_severity_classifier_single_class_edge_case():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="10s", tz="UTC"),
        "service": ["database"] * 10,
        "latency_ms": [10.0] * 10,
    })
    y = ["critical"] * 10

    clf = SeverityClassifierBaseline()
    clf.fit(df, y)
    preds = clf.predict(df)
    assert (preds == "critical").all()
    probs = clf.predict_proba(df)
    assert probs.shape == (10, 1)


def test_severity_classifier_on_synthetic_data():
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

    labels = extract_severity_labels(feat_df, incs)
    incident_mask = labels.notna()

    incident_feats = feat_df[incident_mask].reset_index(drop=True)
    incident_targets = labels[incident_mask].reset_index(drop=True)

    clf = SeverityClassifierBaseline()
    clf.fit(incident_feats, incident_targets)
    preds = clf.predict(incident_feats)
    metrics = clf.evaluate(incident_feats, incident_targets)

    assert len(preds) == len(incident_feats)
    assert metrics.total_samples == len(incident_feats)
