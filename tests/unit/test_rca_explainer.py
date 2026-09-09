"""Unit tests for RCA Explainability Layer."""
from datetime import datetime
import pandas as pd
import pytest

from app.data.preprocess import preprocess_telemetry
from app.data.synthetic import SyntheticConfig, generate_synthetic_telemetry
from app.models.anomaly_isolation_forest import IsolationForestConfig, detect_anomalies_iforest
from app.models.rca_engine import (
    DEFAULT_TOPOLOGY,
    RCAEngine,
    RCAEngineConfig,
    rank_root_causes,
)
from app.models.rca_explainer import (
    CandidateExplanation,
    DependencyEvidence,
    InferenceBreakdown,
    RCAExplainer,
    RCAIncidentExplanation,
    TelemetrySignalEvidence,
    TemporalEvidence,
    explain_rca_result,
    generate_rca_explanation,
)


@pytest.fixture
def synthetic_incident_data():
    """Synthetic telemetry with database connection saturation incident."""
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
    return clean_df, incs


def test_explain_known_database_failure(synthetic_incident_data):
    """Verify structured explanation for database connection saturation cascade."""
    df, incs = synthetic_incident_data
    inc = incs[0]

    train_df = df[df["timestamp"] < "2026-01-01T00:10:00Z"]
    anom_df = detect_anomalies_iforest(
        df,
        config=IsolationForestConfig(contamination=0.05, random_state=42),
        train_df=train_df,
    )

    rca_res = rank_root_causes(df, anomalies_df=anom_df, incident=inc)
    expl = explain_rca_result(rca_res, telemetry_df=df)

    assert isinstance(expl, RCAIncidentExplanation)
    assert expl.incident_id == inc["incident_id"]
    assert expl.probable_root_cause == "database"
    assert expl.confidence in ("HIGH", "MEDIUM")

    top_cand = expl.top_1
    assert top_cand is not None
    assert top_cand.candidate_service == "database"
    assert top_cand.rank == 1
    assert top_cand.classification == "Probable root cause"
    assert top_cand.score > 0.30

    # Check supporting signals & contributing metrics
    assert len(top_cand.supporting_signals) > 0
    signal_text = " ".join(top_cand.supporting_signals).lower()
    assert "connection" in signal_text or "saturation" in signal_text or "latency" in signal_text

    # Check temporal evidence
    assert top_cand.temporal_evidence.earliest_anomaly_timestamp is not None
    assert top_cand.temporal_evidence.onset_delta_seconds == 0.0
    assert top_cand.temporal_evidence.anomaly_samples > 0

    # Check dependency evidence
    assert top_cand.dependency_evidence.topology_depth == 2
    assert "orders_service" in top_cand.dependency_evidence.upstream_callers
    assert top_cand.dependency_evidence.downstream_dependencies == []

def test_distinguish_evidence_from_inference(synthetic_incident_data):
    """Ensure factual telemetry evidence is strictly distinguished from derived inferences."""
    df, incs = synthetic_incident_data
    expl = generate_rca_explanation(df, incident=incs[0])

    top_cand = expl.top_1
    assert top_cand is not None

    # Factual observed evidence
    assert isinstance(top_cand.temporal_evidence, TemporalEvidence)
    assert isinstance(top_cand.dependency_evidence, DependencyEvidence)
    assert all(isinstance(m, TelemetrySignalEvidence) for m in top_cand.contributing_metrics)

    # Derived inference
    assert isinstance(top_cand.inference, InferenceBreakdown)
    assert 0.0 <= top_cand.inference.temporal_score <= 1.0
    assert 0.0 <= top_cand.inference.topology_score <= 1.0
    assert 0.0 <= top_cand.inference.severity_score <= 1.0
    assert top_cand.inference.composite_score == top_cand.score
    assert len(top_cand.inference.confidence_rationale) > 0


def test_language_and_safety_disclaimers(synthetic_incident_data):
    """Ensure no unverified 'root cause confirmed' statements are produced."""
    df, incs = synthetic_incident_data
    expl = generate_rca_explanation(df, incident=incs[0])

    all_texts = [
        expl.summary,
        expl.disclaimer,
        *(c.summary for c in expl.candidates),
        *(lim for c in expl.candidates for lim in c.limitations),
    ]

    for text in all_texts:
        assert "root cause confirmed" not in text.lower()
        assert "definitive causal proof" not in text.lower() or "not" in text.lower()

    # Verify standard terminology usage
    full_report = " ".join(all_texts)
    assert "Probable root cause" in full_report
    assert "Supporting evidence" in full_report


def test_propagation_cascade_explanation():
    """Verify multi-tier cascade (A -> B -> C) explanation roles."""
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=6, freq="10s", tz="UTC")
    records = []
    topo = {"A": ["B"], "B": ["C"], "C": []}

    for t_idx, ts in enumerate(timestamps):
        records.append({"timestamp": ts, "service": "C", "is_anomaly": t_idx >= 1, "anomaly_score": 0.9, "connection_utilization": 0.95, "error_rate": 0.05, "latency_ms": 500.0})
        records.append({"timestamp": ts, "service": "B", "is_anomaly": t_idx >= 2, "anomaly_score": 0.7, "connection_utilization": 0.5, "error_rate": 0.15, "latency_ms": 300.0})
        records.append({"timestamp": ts, "service": "A", "is_anomaly": t_idx >= 3, "anomaly_score": 0.6, "connection_utilization": 0.2, "error_rate": 0.20, "latency_ms": 200.0})

    df = pd.DataFrame(records)
    expl = generate_rca_explanation(
        telemetry_df=df,
        engine_config=RCAEngineConfig(topology=topo),
    )

    assert expl.probable_root_cause == "C"
    cand_c = next(c for c in expl.candidates if c.candidate_service == "C")
    cand_b = next(c for c in expl.candidates if c.candidate_service == "B")
    cand_a = next(c for c in expl.candidates if c.candidate_service == "A")

    assert cand_c.classification == "Probable root cause"
    assert "Cascading failure symptom" in cand_b.classification
    assert "Cascading failure symptom" in cand_a.classification
    assert cand_c.temporal_evidence.onset_delta_seconds == 0.0
    assert cand_b.temporal_evidence.onset_delta_seconds == 10.0
    assert cand_a.temporal_evidence.onset_delta_seconds == 20.0

def test_deterministic_explanations(synthetic_incident_data):
    """Verify explanations are completely deterministic across runs."""
    df, incs = synthetic_incident_data
    expl1 = generate_rca_explanation(df, incident=incs[0])
    expl2 = generate_rca_explanation(df, incident=incs[0])

    assert expl1.to_dict() == expl2.to_dict()


def test_empty_and_zero_anomaly_edge_cases():
    """Verify graceful explanations when telemetry is empty or has zero anomalies."""
    # 1. Empty dataframe
    empty_expl = generate_rca_explanation(pd.DataFrame())
    assert empty_expl.candidates == []
    assert empty_expl.probable_root_cause is None
    assert empty_expl.confidence == "LOW"

    # 2. Telemetry with no anomalies
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=5, freq="10s", tz="UTC")
    records = []
    for srv in DEFAULT_TOPOLOGY:
        for ts in timestamps:
            records.append({
                "timestamp": ts,
                "service": srv,
                "is_anomaly": False,
                "anomaly_score": 0.05,
                "connection_utilization": 0.2,
                "error_rate": 0.001,
                "latency_ms": 25.0,
            })
    no_anom_df = pd.DataFrame(records)
    res = generate_rca_explanation(no_anom_df)
    assert res.probable_root_cause is None
    assert all(c.classification == "Unaffected service" for c in res.candidates)


def test_serialization_structure():
    """Verify dictionary serialization has all specified fields."""
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=3, freq="10s", tz="UTC")
    df = pd.DataFrame({
        "timestamp": list(timestamps) * 2,
        "service": ["database"] * 3 + ["orders_service"] * 3,
        "is_anomaly": [True] * 3 + [False] * 3,
        "anomaly_score": [0.85] * 3 + [0.1] * 3,
        "connection_utilization": [0.95] * 3 + [0.2] * 3,
        "error_rate": [0.01] * 3 + [0.001] * 3,
        "latency_ms": [450.0] * 3 + [50.0] * 3,
    })
    expl = generate_rca_explanation(df)
    d = expl.to_dict()

    assert "probable_root_cause" in d
    assert "confidence" in d
    assert "summary" in d
    assert "candidates" in d
    assert "disclaimer" in d

    cand_d = d["candidates"][0]
    required_keys = [
        "candidate_service",
        "rank",
        "score",
        "confidence",
        "classification",
        "supporting_signals",
        "contributing_metrics",
        "temporal_evidence",
        "dependency_evidence",
        "inference",
        "summary",
        "limitations",
    ]
    for k in required_keys:
        assert k in cand_d


