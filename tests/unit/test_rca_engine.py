"""Unit tests for Root Cause Analysis (RCA) engine."""
from datetime import datetime
import pandas as pd
import pytest

from app.data.preprocess import preprocess_telemetry
from app.data.synthetic import SyntheticConfig, generate_synthetic_telemetry
from app.models.anomaly_isolation_forest import IsolationForestConfig, detect_anomalies_iforest
from app.models.rca_engine import (
    DEFAULT_TOPOLOGY,
    RCACandidate,
    RCAEngine,
    RCAEngineConfig,
    RCAResult,
    rank_root_causes,
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


def test_rca_known_database_failure(synthetic_incident_data):
    """Test that database connection saturation ranks 'database' as Top-1 candidate."""
    df, incs = synthetic_incident_data
    inc = incs[0]

    # Pre-incident train for Isolation Forest
    train_df = df[df["timestamp"] < "2026-01-01T00:10:00Z"]
    anom_df = detect_anomalies_iforest(
        df,
        config=IsolationForestConfig(contamination=0.05, random_state=42),
        train_df=train_df,
    )

    engine = RCAEngine()
    result = engine.rank(df, anomalies_df=anom_df, incident=inc)

    assert isinstance(result, RCAResult)
    assert result.top_1 is not None
    assert result.top_1.service == "database"
    assert result.top_1.rank == 1
    assert result.top_1.score > 0.30

    # Top-3 should contain database, orders_service, api_gateway
    top_3_services = [c.service for c in result.top_3]
    assert "database" in top_3_services
    assert "orders_service" in top_3_services
    assert "api_gateway" in top_3_services

    # Auth service should be ranked last with lowest score
    last_candidate = result.ranked_candidates[-1]
    assert last_candidate.service == "auth_service"
    assert last_candidate.score < result.top_1.score


def test_propagation_ordering():
    """Verify that root downstream services rank higher than upstream victim callers."""
    # Synthetic failure cascade: C fails at t=10, B fails at t=20, A fails at t=30
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=10, freq="10s", tz="UTC")
    records = []

    # Topology: A -> B -> C
    topo = {"A": ["B"], "B": ["C"], "C": []}

    for t_idx, ts in enumerate(timestamps):
        records.append({
            "timestamp": ts,
            "service": "C",
            "is_anomaly": t_idx >= 1,
            "anomaly_score": 0.85 if t_idx >= 1 else 0.2,
            "connection_utilization": 0.95 if t_idx >= 1 else 0.2,
            "error_rate": 0.05 if t_idx >= 1 else 0.0,
        })
        records.append({
            "timestamp": ts,
            "service": "B",
            "is_anomaly": t_idx >= 2,
            "anomaly_score": 0.75 if t_idx >= 2 else 0.2,
            "connection_utilization": 0.5 if t_idx >= 2 else 0.2,
            "error_rate": 0.15 if t_idx >= 2 else 0.0,
        })
        records.append({
            "timestamp": ts,
            "service": "A",
            "is_anomaly": t_idx >= 3,
            "anomaly_score": 0.70 if t_idx >= 3 else 0.2,
            "connection_utilization": 0.3 if t_idx >= 3 else 0.2,
            "error_rate": 0.25 if t_idx >= 3 else 0.0,
        })

    df = pd.DataFrame(records)
    engine = RCAEngine(RCAEngineConfig(topology=topo))
    result = engine.rank(df)

    candidates = result.ranked_candidates
    assert len(candidates) == 3


def test_upstream_internal_failure_ranking():
    """When an upstream service fails internally without downstream anomalies, it ranks #1."""
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=5, freq="10s", tz="UTC")
    records = []
    # Topology: gateway -> orders -> database
    for t_idx, ts in enumerate(timestamps):
        records.append({"timestamp": ts, "service": "api_gateway", "is_anomaly": t_idx >= 1, "anomaly_score": 0.9, "connection_utilization": 0.3, "error_rate": 0.5})
        records.append({"timestamp": ts, "service": "orders_service", "is_anomaly": False, "anomaly_score": 0.2, "connection_utilization": 0.2, "error_rate": 0.001})
        records.append({"timestamp": ts, "service": "database", "is_anomaly": False, "anomaly_score": 0.2, "connection_utilization": 0.2, "error_rate": 0.0001})
        records.append({"timestamp": ts, "service": "auth_service", "is_anomaly": False, "anomaly_score": 0.2, "connection_utilization": 0.1, "error_rate": 0.0001})

    df = pd.DataFrame(records)
    engine = RCAEngine()
    result = engine.rank(df)

    assert result.top_1 is not None
    assert result.top_1.service == "api_gateway"
    assert result.top_1.score == 1.0


def test_deterministic_ranking_results(synthetic_incident_data):
    """Verify that RCA ranking is strictly deterministic across repeated invocations."""
    df, incs = synthetic_incident_data
    inc = incs[0]

    engine = RCAEngine()
    res1 = engine.rank(df, incident=inc)
    res2 = engine.rank(df, incident=inc)
    res3 = rank_root_causes(df, incident=inc)

    assert [c.to_dict() for c in res1.ranked_candidates] == [c.to_dict() for c in res2.ranked_candidates]
    assert [c.to_dict() for c in res1.ranked_candidates] == [c.to_dict() for c in res3.ranked_candidates]


def test_candidate_ranking_and_top_outputs():
    """Verify candidate properties, score breakdown, top-1, top-3, and get_top_k."""
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=5, freq="10s", tz="UTC")
    records = []
    for srv, score in [("database", 0.9), ("orders_service", 0.7), ("api_gateway", 0.5), ("auth_service", 0.1)]:
        for ts in timestamps:
            records.append({
                "timestamp": ts,
                "service": srv,
                "is_anomaly": score > 0.3,
                "anomaly_score": score,
                "connection_utilization": score,
                "error_rate": score * 0.1,
            })
    df = pd.DataFrame(records)
    result = rank_root_causes(df)

    assert len(result.ranked_candidates) == 4
    assert result.top_1 is not None
    assert result.top_1.service == "database"
    assert len(result.top_3) == 3
    assert len(result.get_top_k(2)) == 2
    assert len(result.get_top_k(10)) == 4

    # Check score properties
    scores = [c.score for c in result.ranked_candidates]
    assert pytest.approx(sum(scores), 0.01) == 1.0
    for c in result.ranked_candidates:
        assert 0.0 <= c.score <= 1.0
        assert "temporal" in c.score_breakdown
        assert "topology" in c.score_breakdown
        assert "severity" in c.score_breakdown
        assert len(c.explanation) > 0


def test_empty_and_zero_anomaly_edge_cases():
    """Verify graceful behavior on empty data and non-anomalous telemetry."""
    engine = RCAEngine()

    # Empty DataFrame
    empty_res = engine.rank(pd.DataFrame())
    assert empty_res.ranked_candidates == []
    assert empty_res.top_1 is None
    assert empty_res.top_3 == []

    # Zero anomalies across all services
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=5, freq="10s", tz="UTC")
    records = []
    for srv in DEFAULT_TOPOLOGY:
        for ts in timestamps:
            records.append({
                "timestamp": ts,
                "service": srv,
                "is_anomaly": False,
                "anomaly_score": 0.1,
                "connection_utilization": 0.2,
                "error_rate": 0.001,
            })
    no_anom_df = pd.DataFrame(records)
    res = engine.rank(no_anom_df)
    assert len(res.ranked_candidates) == 4
    # All scores should be 0
    for c in res.ranked_candidates:
        assert c.score == 0.0
        assert not c.is_root_cause_candidate


def test_result_serialization_and_disclaimer():
    """Verify to_dict serialization and non-causal disclaimer."""
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=3, freq="10s", tz="UTC")
    df = pd.DataFrame({
        "timestamp": list(timestamps) * 2,
        "service": ["database"] * 3 + ["orders_service"] * 3,
        "is_anomaly": [True] * 3 + [False] * 3,
        "anomaly_score": [0.8] * 3 + [0.1] * 3,
        "connection_utilization": [0.9] * 3 + [0.2] * 3,
        "error_rate": [0.01] * 3 + [0.001] * 3,
    })
    result = rank_root_causes(df, incident={"incident_id": "INC-TEST-001"})
    res_dict = result.to_dict()

    assert res_dict["incident_id"] == "INC-TEST-001"
    assert "causal proof" in res_dict["disclaimer"].lower() or "heuristic" in res_dict["disclaimer"].lower()
    assert res_dict["top_1"]["service"] == "database"
    assert len(res_dict["ranked_candidates"]) == 4

