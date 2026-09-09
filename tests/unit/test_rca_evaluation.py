"""Unit tests for RCA evaluation metrics."""
from typing import Any
import pytest

from app.models.rca_engine import RCACandidate, RCAResult
from evaluation.rca_evaluation import RCAMetrics, evaluate_rca_rankings


def _make_dummy_result(ordered_services: list[str], incident_id: str = "INC-1") -> RCAResult:
    candidates = []
    for i, srv in enumerate(ordered_services, start=1):
        candidates.append(
            RCACandidate(
                service=srv,
                score=1.0 / i,
                raw_score=10.0 / i,
                rank=i,
                earliest_anomaly_time=None,
                anomaly_count=10,
                anomaly_ratio=1.0,
                mean_anomaly_score=0.8,
                max_anomaly_score=0.9,
                score_breakdown={},
                explanation="",
            )
        )
    return RCAResult(ranked_candidates=candidates, incident_id=incident_id)


def test_rca_metrics_perfect_ranking():
    """Verify metrics calculation for perfect top-1 rankings."""
    results = [
        (_make_dummy_result(["db", "api", "auth"]), {"incident_id": "1", "root_cause_service": "db"}),
        (_make_dummy_result(["api", "auth", "db"]), {"incident_id": "2", "root_cause_service": "api"}),
    ]

    metrics = evaluate_rca_rankings(results)
    assert metrics.total_incidents == 2
    assert metrics.valid_evaluations == 2
    assert metrics.missing_ground_truth == 0
    assert metrics.top_1_accuracy == 1.0
    assert metrics.top_3_accuracy == 1.0
    assert metrics.mean_reciprocal_rank == 1.0


def test_rca_metrics_mixed_ranking():
    """Verify MRR and accuracy degradation for lower ranked ground truth."""
    results = [
        # Rank 2 = MRR 0.5
        (_make_dummy_result(["api", "db", "auth"]), {"incident_id": "1", "root_cause_service": "db"}),
        # Rank 3 = MRR 0.333
        (_make_dummy_result(["auth", "api", "db"]), {"incident_id": "2", "root_cause_service": "db"}),
        # Rank 4 = MRR 0.25 (not top-3)
        (_make_dummy_result(["a", "b", "c", "db"]), {"incident_id": "3", "root_cause_service": "db"}),
    ]

    metrics = evaluate_rca_rankings(results)
    assert metrics.valid_evaluations == 3
    assert metrics.top_1_accuracy == 0.0
    # 2/3 incidents in top 3
    assert metrics.top_3_accuracy == pytest.approx(0.6667, 0.01)
    # MRR = (1/2 + 1/3 + 1/4) / 3 = 1.0833 / 3 = 0.3611
    assert metrics.mean_reciprocal_rank == pytest.approx(0.3611, 0.01)


def test_rca_metrics_missing_ground_truth():
    """Verify resilient evaluation when ground truth is missing."""
    results = [
        (_make_dummy_result(["db"]), {"incident_id": "1"}),  # No root_cause_service
    ]
    metrics = evaluate_rca_rankings(results)
    assert metrics.valid_evaluations == 0
    assert metrics.missing_ground_truth == 1
    assert metrics.top_1_accuracy == 0.0


def test_rca_metrics_empty():
    empty_metrics = evaluate_rca_rankings([])
    assert empty_metrics.total_incidents == 0
    assert empty_metrics.mean_reciprocal_rank == 0.0
