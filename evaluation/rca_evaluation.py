"""Root Cause Analysis (RCA) ranking evaluation metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import pandas as pd

from app.models.rca_engine import RCAResult
from app.models.rca_explainer import RCAIncidentExplanation


@dataclass(frozen=True)
class RCAMetrics:
    """Evaluation metrics for incident root cause ranking."""
    top_1_accuracy: float
    top_3_accuracy: float
    mean_reciprocal_rank: float
    total_incidents: int
    valid_evaluations: int
    missing_ground_truth: int
    details: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_rca_rankings(
    results: list[tuple[RCAResult, dict[str, Any]]]
) -> RCAMetrics:
    """Calculate ranking accuracy metrics against ground truth incidents.

    Args:
        results: List of tuples containing (RCAResult, Incident Ground Truth Dict).

    Returns:
        RCAMetrics
    """
    total = len(results)
    if total == 0:
        return RCAMetrics(0.0, 0.0, 0.0, 0, 0, 0, [])

    top_1_hits = 0
    top_3_hits = 0
    mrr_sum = 0.0
    valid = 0
    missing_gt = 0
    details = []

    for res, inc in results:
        gt_root = inc.get("root_cause_service")
        if not gt_root:
            missing_gt += 1
            details.append({"incident_id": inc.get("incident_id"), "error": "Missing ground truth"})
            continue

        cands = res.ranked_candidates
        rank_found = None
        for cand in cands:
            if cand.service == gt_root:
                rank_found = cand.rank
                break

        did_top_1 = (rank_found == 1)
        did_top_3 = (rank_found is not None and rank_found <= 3)

        if rank_found is not None:
            top_1_hits += int(did_top_1)
            top_3_hits += int(did_top_3)
            mrr_sum += 1.0 / float(rank_found)
        valid += 1

        details.append({
            "incident_id": inc.get("incident_id"),
            "ground_truth_root": gt_root,
            "predicted_top_1": cands[0].service if cands else None,
            "actual_rank": rank_found,
            "top_1_hit": did_top_1,
            "top_3_hit": did_top_3,
        })

    if valid == 0:
        return RCAMetrics(0.0, 0.0, 0.0, total, 0, missing_gt, details)

    return RCAMetrics(
        top_1_accuracy=round(float(top_1_hits / valid), 4),
        top_3_accuracy=round(float(top_3_hits / valid), 4),
        mean_reciprocal_rank=round(float(mrr_sum / valid), 4),
        total_incidents=total,
        valid_evaluations=valid,
        missing_ground_truth=missing_gt,
        details=details,
    )
