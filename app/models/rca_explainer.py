"""Deterministic explainability layer for Root Cause Analysis (RCA) ranking.

Translates mathematical RCA ranking signals, temporal anomaly onsets,
topology flows, and metric deviations into structured, engineer-readable
evidence.

Important:
    RCA explanations present *probable* root cause hypotheses and supporting
    evidence. They do not claim absolute causal proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

from app.models.rca_engine import (
    DEFAULT_TOPOLOGY,
    RCACandidate,
    RCAEngineConfig,
    RCAResult,
    rank_root_causes,
)

METRIC_FORMATTERS: dict[str, tuple[str, str, float]] = {
    "connection_utilization": ("Connection Pool Utilization", "%", 100.0),
    "error_rate": ("Error Rate", "%", 100.0),
    "latency_ms": ("Response Latency", "ms", 1.0),
    "cpu_usage_pct": ("CPU Usage", "%", 1.0),
    "memory_usage_pct": ("Memory Usage", "%", 1.0),
    "active_connections": ("Active Connections", " conn", 1.0),
    "request_rate_rps": ("Request Rate", " rps", 1.0),
}


@dataclass(frozen=True)
class RCAExplainerConfig:
    """Configuration for RCA explainability layer."""
    high_confidence_score: float = 0.45
    high_confidence_margin: float = 0.15
    medium_confidence_score: float = 0.30
    tracked_metrics: tuple[str, ...] = (
        "connection_utilization",
        "error_rate",
        "latency_ms",
        "cpu_usage_pct",
        "memory_usage_pct",
        "active_connections",
    )


@dataclass(frozen=True)
class TelemetrySignalEvidence:
    """Observed factual telemetry metric deviation."""
    metric_name: str
    display_name: str
    peak_value: float
    mean_value: float
    unit: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalEvidence:
    """Observed factual temporal anomaly timings."""
    earliest_anomaly_timestamp: str | None
    onset_delta_seconds: float | None
    anomaly_samples: int
    anomaly_ratio: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DependencyEvidence:
    """Observed factual topological relationships and propagation path."""
    topology_depth: int
    upstream_callers: list[str]
    downstream_dependencies: list[str]
    anomalous_callers: list[str]
    anomalous_dependencies: list[str]
    propagation_role: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceBreakdown:
    """Derived heuristic ranking inference (distinguished from factual evidence)."""
    temporal_score: float
    topology_score: float
    severity_score: float
    composite_score: float
    confidence: str
    confidence_rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateExplanation:
    """Complete structured explanation for a single RCA candidate."""
    candidate_service: str
    rank: int
    score: float
    confidence: str
    classification: str
    supporting_signals: list[str]
    contributing_metrics: list[TelemetrySignalEvidence]
    temporal_evidence: TemporalEvidence
    dependency_evidence: DependencyEvidence
    inference: InferenceBreakdown
    summary: str
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_service": self.candidate_service,
            "rank": self.rank,
            "score": self.score,
            "confidence": self.confidence,
            "classification": self.classification,
            "supporting_signals": list(self.supporting_signals),
            "contributing_metrics": [m.to_dict() for m in self.contributing_metrics],
            "temporal_evidence": self.temporal_evidence.to_dict(),
            "dependency_evidence": self.dependency_evidence.to_dict(),
            "inference": self.inference.to_dict(),
            "summary": self.summary,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class RCAIncidentExplanation:
    """Structured incident explanation report across all candidates."""
    incident_id: str | None
    evaluation_window: tuple[str | None, str | None]
    probable_root_cause: str | None
    confidence: str
    summary: str
    candidates: list[CandidateExplanation]
    disclaimer: str = (
        "This explanation represents a probabilistic heuristic ranking based on temporal "
        "anomaly onset, topology graph structure, and metric deviation magnitudes. It is "
        "supporting evidence for triage and not verified causal proof."
    )

    @property
    def top_1(self) -> CandidateExplanation | None:
        return self.candidates[0] if self.candidates else None

    @property
    def top_3(self) -> list[CandidateExplanation]:
        return self.candidates[:3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "evaluation_window": list(self.evaluation_window),
            "probable_root_cause": self.probable_root_cause,
            "confidence": self.confidence,
            "summary": self.summary,
            "candidates": [c.to_dict() for c in self.candidates],
            "top_1": self.top_1.to_dict() if self.top_1 else None,
            "top_3": [c.to_dict() for c in self.top_3],
            "disclaimer": self.disclaimer,
        }


class RCAExplainer:
    """Explains RCA rankings by synthesizing observed telemetry evidence and heuristic inferences."""

    def __init__(
        self,
        config: RCAExplainerConfig | None = None,
        topology: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.config = config or RCAExplainerConfig()
        self.topology: dict[str, list[str]] = (
            {k: list(v) for k, v in topology.items()}
            if topology is not None
            else {k: list(v) for k, v in DEFAULT_TOPOLOGY.items()}
        )
        self.reverse_callers: dict[str, list[str]] = {s: [] for s in self.topology}
        for parent, children in self.topology.items():
            for child in children:
                if child in self.reverse_callers:
                    self.reverse_callers[child].append(parent)

    def explain(
        self,
        rca_result: RCAResult,
        telemetry_df: pd.DataFrame,
    ) -> RCAIncidentExplanation:
        """Generate structured explanations for all candidates in an RCAResult."""
        if not rca_result.ranked_candidates:
            return RCAIncidentExplanation(
                incident_id=rca_result.incident_id,
                evaluation_window=rca_result.evaluation_window,
                probable_root_cause=None,
                confidence="LOW",
                summary="No telemetry candidates available for RCA explanation.",
                candidates=[],
            )

        work_df = telemetry_df.copy()
        if not work_df.empty and "timestamp" in work_df.columns:
            work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)
            win_start, win_end = rca_result.evaluation_window
            if win_start:
                work_df = work_df[work_df["timestamp"] >= pd.to_datetime(win_start, utc=True)]
            if win_end:
                work_df = work_df[work_df["timestamp"] <= pd.to_datetime(win_end, utc=True)]

        all_onsets = [
            pd.to_datetime(c.earliest_anomaly_time, utc=True)
            for c in rca_result.ranked_candidates
            if c.earliest_anomaly_time is not None
        ]
        global_earliest_ts = min(all_onsets) if all_onsets else None
        anomalous_services = {c.service for c in rca_result.ranked_candidates if c.anomaly_count > 0}

        candidate_explanations: list[CandidateExplanation] = []
        for cand in rca_result.ranked_candidates:
            srv = cand.service
            srv_df = work_df[work_df["service"] == srv] if "service" in work_df.columns else pd.DataFrame()

            contributing_metrics, supporting_signals = self._extract_telemetry_evidence(srv_df)
            temp_ev = self._extract_temporal_evidence(cand, global_earliest_ts)
            dep_ev = self._extract_dependency_evidence(srv, anomalous_services)
            confidence, conf_rationale = self._evaluate_confidence(cand, rca_result.ranked_candidates)
            classification = self._determine_classification(cand, dep_ev)

            limitations = [
                "Heuristic candidate ranking is not definitive causal proof.",
                "Unobserved network hops, external third-party outages, or OS hardware faults cannot be ruled out.",
                "Propagation reasoning assumes a directed acyclic graph (DAG) topology.",
            ]

            inference = InferenceBreakdown(
                temporal_score=cand.score_breakdown.get("temporal", 0.0),
                topology_score=cand.score_breakdown.get("topology", 0.0),
                severity_score=cand.score_breakdown.get("severity", 0.0),
                composite_score=cand.score,
                confidence=confidence,
                confidence_rationale=conf_rationale,
            )

            summary = self._build_candidate_summary(
                cand=cand,
                classification=classification,
                confidence=confidence,
                supporting_signals=supporting_signals,
                temporal_evidence=temp_ev,
                dependency_evidence=dep_ev,
            )

            candidate_explanations.append(
                CandidateExplanation(
                    candidate_service=srv,
                    rank=cand.rank,
                    score=cand.score,
                    confidence=confidence,
                    classification=classification,
                    supporting_signals=supporting_signals,
                    contributing_metrics=contributing_metrics,
                    temporal_evidence=temp_ev,
                    dependency_evidence=dep_ev,
                    inference=inference,
                    summary=summary,
                    limitations=limitations,
                )
            )

        top_cand = candidate_explanations[0] if candidate_explanations else None
        overall_root = top_cand.candidate_service if top_cand and top_cand.score > 0 else None
        overall_conf = top_cand.confidence if top_cand else "LOW"

        if overall_root:
            overall_summary = (
                f"Probable root cause identified as '{overall_root}' (Score: {top_cand.score:.3f}, Confidence: {overall_conf}). "
                f"{top_cand.summary}"
            )
        else:
            overall_summary = "No anomalous telemetry observed; no probable root cause identified."

        return RCAIncidentExplanation(
            incident_id=rca_result.incident_id,
            evaluation_window=rca_result.evaluation_window,
            probable_root_cause=overall_root,
            confidence=overall_conf,
            summary=overall_summary,
            candidates=candidate_explanations,
        )


    def _extract_telemetry_evidence(
        self,
        srv_df: pd.DataFrame,
    ) -> tuple[list[TelemetrySignalEvidence], list[str]]:
        """Extract observed factual metric peaks, means, and human-readable signal summaries."""
        metrics: list[TelemetrySignalEvidence] = []
        signals: list[str] = []

        if srv_df.empty:
            return metrics, signals

        for col in self.config.tracked_metrics:
            if col not in srv_df.columns:
                continue

            vals = srv_df[col].dropna()
            if vals.empty:
                continue

            display_name, unit, mult = METRIC_FORMATTERS.get(col, (col.replace("_", " ").title(), "", 1.0))
            peak_val = float(vals.max())
            mean_val = float(vals.mean())
            peak_disp = peak_val * mult
            mean_disp = mean_val * mult

            if col == "connection_utilization" and peak_val >= 0.70:
                desc = f"Observed peak {display_name.lower()} of {peak_disp:.1f}{unit} (mean {mean_disp:.1f}{unit})"
                signals.append(f"Severe connection saturation: peak {peak_disp:.1f}%")
            elif col == "error_rate" and peak_val >= 0.01:
                desc = f"Observed peak {display_name.lower()} of {peak_disp:.2f}{unit} (mean {mean_disp:.2f}{unit})"
                signals.append(f"Elevated error rate: peak {peak_disp:.2f}%")
            elif col == "latency_ms" and peak_val >= 100.0:
                desc = f"Observed peak {display_name.lower()} of {peak_disp:.1f}{unit} (mean {mean_disp:.1f}{unit})"
                signals.append(f"High response latency: peak {peak_disp:.1f}ms")
            elif col == "cpu_usage_pct" and peak_val >= 75.0:
                desc = f"Observed peak {display_name.lower()} of {peak_disp:.1f}{unit} (mean {mean_disp:.1f}{unit})"
                signals.append(f"High CPU utilization: peak {peak_disp:.1f}%")
            elif col == "active_connections" and peak_val > 50:
                desc = f"Observed peak {display_name.lower()} of {peak_disp:.0f}{unit}"
                signals.append(f"High active connections: {peak_disp:.0f}")
            else:
                desc = f"{display_name}: peak {peak_disp:.2f}{unit}, mean {mean_disp:.2f}{unit}"

            metrics.append(
                TelemetrySignalEvidence(
                    metric_name=col,
                    display_name=display_name,
                    peak_value=round(peak_val, 4),
                    mean_value=round(mean_val, 4),
                    unit=unit,
                    description=desc,
                )
            )

        if not signals and not srv_df.empty:
            signals.append("Nominal metric ranges across observed window.")

        return metrics, signals

    def _extract_temporal_evidence(
        self,
        cand: RCACandidate,
        global_earliest_ts: pd.Timestamp | None,
    ) -> TemporalEvidence:
        """Extract factual temporal anomaly timings and relative delta."""
        earliest_str = cand.earliest_anomaly_time
        onset_delta: float | None = None

        if earliest_str and global_earliest_ts is not None:
            cand_ts = pd.to_datetime(earliest_str, utc=True)
            onset_delta = float(round((cand_ts - global_earliest_ts).total_seconds(), 2))

        if cand.anomaly_count == 0 or earliest_str is None:
            summary = "No anomalous samples observed during evaluation window."
        elif onset_delta == 0.0:
            summary = (
                f"Earliest anomaly onset at {earliest_str} (t0 across all services); "
                f"anomalous in {cand.anomaly_count} samples ({cand.anomaly_ratio * 100:.1f}% persistence)."
            )
        else:
            summary = (
                f"Anomaly onset at {earliest_str} (+{onset_delta:.1f}s after primary onset); "
                f"anomalous in {cand.anomaly_count} samples ({cand.anomaly_ratio * 100:.1f}% persistence)."
            )

        return TemporalEvidence(
            earliest_anomaly_timestamp=earliest_str,
            onset_delta_seconds=onset_delta,
            anomaly_samples=cand.anomaly_count,
            anomaly_ratio=cand.anomaly_ratio,
            summary=summary,
        )


    def _extract_dependency_evidence(
        self,
        service: str,
        anomalous_services: set[str],
    ) -> DependencyEvidence:
        """Extract topological relationships, upstream callers, and downstream dependencies."""
        callers = self.reverse_callers.get(service, [])
        deps = self.topology.get(service, [])
        anom_callers = [c for c in callers if c in anomalous_services]
        anom_deps = [d for d in deps if d in anomalous_services]

        depth = 0
        if callers:
            depth = 1 if "api_gateway" in callers else 2
            if "orders_service" in callers:
                depth = 2

        if not deps and callers:
            role = "Leaf downstream dependency"
        elif deps and not callers:
            role = "Top-level ingress caller"
        elif deps and callers:
            role = "Intermediate transit service"
        else:
            role = "Independent isolated service"

        summary_parts = [f"Topology role: {role} (depth {depth})."]
        if anom_callers:
            summary_parts.append(f"Observed propagation to upstream callers: {', '.join(anom_callers)}.")
        if anom_deps:
            summary_parts.append(f"Downstream dependencies exhibiting anomalies: {', '.join(anom_deps)}.")
        if not anom_deps and not anom_callers:
            summary_parts.append("No active propagation along dependency graph edges.")

        return DependencyEvidence(
            topology_depth=depth,
            upstream_callers=callers,
            downstream_dependencies=deps,
            anomalous_callers=anom_callers,
            anomalous_dependencies=anom_deps,
            propagation_role=role,
            summary=" ".join(summary_parts),
        )

    def _evaluate_confidence(
        self,
        cand: RCACandidate,
        all_candidates: list[RCACandidate],
    ) -> tuple[str, str]:
        """Evaluate confidence level and rationale based on score distribution."""
        if cand.score <= 0.0 or cand.anomaly_count == 0:
            return "LOW", "No anomalous telemetry signals observed for this service."

        if cand.rank == 1:
            runner_up_score = all_candidates[1].score if len(all_candidates) > 1 else 0.0
            margin = cand.score - runner_up_score

            if cand.score >= self.config.high_confidence_score or margin >= self.config.high_confidence_margin:
                return "HIGH", f"Decisive candidate score ({cand.score:.3f}) with strong margin (+{margin:.3f}) over runner-up."
            elif cand.score >= self.config.medium_confidence_score:
                return "MEDIUM", f"Moderate candidate score ({cand.score:.3f}) with +{margin:.3f} lead over runner-up."
            else:
                return "LOW", f"Diffused candidate score ({cand.score:.3f}); multiple services exhibit simultaneous degradation."

        return "MEDIUM" if cand.score >= 0.20 else "LOW", f"Rank {cand.rank} candidate score ({cand.score:.3f})."

    def _determine_classification(
        self,
        cand: RCACandidate,
        dep_ev: DependencyEvidence,
    ) -> str:
        """Classify candidate role without asserting definitive proof."""
        if cand.score <= 0.0 or cand.anomaly_count == 0:
            return "Unaffected service"

        if cand.rank == 1:
            return "Probable root cause"

        if dep_ev.anomalous_dependencies:
            return "Cascading failure symptom (downstream dependency failing)"

        if dep_ev.anomalous_callers:
            return "Secondary contributing service"

        return "Correlated anomaly"

    def _build_candidate_summary(
        self,
        cand: RCACandidate,
        classification: str,
        confidence: str,
        supporting_signals: list[str],
        temporal_evidence: TemporalEvidence,
        dependency_evidence: DependencyEvidence,
    ) -> str:
        """Synthesize cohesive engineer-readable summary paragraph."""
        srv = cand.service
        if cand.anomaly_count == 0:
            return f"Service '{srv}' exhibited nominal telemetry behavior with 0 detected anomalies."

        signals_str = "; ".join(supporting_signals[:2])
        if classification == "Probable root cause":
            return (
                f"Probable root cause: '{srv}' ranked #1 (Score: {cand.score:.3f}, Confidence: {confidence}). "
                f"Temporal evidence: {temporal_evidence.summary} "
                f"Dependency evidence: {dependency_evidence.summary} "
                f"Supporting evidence: {signals_str}."
            )
        else:
            return (
                f"Observed propagation: '{srv}' ranked #{cand.rank} as a {classification.lower()} (Score: {cand.score:.3f}). "
                f"{temporal_evidence.summary} "
                f"{dependency_evidence.summary} "
                f"Supporting evidence: {signals_str}."
            )



def explain_rca_result(
    rca_result: RCAResult,
    telemetry_df: pd.DataFrame,
    config: RCAExplainerConfig | None = None,
    topology: Mapping[str, Sequence[str]] | None = None,
) -> RCAIncidentExplanation:
    """Convenience helper to explain an RCAResult with telemetry evidence."""
    explainer = RCAExplainer(config=config, topology=topology)
    return explainer.explain(rca_result, telemetry_df)


def generate_rca_explanation(
    telemetry_df: pd.DataFrame,
    anomalies_df: pd.DataFrame | None = None,
    incident: dict[str, Any] | None = None,
    engine_config: RCAEngineConfig | None = None,
    explainer_config: RCAExplainerConfig | None = None,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> RCAIncidentExplanation:
    """End-to-end convenience pipeline: ranks candidates and produces structured explanation."""
    rca_res = rank_root_causes(
        telemetry_df=telemetry_df,
        anomalies_df=anomalies_df,
        incident=incident,
        config=engine_config,
        start_time=start_time,
        end_time=end_time,
    )
    return explain_rca_result(
        rca_result=rca_res,
        telemetry_df=telemetry_df,
        config=explainer_config,
        topology=engine_config.topology if engine_config else None,
    )

