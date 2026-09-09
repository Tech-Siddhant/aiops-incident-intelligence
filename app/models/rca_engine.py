"""Lightweight deterministic Root Cause Analysis (RCA) candidate generation and ranking engine.

Disclaimer:
    RCA candidate ranking is a heuristic ranking problem based on temporal anomaly
    sequences, dependency graph topology, and telemetry severity. It provides probable
    candidate rankings, NOT definitive causal proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

from app.models.anomaly_isolation_forest import detect_anomalies_iforest

# Microservice topology (caller -> dependencies)
DEFAULT_TOPOLOGY: dict[str, list[str]] = {
    "api_gateway": ["auth_service", "orders_service"],
    "orders_service": ["database"],
    "auth_service": [],
    "database": [],
}


@dataclass(frozen=True)
class RCAEngineConfig:
    """Configuration for Root Cause Analysis candidate ranking engine."""
    topology: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_TOPOLOGY.items()})
    temporal_weight: float = 0.40
    topology_weight: float = 0.35
    severity_weight: float = 0.25
    epsilon: float = 1e-6


@dataclass(frozen=True)
class RCACandidate:
    """Ranked root-cause candidate with score breakdown and explanation."""
    service: str
    score: float
    raw_score: float
    rank: int
    earliest_anomaly_time: str | None
    anomaly_count: int
    anomaly_ratio: float
    mean_anomaly_score: float
    max_anomaly_score: float
    score_breakdown: dict[str, float]
    explanation: str
    is_root_cause_candidate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RCAResult:
    """Container for RCA candidate ranking results."""
    ranked_candidates: list[RCACandidate]
    incident_id: str | None = None
    evaluation_window: tuple[str | None, str | None] = (None, None)
    disclaimer: str = (
        "RCA ranking provides heuristic candidate likelihoods based on temporal ordering, "
        "topology, and metric deviations; it does not constitute deterministic causal proof."
    )

    @property
    def top_1(self) -> RCACandidate | None:
        """Return the highest ranked candidate, or None if no candidates exist."""
        return self.ranked_candidates[0] if self.ranked_candidates else None

    @property
    def top_3(self) -> list[RCACandidate]:
        """Return up to the top 3 ranked candidates."""
        return self.ranked_candidates[:3]

    def get_top_k(self, k: int) -> list[RCACandidate]:
        """Return up to top-k candidates."""
        return self.ranked_candidates[:max(0, k)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "evaluation_window": list(self.evaluation_window),
            "disclaimer": self.disclaimer,
            "top_1": self.top_1.to_dict() if self.top_1 else None,
            "top_3": [c.to_dict() for c in self.top_3],
            "ranked_candidates": [c.to_dict() for c in self.ranked_candidates],
        }


def _compute_topology_metadata(
    topology: Mapping[str, Sequence[str]],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Compute reverse caller mapping and depth from root entrypoints for a DAG topology.

    # ponytail: O(V+E) graph traversal; upgrade to cycle-safe topological sort if cycles added.
    """
    callers: dict[str, list[str]] = {s: [] for s in topology}
    for caller, deps in topology.items():
        for dep in deps:
            if dep not in callers:
                callers[dep] = []
            callers[dep].append(caller)

    # Roots are services with no incoming callers
    roots = [s for s in topology if not callers.get(s)]
    if not roots:
        roots = list(topology.keys())

    depths: dict[str, int] = {s: 0 for s in topology}

    def _dfs_depth(node: str, current_depth: int) -> None:
        if current_depth > depths.get(node, 0):
            depths[node] = current_depth
        for child in topology.get(node, []):
            _dfs_depth(child, current_depth + 1)

    for root in roots:
        _dfs_depth(root, 0)

    return callers, depths


class RCAEngine:
    """Deterministic lightweight Root Cause Analysis ranking engine."""

    def __init__(self, config: RCAEngineConfig | None = None) -> None:
        self.config = config or RCAEngineConfig()
        self.callers_, self.depths_ = _compute_topology_metadata(self.config.topology)

    def rank(
        self,
        telemetry_df: pd.DataFrame,
        anomalies_df: pd.DataFrame | None = None,
        incident: dict[str, Any] | None = None,
        start_time: str | pd.Timestamp | None = None,
        end_time: str | pd.Timestamp | None = None,
    ) -> RCAResult:
        """Generate and rank root-cause candidate services.

        Args:
            telemetry_df: Microservice telemetry DataFrame.
            anomalies_df: Optional anomaly detection output DataFrame with
                          ['timestamp', 'service', 'is_anomaly', 'anomaly_score'].
                          If None and telemetry_df lacks anomaly flags, Isolation Forest is run.
            incident: Optional incident metadata dictionary (e.g. from incidents.json).
            start_time: Optional explicit evaluation window start.
            end_time: Optional explicit evaluation window end.

        Returns:
            RCAResult containing ranked candidates, top-1, top-3, and score breakdowns.
        """
        if telemetry_df.empty:
            return RCAResult(ranked_candidates=[], incident_id=incident.get("incident_id") if incident else None)

        # 1. Resolve evaluation time window
        win_start: pd.Timestamp | None = None
        win_end: pd.Timestamp | None = None
        inc_id: str | None = None

        if incident:
            inc_id = incident.get("incident_id")
            if "start_time" in incident:
                win_start = pd.to_datetime(incident["start_time"], utc=True)
            if "end_time" in incident:
                win_end = pd.to_datetime(incident["end_time"], utc=True)

        if start_time is not None:
            win_start = pd.to_datetime(start_time, utc=True)
        if end_time is not None:
            win_end = pd.to_datetime(end_time, utc=True)

        # 2. Merge telemetry with anomaly signals
        work_df = telemetry_df.copy()
        work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)

        if anomalies_df is not None and not anomalies_df.empty:
            anom_copy = anomalies_df.copy()
            anom_copy["timestamp"] = pd.to_datetime(anom_copy["timestamp"], utc=True)
            merge_cols = [c for c in anom_copy.columns if c not in ("timestamp", "service") and c in work_df.columns]
            if merge_cols:
                work_df = work_df.drop(columns=merge_cols)
            work_df = work_df.merge(anom_copy, on=["timestamp", "service"], how="left")
        elif "is_anomaly" not in work_df.columns:
            anom_res = detect_anomalies_iforest(work_df)
            work_df = work_df.merge(anom_res, on=["timestamp", "service"], how="left")

        # Fill default anomaly columns if missing
        if "is_anomaly" not in work_df.columns:
            work_df["is_anomaly"] = False
        else:
            work_df["is_anomaly"] = work_df["is_anomaly"].fillna(False).astype(bool)

        if "anomaly_score" not in work_df.columns:
            work_df["anomaly_score"] = 0.0
        else:
            work_df["anomaly_score"] = work_df["anomaly_score"].fillna(0.0).astype(float)

        # 3. Filter data to evaluation window
        if win_start is not None:
            work_df = work_df[work_df["timestamp"] >= win_start]
        if win_end is not None:
            work_df = work_df[work_df["timestamp"] <= win_end]

        if work_df.empty:
            return RCAResult(
                ranked_candidates=[],
                incident_id=inc_id,
                evaluation_window=(
                    win_start.strftime("%Y-%m-%dT%H:%M:%SZ") if win_start else None,
                    win_end.strftime("%Y-%m-%dT%H:%M:%SZ") if win_end else None,
                ),
            )
        # 4. Extract per-service candidate statistics
        known_services = list(dict.fromkeys(list(self.config.topology.keys()) + work_df["service"].unique().tolist()))
        service_stats: dict[str, dict[str, Any]] = {}

        for srv in known_services:
            srv_df = work_df[work_df["service"] == srv]
            total_samples = len(srv_df)
            if total_samples == 0:
                service_stats[srv] = {
                    "total_samples": 0,
                    "anomaly_count": 0,
                    "anomaly_ratio": 0.0,
                    "earliest_ts": None,
                    "mean_anomaly_score": 0.0,
                    "max_anomaly_score": 0.0,
                    "max_conn_util": 0.0,
                    "max_error_rate": 0.0,
                }
                continue

            anom_df = srv_df[srv_df["is_anomaly"]]
            anom_count = len(anom_df)
            earliest_ts = anom_df["timestamp"].min() if not anom_df.empty else None
            mean_anom = float(srv_df["anomaly_score"].mean()) if "anomaly_score" in srv_df else 0.0
            max_anom = float(srv_df["anomaly_score"].max()) if "anomaly_score" in srv_df else 0.0
            max_conn = float(srv_df["connection_utilization"].max()) if "connection_utilization" in srv_df else 0.0
            max_err = float(srv_df["error_rate"].max()) if "error_rate" in srv_df else 0.0

            service_stats[srv] = {
                "total_samples": total_samples,
                "anomaly_count": anom_count,
                "anomaly_ratio": float(anom_count / total_samples) if total_samples > 0 else 0.0,
                "earliest_ts": earliest_ts,
                "mean_anomaly_score": mean_anom,
                "max_anomaly_score": max_anom,
                "max_conn_util": max_conn,
                "max_error_rate": max_err,
            }

        # 5. Temporal Scoring (earlier onset -> higher score)
        valid_onsets = [st["earliest_ts"] for st in service_stats.values() if st["earliest_ts"] is not None]
        t_min = min(valid_onsets) if valid_onsets else None
        t_max = max(valid_onsets) if valid_onsets else None
        dt_span = (t_max - t_min).total_seconds() if (t_min and t_max and t_max > t_min) else 0.0

        max_depth = max(self.depths_.values()) if self.depths_ else 1
        all_max_scores = [st["max_anomaly_score"] for st in service_stats.values()]
        global_max_anom = max(all_max_scores) if all_max_scores else 0.0
        global_min_anom = min(all_max_scores) if all_max_scores else 0.0
        anom_span = global_max_anom - global_min_anom

        cfg = self.config
        unnormalized_candidates: list[dict[str, Any]] = []

        for srv in known_services:
            st = service_stats[srv]
            has_anomaly = st["anomaly_count"] > 0 and st["earliest_ts"] is not None

            if not has_anomaly:
                s_temp = 0.0
                s_topo = 0.0
                s_sev = 0.0
                s_raw = 0.0
            else:
                # 5a. Temporal score: onset priority + anomaly persistence
                if dt_span > 0:
                    dt_delay = (st["earliest_ts"] - t_min).total_seconds()
                    s_onset = 1.0 - (dt_delay / dt_span)
                else:
                    s_onset = 1.0
                s_temp = float(np.clip(0.70 * s_onset + 0.30 * st["anomaly_ratio"], 0.0, 1.0))

                # 5b. Topology score: depth + caller propagation - dependency dampening
                srv_deps = self.config.topology.get(srv, [])
                srv_callers = self.callers_.get(srv, [])

                dep_anom_cnt = sum(1 for d in srv_deps if service_stats.get(d, {}).get("anomaly_count", 0) > 0)
                dep_ratio = dep_anom_cnt / len(srv_deps) if srv_deps else 0.0

                caller_anom_cnt = sum(1 for c in srv_callers if service_stats.get(c, {}).get("anomaly_count", 0) > 0)
                caller_ratio = caller_anom_cnt / len(srv_callers) if srv_callers else 0.0

                depth_ratio = self.depths_.get(srv, 0) / max_depth if max_depth > 0 else 0.0
                s_topo_base = 0.40 * depth_ratio + 0.40 * caller_ratio + 0.20 * (1.0 - dep_ratio)
                s_topo = float(np.clip(s_topo_base * (1.0 - 0.50 * dep_ratio), 0.0, 1.0))

                # 5c. Severity score: anomaly intensity + metric saturation
                s_anom = (st["max_anomaly_score"] - global_min_anom) / (anom_span + cfg.epsilon) if anom_span > 0 else 1.0
                s_sat = max(st["max_conn_util"], st["max_error_rate"])
                s_sev = float(np.clip(0.60 * s_anom + 0.40 * s_sat, 0.0, 1.0))

                # Composite weighted raw score
                s_raw = float(
                    cfg.temporal_weight * s_temp
                    + cfg.topology_weight * s_topo
                    + cfg.severity_weight * s_sev
                )

            unnormalized_candidates.append({
                "service": srv,
                "raw_score": s_raw,
                "s_temp": s_temp,
                "s_topo": s_topo,
                "s_sev": s_sev,
                "stats": st,
            })

        # 6. Normalize candidate scores
        total_raw = sum(c["raw_score"] for c in unnormalized_candidates)
        candidates: list[RCACandidate] = []

        # Deterministic sorting: raw_score desc, earliest_ts asc, service asc
        def _sort_key(c: dict[str, Any]) -> tuple[float, pd.Timestamp, str]:
            e_ts = c["stats"]["earliest_ts"]
            ts_key = e_ts if e_ts is not None else pd.Timestamp.max.tz_localize("UTC")
            return (-round(c["raw_score"], 6), ts_key, c["service"])

        sorted_raw = sorted(unnormalized_candidates, key=_sort_key)

        for rank_idx, c in enumerate(sorted_raw, start=1):
            srv = c["service"]
            st = c["stats"]
            norm_score = float(round(c["raw_score"] / total_raw, 4)) if total_raw > 0 else 0.0
            earliest_str = st["earliest_ts"].strftime("%Y-%m-%dT%H:%M:%SZ") if st["earliest_ts"] else None

            # Explainable rationale breakdown
            s_temp, s_topo, s_sev = c["s_temp"], c["s_topo"], c["s_sev"]
            explanation = (
                f"Candidate '{srv}' (Rank {rank_idx}, score {norm_score:.3f}): "
                f"temporal={s_temp:.2f} (onset: {earliest_str or 'None'}), "
                f"topology={s_topo:.2f} (depth: {self.depths_.get(srv, 0)}), "
                f"severity={s_sev:.2f} (max_score: {st['max_anomaly_score']:.3f})."
            )

            candidate = RCACandidate(
                service=srv,
                score=norm_score,
                raw_score=round(c["raw_score"], 4),
                rank=rank_idx,
                earliest_anomaly_time=earliest_str,
                anomaly_count=st["anomaly_count"],
                anomaly_ratio=round(st["anomaly_ratio"], 4),
                mean_anomaly_score=round(st["mean_anomaly_score"], 4),
                max_anomaly_score=round(st["max_anomaly_score"], 4),
                score_breakdown={
                    "temporal": round(s_temp, 4),
                    "topology": round(s_topo, 4),
                    "severity": round(s_sev, 4),
                },
                explanation=explanation,
                is_root_cause_candidate=(c["raw_score"] > 0),
            )
            candidates.append(candidate)

        return RCAResult(
            ranked_candidates=candidates,
            incident_id=inc_id,
            evaluation_window=(
                win_start.strftime("%Y-%m-%dT%H:%M:%SZ") if win_start else None,
                win_end.strftime("%Y-%m-%dT%H:%M:%SZ") if win_end else None,
            ),
        )


def rank_root_causes(
    telemetry_df: pd.DataFrame,
    anomalies_df: pd.DataFrame | None = None,
    incident: dict[str, Any] | None = None,
    config: RCAEngineConfig | None = None,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> RCAResult:
    """Convenience function to evaluate and rank root-cause candidates."""
    engine = RCAEngine(config=config)
    return engine.rank(
        telemetry_df,
        anomalies_df=anomalies_df,
        incident=incident,
        start_time=start_time,
        end_time=end_time,
    )


