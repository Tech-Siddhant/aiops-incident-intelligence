"""Lightweight MLOps monitoring and health validation layer.

ponytail: Uses stdlib tracemalloc and time.perf_counter for resource/latency telemetry.
No external daemon (Prometheus, StatsD, Datadog) required.
"""
from __future__ import annotations

import os
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from app.data.synthetic import REQUIRED_COLUMNS
from app.mlops.drift import DataDriftReport, DriftConfig, detect_data_drift


@dataclass
class DataQualityReport:
    """Assessment of telemetry dataset integrity."""
    total_rows: int
    missing_columns: list[str]
    null_counts: dict[str, int]
    infinite_counts: dict[str, int]
    is_valid: bool
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionDistribution:
    """Distribution statistics for model inferences."""
    total_predictions: int
    positive_predictions: int
    positive_rate: float
    score_mean: float
    score_min: float
    score_max: float
    score_p50: float
    score_p90: float
    score_p99: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceUsage:
    """Measured operational runtime overhead."""
    latency_ms: float
    peak_memory_bytes: int
    peak_memory_kb: float
    artifact_size_bytes: int | None = None
    artifact_size_kb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelHealthReport:
    """Consolidated model health, drift, quality, and operational telemetry."""
    model_name: str
    model_version: str
    health_status: str  # "NORMAL", "WARNING", "CRITICAL"
    data_quality: DataQualityReport
    resource_usage: ResourceUsage
    drift_report: DataDriftReport | None = None
    prediction_distribution: PredictionDistribution | None = None
    evaluation_metrics: dict[str, float] = field(default_factory=dict)
    status_reasons: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.drift_report:
            d["drift_report"] = self.drift_report.to_dict()
        return d


def check_data_quality(
    df: pd.DataFrame,
    required_columns: Sequence[str] | None = None,
) -> DataQualityReport:
    """Validate dataset structure, completeness, and value sanity."""
    req_cols = list(required_columns) if required_columns is not None else list(REQUIRED_COLUMNS)
    missing = [c for c in req_cols if c not in df.columns]
    issues: list[str] = []

    if missing:
        issues.append(f"Missing required columns: {missing}")

    null_counts: dict[str, int] = {}
    inf_counts: dict[str, int] = {}

    for col in df.columns:
        null_c = int(df[col].isna().sum())
        if null_c > 0:
            null_counts[col] = null_c
            issues.append(f"Column '{col}' has {null_c} null values.")

        if pd.api.types.is_numeric_dtype(df[col]):
            inf_c = int(np.isinf(df[col].to_numpy(dtype=float, na_value=0.0)).sum())
            if inf_c > 0:
                inf_counts[col] = inf_c
                issues.append(f"Column '{col}' has {inf_c} infinite values.")

    is_valid = len(missing) == 0 and len(null_counts) == 0 and len(inf_counts) == 0

    return DataQualityReport(
        total_rows=len(df),
        missing_columns=missing,
        null_counts=null_counts,
        infinite_counts=inf_counts,
        is_valid=is_valid,
        issues=issues,
    )


def profile_inference(
    model: Any,
    df: pd.DataFrame,
    artifact_path: str | Path | None = None,
) -> tuple[Any, ResourceUsage, PredictionDistribution]:
    """Execute model prediction under memory and latency profiling."""
    tracemalloc.start()
    t0 = time.perf_counter()

    preds = model.predict(df)

    t1 = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latency_ms = round((t1 - t0) * 1000.0, 3)
    peak_kb = round(peak_bytes / 1024.0, 2)

    art_bytes = None
    art_kb = None
    if artifact_path:
        p = Path(artifact_path)
        if p.exists():
            art_bytes = p.stat().st_size
            art_kb = round(art_bytes / 1024.0, 2)

    res_usage = ResourceUsage(
        latency_ms=latency_ms,
        peak_memory_bytes=peak_bytes,
        peak_memory_kb=peak_kb,
        artifact_size_bytes=art_bytes,
        artifact_size_kb=art_kb,
    )

    if isinstance(preds, pd.DataFrame):
        if "is_anomaly" in preds.columns:
            pos_flags = preds["is_anomaly"].to_numpy(dtype=bool)
            scores = preds["anomaly_score"].to_numpy(dtype=float) if "anomaly_score" in preds.columns else pos_flags.astype(float)
        elif "is_incident_predicted" in preds.columns:
            pos_flags = preds["is_incident_predicted"].to_numpy(dtype=bool)
            scores = preds["incident_probability"].to_numpy(dtype=float) if "incident_probability" in preds.columns else pos_flags.astype(float)
        else:
            pos_flags = np.zeros(len(preds), dtype=bool)
            scores = np.zeros(len(preds), dtype=float)
    elif isinstance(preds, np.ndarray):
        if preds.dtype == bool:
            pos_flags = preds
            scores = preds.astype(float)
        else:
            scores = preds.astype(float)
            pos_flags = scores > 0.5
    else:
        pos_flags = np.zeros(len(df), dtype=bool)
        scores = np.zeros(len(df), dtype=float)

    n = len(pos_flags)
    pos_count = int(np.sum(pos_flags))
    pos_rate = round(pos_count / n, 4) if n > 0 else 0.0

    pred_dist = PredictionDistribution(
        total_predictions=n,
        positive_predictions=pos_count,
        positive_rate=pos_rate,
        score_mean=round(float(np.mean(scores)), 4) if n > 0 else 0.0,
        score_min=round(float(np.min(scores)), 4) if n > 0 else 0.0,
        score_max=round(float(np.max(scores)), 4) if n > 0 else 0.0,
        score_p50=round(float(np.percentile(scores, 50)), 4) if n > 0 else 0.0,
        score_p90=round(float(np.percentile(scores, 90)), 4) if n > 0 else 0.0,
        score_p99=round(float(np.percentile(scores, 99)), 4) if n > 0 else 0.0,
    )
    return preds, res_usage, pred_dist



def evaluate_model_health(
    model: Any,
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
    model_name: str = "isolation_forest",
    model_version: str = "1.0.0",
    evaluation_metrics: dict[str, float] | None = None,
    artifact_path: str | Path | None = None,
    required_columns: Sequence[str] | None = None,
    latency_sla_ms: float = 500.0,
    drift_config: DriftConfig | None = None,
) -> ModelHealthReport:
    """Perform holistic model health, data quality, drift, and performance monitoring."""
    status_reasons: list[str] = []
    status = "NORMAL"

    dq = check_data_quality(current_df, required_columns=required_columns)
    if not dq.is_valid:
        status = "CRITICAL"
        status_reasons.append(f"Data quality check failed with {len(dq.issues)} issue(s).")

    preds, res_usage, pred_dist = profile_inference(model, current_df, artifact_path=artifact_path)

    if res_usage.latency_ms > (2 * latency_sla_ms):
        status = "CRITICAL"
        status_reasons.append(f"Inference latency ({res_usage.latency_ms}ms) breached 2x SLA limit ({latency_sla_ms}ms).")
    elif res_usage.latency_ms > latency_sla_ms:
        if status != "CRITICAL":
            status = "WARNING"
        status_reasons.append(f"Inference latency ({res_usage.latency_ms}ms) exceeded SLA ({latency_sla_ms}ms).")

    drift_report = None
    if reference_df is not None:
        drift_report = detect_data_drift(reference_df, current_df, config=drift_config)
        if drift_report.drift_share >= 0.3:
            status = "CRITICAL"
            status_reasons.append(f"Critical drift: {len(drift_report.drifted_features)} features drifted ({drift_report.drift_share*100:.1f}%).")
        elif drift_report.has_drift or len(drift_report.warning_features) > 0:
            if status != "CRITICAL":
                status = "WARNING"
            status_reasons.append(f"Drift alert: {len(drift_report.drifted_features)} drifted, {len(drift_report.warning_features)} warning features.")

    if not status_reasons:
        status_reasons.append("All operational, data quality, and drift checks healthy.")

    return ModelHealthReport(
        model_name=model_name,
        model_version=model_version,
        health_status=status,
        data_quality=dq,
        resource_usage=res_usage,
        drift_report=drift_report,
        prediction_distribution=pred_dist,
        evaluation_metrics=evaluation_metrics or {},
        status_reasons=status_reasons,
    )
