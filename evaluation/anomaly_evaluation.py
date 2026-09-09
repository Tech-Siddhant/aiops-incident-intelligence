"""Anomaly detection evaluation and benchmarking suite."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from app.data.preprocess import get_incident_labels
from app.models.anomaly_baseline import BaselineConfig, detect_anomalies
from app.models.anomaly_isolation_forest import (
    IsolationForestConfig,
    IsolationForestDetector,
)


@dataclass(frozen=True)
class AnomalyMetrics:
    """Evaluation metrics for anomaly detection."""
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    detection_delay_seconds: float | None
    total_samples: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_metrics(
    predictions_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    incidents: list[dict[str, Any]] | None = None,
) -> AnomalyMetrics:
    """Calculate precision, recall, F1, FPR, and detection delay against ground truth.

    Args:
        predictions_df: DataFrame with ['timestamp', 'service', 'is_anomaly'].
        labels_df: DataFrame with ['is_incident'].
        incidents: Optional list of incident metadata dicts for detection delay calculation.

    Returns:
        AnomalyMetrics dataclass.
    """
    if len(predictions_df) != len(labels_df):
        raise ValueError(
            f"Row count mismatch between predictions ({len(predictions_df)}) and labels ({len(labels_df)})."
        )

    y_pred = predictions_df["is_anomaly"].astype(bool).to_numpy()
    y_true = labels_df["is_incident"].astype(bool).to_numpy()

    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    tn = int(np.sum(~y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))

    total = len(y_true)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    delays: list[float] = []
    if incidents:
        ts = pd.to_datetime(predictions_df["timestamp"], utc=True)
        services = predictions_df["service"].astype(str)

        for inc in incidents:
            start_ts = pd.to_datetime(inc["start_time"], utc=True)
            end_ts = pd.to_datetime(inc["end_time"], utc=True)
            root_svc = inc.get("root_cause_service")
            affected = set(inc.get("affected_services", [root_svc]))

            # Prioritize root cause service detection, fallback to any affected service
            svc_mask = services == root_svc if root_svc else services.isin(affected)
            mask = (ts >= start_ts) & (ts <= end_ts) & svc_mask & y_pred

            matched_ts = ts[mask]
            if not matched_ts.empty:
                earliest_det = matched_ts.min()
                delay = (earliest_det - start_ts).total_seconds()
                delays.append(max(0.0, float(delay)))

    mean_delay = float(np.mean(delays)) if delays else None

    return AnomalyMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        false_positive_rate=round(fpr, 4),
        detection_delay_seconds=round(mean_delay, 2) if mean_delay is not None else None,
        total_samples=total,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def run_anomaly_benchmark(
    telemetry_df: pd.DataFrame,
    incidents: list[dict[str, Any]],
    baseline_config: BaselineConfig | None = None,
    iforest_config: IsolationForestConfig | None = None,
    train_df: pd.DataFrame | None = None,
) -> dict[str, AnomalyMetrics]:
    """Execute comparative benchmark between Statistical Baseline and Isolation Forest."""
    labels_df = get_incident_labels(telemetry_df, incidents)

    # 1. Statistical Baseline
    base_preds = detect_anomalies(telemetry_df, config=baseline_config)
    base_metrics = calculate_metrics(base_preds, labels_df, incidents=incidents)

    # 2. Isolation Forest
    iforest = IsolationForestDetector(config=iforest_config)
    if train_df is not None:
        iforest.fit(train_df)
        iforest_preds = iforest.predict(telemetry_df)
    else:
        # Default: train on pre-incident baseline data if single incident
        if incidents:
            earliest_inc = min(pd.to_datetime(inc["start_time"], utc=True) for inc in incidents)
            fit_data = telemetry_df[pd.to_datetime(telemetry_df["timestamp"], utc=True) < earliest_inc]
            if len(fit_data) >= 10:
                iforest.fit(fit_data)
                iforest_preds = iforest.predict(telemetry_df)
            else:
                iforest_preds = iforest.fit_predict(telemetry_df)
        else:
            iforest_preds = iforest.fit_predict(telemetry_df)

    iforest_metrics = calculate_metrics(iforest_preds, labels_df, incidents=incidents)

    return {
        "statistical_baseline": base_metrics,
        "isolation_forest": iforest_metrics,
    }
