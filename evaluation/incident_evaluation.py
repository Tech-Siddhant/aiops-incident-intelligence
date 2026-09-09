"""Unified evaluation and failure-analysis suite for Phase 4 (Prediction & Severity)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.data.features import extract_features
from app.data.preprocess import preprocess_telemetry
from app.models.incident_predictor import (
    IncidentPredictionConfig,
    IncidentPredictorBaseline,
    PredictionMetrics,
    chronological_train_val_test_split,
    create_horizon_labels,
)
from app.models.severity_classifier import (
    SeverityClassificationConfig,
    SeverityClassifierBaseline,
    SeverityMetrics,
    calculate_severity_metrics,
    extract_severity_labels,
)


@dataclass(frozen=True)
class IncidentEvaluationResult:
    """Unified evaluation metrics across incident prediction and severity classification."""
    prediction_train: PredictionMetrics
    prediction_val: PredictionMetrics
    prediction_test: PredictionMetrics
    prediction_lead_time_seconds: float | None
    prediction_false_positives: int
    prediction_false_negatives: int
    severity_metrics: SeverityMetrics
    top_predictive_features: list[tuple[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_prediction_lead_time(
    predictions_df: pd.DataFrame,
    incidents: list[dict[str, Any]],
    horizon_seconds: int = 1800,
) -> float | None:
    """Calculate mean lead time (seconds) between earliest pre-incident prediction and start time.

    Only considers true alerts that triggered before the actual incident start ($t < start$).
    """
    if predictions_df.empty or not incidents:
        return None

    ts = pd.to_datetime(predictions_df["timestamp"], utc=True)
    services = predictions_df["service"].astype(str) if "service" in predictions_df.columns else pd.Series(["all"] * len(predictions_df))
    flags = predictions_df["is_predicted_incident"].astype(bool).to_numpy()
    horizon_td = pd.Timedelta(seconds=horizon_seconds)

    lead_times: list[float] = []

    for inc in incidents:
        start_ts = pd.to_datetime(inc["start_time"], utc=True)
        root_svc = inc.get("root_cause_service")
        affected = set(inc.get("affected_services", [root_svc] if root_svc else []))

        # Look in the pre-incident horizon window [start - horizon, start)
        svc_mask = services == root_svc if root_svc else (services.isin(affected) if affected else True)
        time_mask = (ts >= (start_ts - horizon_td)) & (ts < start_ts)
        pre_incident_mask = time_mask & svc_mask & flags

        matched = ts[pre_incident_mask]
        if not matched.empty:
            earliest_alert = matched.min()
            lead_time = (start_ts - earliest_alert).total_seconds()
            lead_times.append(max(0.0, float(lead_time)))

    return float(np.mean(lead_times)) if lead_times else None


def run_incident_evaluation(
    telemetry_df: pd.DataFrame,
    incidents: list[dict[str, Any]],
    prediction_config: IncidentPredictionConfig | None = None,
    severity_config: SeverityClassificationConfig | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> IncidentEvaluationResult:
    """Run end-to-end chronological evaluation for Incident Prediction & Severity Classification."""
    pred_cfg = prediction_config or IncidentPredictionConfig()
    sev_cfg = severity_config or SeverityClassificationConfig()

    # Preprocess and feature engineering if not already present
    working_df = telemetry_df.copy()
    if "latency_ms_rolling_mean_60s" not in working_df.columns:
        clean_df, _ = preprocess_telemetry(working_df)
        working_df = extract_features(clean_df)

    # 1. INCIDENT PREDICTION EVALUATION
    target_labels = create_horizon_labels(
        working_df,
        incidents,
        horizon_seconds=pred_cfg.horizon_seconds,
        service_specific=pred_cfg.service_specific,
        lead_time_only=pred_cfg.lead_time_only,
    )

    train_ratio = max(0.0, 1.0 - (val_ratio + test_ratio))
    train_df, val_df, test_df, y_train, y_val, y_test = chronological_train_val_test_split(
        working_df,
        target_labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    predictor = IncidentPredictorBaseline(pred_cfg)
    predictor.fit(train_df, y_train)

    train_metrics = predictor.evaluate(train_df, y_train)
    val_metrics = predictor.evaluate(val_df, y_val)
    test_metrics = predictor.evaluate(test_df, y_test)

    # Prediction diagnostics on test split
    test_preds = predictor.predict(test_df)
    y_test_arr = np.asarray(y_test, dtype=bool)
    test_fp = int(np.sum(test_preds & ~y_test_arr))
    test_fn = int(np.sum(~test_preds & y_test_arr))

    # Lead time across full dataset predictions
    full_pred_df = predictor.predict_dataframe(working_df)
    lead_time = calculate_prediction_lead_time(
        full_pred_df, incidents, horizon_seconds=pred_cfg.horizon_seconds
    )

    # Feature importance (logistic regression weights)
    top_features: list[tuple[str, float]] = []
    if hasattr(predictor.model_, "coef_") and predictor.model_.coef_ is not None and len(predictor.model_.coef_) > 0:
        coefs = predictor.model_.coef_[0]
        feat_pairs = list(zip(predictor.feature_cols_, [float(c) for c in coefs]))
        top_features = sorted(feat_pairs, key=lambda x: abs(x[1]), reverse=True)[:10]

    # 2. SEVERITY CLASSIFICATION EVALUATION
    sev_labels = extract_severity_labels(working_df, incidents)
    incident_mask = sev_labels.notna()

    if incident_mask.sum() > 0:
        sev_feats = working_df[incident_mask].reset_index(drop=True)
        sev_targets = sev_labels[incident_mask].reset_index(drop=True)

        # Chronological split on incident occurrences
        n_sev = len(sev_feats)
        n_train = max(1, int(n_sev * (1.0 - test_ratio)))
        sev_train_X, sev_test_X = sev_feats.iloc[:n_train], sev_feats.iloc[n_train:]
        sev_train_y, sev_test_y = sev_targets.iloc[:n_train], sev_targets.iloc[n_train:]

        sev_clf = SeverityClassifierBaseline(sev_cfg)
        sev_clf.fit(sev_train_X, sev_train_y)
        sev_metrics = (
            sev_clf.evaluate(sev_test_X, sev_test_y)
            if len(sev_test_X) > 0
            else sev_clf.evaluate(sev_train_X, sev_train_y)
        )
    else:
        sev_metrics = calculate_severity_metrics([], [])

    return IncidentEvaluationResult(
        prediction_train=train_metrics,
        prediction_val=val_metrics,
        prediction_test=test_metrics,
        prediction_lead_time_seconds=lead_time,
        prediction_false_positives=test_fp,
        prediction_false_negatives=test_fn,
        severity_metrics=sev_metrics,
        top_predictive_features=top_features,
    )
