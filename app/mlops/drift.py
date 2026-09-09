"""Lightweight data and model drift detection.

ponytail: Vectorized numpy/scipy operations per feature. Fast and sufficient
for in-memory DataFrames without requiring streaming workers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DriftConfig:
    """Configurable thresholds for data drift detection."""
    psi_warning_threshold: float = 0.1
    psi_drift_threshold: float = 0.2
    ks_p_value_threshold: float = 0.05
    ks_stat_warning_threshold: float = 0.1
    ks_stat_drift_threshold: float = 0.2
    num_bins: int = 10
    epsilon: float = 1e-4


@dataclass
class FeatureDriftResult:
    """Drift evaluation result for a single numeric feature."""
    feature_name: str
    drift_status: str  # "NO_DRIFT", "WARNING", "DRIFT", "MISSING", "INSUFFICIENT_DATA"
    psi: float | None = None
    ks_statistic: float | None = None
    ks_p_value: float | None = None
    is_drift: bool = False
    is_warning: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataDriftReport:
    """Summary report of feature drift across reference and current datasets."""
    reference_count: int
    current_count: int
    drifted_features: list[str]
    warning_features: list[str]
    feature_results: dict[str, FeatureDriftResult]
    has_drift: bool
    drift_share: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["feature_results"] = {
            k: v.to_dict() if isinstance(v, FeatureDriftResult) else v
            for k, v in self.feature_results.items()
        }
        return d


@dataclass
class PerformanceDegradationResult:
    """Evaluation result comparing baseline model metrics with new ground-truth metrics."""
    baseline_metrics: dict[str, float]
    current_metrics: dict[str, float]
    degraded_metrics: dict[str, float]
    is_degraded: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def calculate_psi(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between two distributions."""
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    curr = curr[np.isfinite(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0

    if np.min(ref) == np.max(ref):
        return 0.0 if np.all(curr == ref[0]) else 1.0

    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.unique(np.percentile(ref, quantiles))
    if len(bins) < 2:
        return 0.0 if np.all(curr == ref[0]) else 1.0

    bins[0] = -np.inf
    bins[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bins)
    curr_counts, _ = np.histogram(curr, bins=bins)

    ref_probs = np.maximum(ref_counts / len(ref), epsilon)
    curr_probs = np.maximum(curr_counts / len(curr), epsilon)
    ref_probs = ref_probs / np.sum(ref_probs)
    curr_probs = curr_probs / np.sum(curr_probs)

    return float(np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs)))


def evaluate_feature_drift(
    ref_series: pd.Series | np.ndarray,
    curr_series: pd.Series | np.ndarray,
    feature_name: str,
    config: DriftConfig | None = None,
) -> FeatureDriftResult:
    """Evaluate drift for a single feature using PSI and KS-test."""
    cfg = config or DriftConfig()
    ref = np.asarray(ref_series, dtype=float)
    curr = np.asarray(curr_series, dtype=float)

    ref_clean = ref[np.isfinite(ref)]
    curr_clean = curr[np.isfinite(curr)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return FeatureDriftResult(
            feature_name=feature_name,
            drift_status="INSUFFICIENT_DATA",
            message="Feature contains no finite values.",
        )

    if np.min(ref_clean) == np.max(ref_clean) and np.min(curr_clean) == np.max(curr_clean):
        if ref_clean[0] == curr_clean[0]:
            return FeatureDriftResult(
                feature_name=feature_name,
                drift_status="NO_DRIFT",
                psi=0.0,
                ks_statistic=0.0,
                ks_p_value=1.0,
                message="Constant identical values across datasets.",
            )
        return FeatureDriftResult(
            feature_name=feature_name,
            drift_status="DRIFT",
            psi=1.0,
            ks_statistic=1.0,
            ks_p_value=0.0,
            is_drift=True,
            message="Constant value shifted completely.",
        )

    psi_val = calculate_psi(ref_clean, curr_clean, num_bins=cfg.num_bins, epsilon=cfg.epsilon)
    ks_res = stats.ks_2samp(ref_clean, curr_clean)
    ks_stat = float(ks_res.statistic)
    ks_pval = float(ks_res.pvalue)

    is_significant = (
        psi_val >= cfg.psi_drift_threshold
        or (ks_pval < cfg.ks_p_value_threshold and ks_stat >= cfg.ks_stat_drift_threshold)
    )
    is_warning = (
        not is_significant
        and (
            psi_val >= cfg.psi_warning_threshold
            or (ks_pval < cfg.ks_p_value_threshold and ks_stat >= cfg.ks_stat_warning_threshold)
        )
    )

    if is_significant:
        status = "DRIFT"
        msg = f"Significant drift (PSI={psi_val:.4f}, KS={ks_stat:.4f}, p={ks_pval:.4e})"
    elif is_warning:
        status = "WARNING"
        msg = f"Moderate drift / warning (PSI={psi_val:.4f}, KS={ks_stat:.4f}, p={ks_pval:.4e})"
    else:
        status = "NO_DRIFT"
        msg = f"Distribution stable (PSI={psi_val:.4f}, KS={ks_stat:.4f}, p={ks_pval:.4e})"

    return FeatureDriftResult(
        feature_name=feature_name,
        drift_status=status,
        psi=round(psi_val, 4),
        ks_statistic=round(ks_stat, 4),
        ks_p_value=round(ks_pval, 4),
        is_drift=is_significant,
        is_warning=is_warning,
        message=msg,
    )

    is_degraded: bool
    details: dict[str, Any]

def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str] | None = None,
    config: DriftConfig | None = None,
) -> DataDriftReport:
    """Compare reference and current DataFrames across numerical features."""
    cfg = config or DriftConfig()

    if features is None:
        ref_num = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        curr_num = current_df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_check = sorted(set(ref_num).union(curr_num))
    else:
        cols_to_check = features

    results: dict[str, FeatureDriftResult] = {}
    drifted: list[str] = []
    warnings: list[str] = []

    for col in cols_to_check:
        if col not in reference_df.columns or col not in current_df.columns:
            results[col] = FeatureDriftResult(
                feature_name=col,
                drift_status="MISSING",
                is_drift=True,
                message=f"Feature '{col}' missing from {'reference' if col not in reference_df.columns else 'current'} dataset.",
            )
            drifted.append(col)
            continue

        res = evaluate_feature_drift(
            reference_df[col],
            current_df[col],
            feature_name=col,
            config=cfg,
        )
        results[col] = res
        if res.is_drift:
            drifted.append(col)
        elif res.is_warning:
            warnings.append(col)

    total_checked = len(cols_to_check)
    drift_share = (len(drifted) / total_checked) if total_checked > 0 else 0.0

    return DataDriftReport(
        reference_count=len(reference_df),
        current_count=len(current_df),
        drifted_features=drifted,
        warning_features=warnings,
        feature_results=results,
        has_drift=len(drifted) > 0,
        drift_share=round(drift_share, 4),
    )


def detect_performance_degradation(
    baseline_metrics: dict[str, float],
    current_metrics: dict[str, float],
    max_allowed_drop: float = 0.05,
    relative: bool = True,
) -> PerformanceDegradationResult:
    """Identify model metric degradations against established baselines."""
    degraded = {}
    details = {}

    for metric, base_val in baseline_metrics.items():
        if metric not in current_metrics:
            continue
        curr_val = current_metrics[metric]

        if relative:
            drop = (base_val - curr_val) / max(abs(base_val), 1e-6)
        else:
            drop = base_val - curr_val

        details[metric] = {
            "baseline": base_val,
            "current": curr_val,
            "drop": round(drop, 4),
            "threshold": max_allowed_drop,
        }

        if drop > max_allowed_drop:
            degraded[metric] = round(drop, 4)

    return PerformanceDegradationResult(
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        degraded_metrics=degraded,
        is_degraded=len(degraded) > 0,
        details=details,
    )


    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
