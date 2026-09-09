"""Time-series feature engineering pipeline for telemetry data.

Feature Classification:
1. Anomaly Detection Features:
   - Short-term deltas ({metric}_delta_1)
   - Short rolling standard deviations ({metric}_roll_std_3)
   - Point interactions (latency_error_product, connection_pressure)
   - Relative cross-service ratios (latency_to_system_mean_ratio)

2. Incident Prediction Features:
   - Medium and long rolling means ({metric}_roll_mean_6, {metric}_roll_mean_12)
   - Rolling maximums ({metric}_roll_max_12)
   - Multi-step percentage changes ({metric}_pct_change_3)
   - Cumulative system stress indicators (system_max_error_rate, system_mean_cpu)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.data.synthetic import REQUIRED_COLUMNS

DEFAULT_ROLLING_METRICS = (
    "cpu_usage_pct",
    "memory_usage_pct",
    "latency_ms",
    "error_rate",
    "request_rate_rps",
    "active_connections",
    "connection_utilization",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for time-series feature engineering."""
    rolling_windows: tuple[int, ...] = (3, 6, 12)
    delta_steps: tuple[int, ...] = (1, 3)
    rolling_metrics: tuple[str, ...] = DEFAULT_ROLLING_METRICS
    include_cross_service: bool = True
    fill_na_zeros: bool = True
    epsilon: float = 1e-5


def extract_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Extract deterministic time-aware features from preprocessed telemetry."""
    cfg = config or FeatureConfig()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required telemetry columns for feature extraction: {missing}")

    if df.empty:
        return df.copy()

    work_df = df.copy()
    work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)
    work_df = work_df.sort_values(["service", "timestamp"]).reset_index(drop=True)

    engineered: dict[str, pd.Series] = {}
    grouped = work_df.groupby("service", sort=False)

    for metric in cfg.rolling_metrics:
        if metric not in work_df.columns:
            continue
        col_series = grouped[metric]

        for w in cfg.rolling_windows:
            roll = col_series.rolling(window=w, min_periods=1)
            mean_s = roll.mean().reset_index(level=0, drop=True)
            std_s = roll.std(ddof=0).reset_index(level=0, drop=True)
            max_s = roll.max().reset_index(level=0, drop=True)

            if cfg.fill_na_zeros:
                std_s = std_s.fillna(0.0)

            engineered[f"{metric}_roll_mean_{w}"] = mean_s
            engineered[f"{metric}_roll_std_{w}"] = std_s
            engineered[f"{metric}_roll_max_{w}"] = max_s

        for s in cfg.delta_steps:
            delta_s = col_series.diff(periods=s).reset_index(level=0, drop=True)
            prev_s = col_series.shift(periods=s).reset_index(level=0, drop=True)
            pct_s = (delta_s / (prev_s.abs() + cfg.epsilon)).reset_index(level=0, drop=True)

            if cfg.fill_na_zeros:
                delta_s = delta_s.fillna(0.0)
                pct_s = pct_s.fillna(0.0)

            engineered[f"{metric}_delta_{s}"] = delta_s
            engineered[f"{metric}_pct_change_{s}"] = pct_s

    # Domain interaction features
    engineered["latency_error_product"] = work_df["latency_ms"] * work_df["error_rate"]
    engineered["error_per_request"] = work_df["error_rate"] / (work_df["request_rate_rps"] + cfg.epsilon)
    engineered["connection_pressure"] = work_df["connection_utilization"] * work_df["active_connections"]
    engineered["traffic_ratio_in_out"] = work_df["network_in_mbps"] / (work_df["network_out_mbps"] + cfg.epsilon)

    # Cross-service system-level signals
    if cfg.include_cross_service:
        ts_group = work_df.groupby("timestamp")
        sys_mean_latency = ts_group["latency_ms"].transform("mean")
        sys_max_error = ts_group["error_rate"].transform("max")
        sys_mean_cpu = ts_group["cpu_usage_pct"].transform("mean")

        engineered["system_mean_latency"] = sys_mean_latency
        engineered["system_max_error_rate"] = sys_max_error
        engineered["system_mean_cpu"] = sys_mean_cpu
        engineered["latency_to_system_mean_ratio"] = work_df["latency_ms"] / (sys_mean_latency + cfg.epsilon)

    feat_df = pd.DataFrame(engineered, index=work_df.index)
    result_df = pd.concat([work_df, feat_df], axis=1)

    return result_df.sort_values(["timestamp", "service"]).reset_index(drop=True)


def save_features(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save engineered feature DataFrame to Parquet."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False, engine="pyarrow")
    return p
