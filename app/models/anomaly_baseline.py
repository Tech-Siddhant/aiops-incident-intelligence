"""Statistical baseline anomaly detector for telemetry streams.

Detection Logic:
- Operates per service chronologically without future lookahead.
- Maintains a trailing rolling baseline (mean and standard deviation) computed over prior
  observations strictly preceding the current timestamp (using shift(1).rolling(...)).
- Computes absolute Z-scores for each configured metric against its historical baseline.
- If insufficient history (< min_periods) exists, anomaly score defaults to 0.0.
- Overall anomaly_score is the maximum Z-score across monitored metrics for that service.
- An anomaly is flagged (is_anomaly=True) when anomaly_score >= z_threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_MONITORED_METRICS = (
    "latency_ms",
    "error_rate",
    "cpu_usage_pct",
    "connection_utilization",
    "active_connections",
)


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration parameters for the statistical baseline detector."""
    window_size: int = 30
    z_threshold: float = 3.0
    min_periods: int = 5
    metrics: tuple[str, ...] = DEFAULT_MONITORED_METRICS
    epsilon: float = 1e-4


class StatisticalBaselineDetector:
    """Explainable trailing rolling z-score anomaly detector."""

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run statistical anomaly detection on preprocessed telemetry DataFrame.

        Args:
            df: Preprocessed telemetry DataFrame containing ['timestamp', 'service']
                and monitored metric columns.

        Returns:
            DataFrame with columns ['timestamp', 'service', 'anomaly_score', 'is_anomaly', 'anomalous_metrics'].
        """
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "service", "anomaly_score", "is_anomaly", "anomalous_metrics"])

        work_df = df.copy()
        work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)
        work_df = work_df.sort_values(["service", "timestamp"]).reset_index(drop=True)

        cfg = self.config
        metric_zscores: dict[str, pd.Series] = {}
        grouped = work_df.groupby("service", sort=False)

        for metric in cfg.metrics:
            if metric not in work_df.columns:
                continue

            # Trailing baseline strictly preceding current point within each service group
            roll_mean = grouped[metric].transform(
                lambda s: s.shift(1).rolling(window=cfg.window_size, min_periods=cfg.min_periods).mean()
            )
            roll_std = grouped[metric].transform(
                lambda s: s.shift(1).rolling(window=cfg.window_size, min_periods=cfg.min_periods).std(ddof=0)
            ).fillna(0.0)

            # Z-score computation
            z = (work_df[metric] - roll_mean).abs() / (roll_std + cfg.epsilon)
            metric_zscores[metric] = z.fillna(0.0)

        if not metric_zscores:
            work_df["anomaly_score"] = 0.0
            work_df["is_anomaly"] = False
            work_df["anomalous_metrics"] = ""
        else:
            z_df = pd.DataFrame(metric_zscores, index=work_df.index)
            work_df["anomaly_score"] = z_df.max(axis=1).round(4)
            work_df["is_anomaly"] = work_df["anomaly_score"] >= cfg.z_threshold

            # Construct comma-separated list of metrics exceeding threshold
            def _flagged_metrics(row: pd.Series) -> str:
                flagged = [m for m, val in row.items() if val >= cfg.z_threshold]
                return ",".join(flagged)

            work_df["anomalous_metrics"] = z_df.apply(_flagged_metrics, axis=1)

        result_cols = ["timestamp", "service", "anomaly_score", "is_anomaly", "anomalous_metrics"]
        return work_df[result_cols].sort_values(["timestamp", "service"]).reset_index(drop=True)


def detect_anomalies(df: pd.DataFrame, config: BaselineConfig | None = None) -> pd.DataFrame:
    """Functional convenience wrapper for StatisticalBaselineDetector."""
    detector = StatisticalBaselineDetector(config=config)
    return detector.detect(df)
