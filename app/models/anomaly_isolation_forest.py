"""Isolation Forest anomaly detector for telemetry features.

Detection Logic:
- Fits scikit-learn IsolationForest models (per-service or system-wide) on historical telemetry.
- Cleans and handles missing/infinite numeric feature values deterministically.
- Produces continuous anomaly scores (-score_samples) where higher values denote greater anomaly severity.
- Produces boolean anomaly flags (predict == -1) respecting configured contamination.
- Provides lightweight model serialization (joblib).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DEFAULT_IFOREST_FEATURES = (
    "latency_ms",
    "error_rate",
    "cpu_usage_pct",
    "memory_usage_pct",
    "disk_usage_pct",
    "network_in_mbps",
    "network_out_mbps",
    "request_rate_rps",
    "active_connections",
    "connection_utilization",
)


@dataclass(frozen=True)
class IsolationForestConfig:
    """Configuration for Isolation Forest anomaly detection."""
    n_estimators: int = 100
    contamination: float | str = 0.05
    random_state: int = 42
    max_samples: float | int | str = "auto"
    feature_cols: tuple[str, ...] = DEFAULT_IFOREST_FEATURES
    per_service: bool = True
    n_jobs: int = 1


class IsolationForestDetector:
    """Machine learning anomaly detector based on Isolation Forest."""

    def __init__(self, config: IsolationForestConfig | None = None) -> None:
        self.config = config or IsolationForestConfig()
        self.models_: dict[str, IsolationForest] = {}
        self.global_model_: IsolationForest | None = None
        self.feature_medians_: dict[str, float] = {}
        self.is_fitted_: bool = False

    def _prepare_features(self, df: pd.DataFrame, fit_medians: bool = False) -> tuple[pd.DataFrame, list[str]]:
        """Validate, extract, and impute numeric feature columns."""
        available_cols = [c for c in self.config.feature_cols if c in df.columns]
        if not available_cols:
            raise ValueError(f"None of the configured feature columns {self.config.feature_cols} found in DataFrame.")

        feat_df = df[available_cols].copy()
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

        if fit_medians:
            self.feature_medians_ = feat_df.median().fillna(0.0).to_dict()

        # Impute missing values with learned medians or 0.0
        for col in available_cols:
            median_val = self.feature_medians_.get(col, 0.0)
            feat_df[col] = feat_df[col].fillna(median_val)

        return feat_df, available_cols

    def fit(self, df: pd.DataFrame) -> IsolationForestDetector:
        """Fit Isolation Forest model(s) on baseline telemetry data."""
        if df.empty:
            raise ValueError("Cannot fit IsolationForestDetector on empty DataFrame.")
        if "service" not in df.columns:
            raise ValueError("DataFrame must contain 'service' column.")

        clean_feats, feature_cols = self._prepare_features(df, fit_medians=True)
        work_df = df[["service"]].copy()
        work_df = pd.concat([work_df, clean_feats], axis=1)

        cfg = self.config
        self.models_ = {}

        if cfg.per_service:
            for service_name, group in work_df.groupby("service", sort=False):
                X_svc = group[feature_cols].values
                if len(X_svc) < 2:
                    continue
                clf = IsolationForest(
                    n_estimators=cfg.n_estimators,
                    contamination=cfg.contamination,
                    random_state=cfg.random_state,
                    max_samples=cfg.max_samples,
                    n_jobs=cfg.n_jobs,
                )
                clf.fit(X_svc)
                self.models_[str(service_name)] = clf
        else:
            clf = IsolationForest(
                n_estimators=cfg.n_estimators,
                contamination=cfg.contamination,
                random_state=cfg.random_state,
                max_samples=cfg.max_samples,
                n_jobs=cfg.n_jobs,
            )
            clf.fit(clean_feats.values)
            self.global_model_ = clf

        self.is_fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score anomalies on telemetry DataFrame.

        Returns DataFrame with columns ['timestamp', 'service', 'anomaly_score', 'is_anomaly'].
        """
        if not self.is_fitted_:
            raise RuntimeError("IsolationForestDetector must be fitted before predict.")

        if df.empty:
            return pd.DataFrame(columns=["timestamp", "service", "anomaly_score", "is_anomaly"])

        for req_col in ("timestamp", "service"):
            if req_col not in df.columns:
                raise ValueError(f"Missing required column '{req_col}' for anomaly prediction.")

        work_df = df.copy()
        work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)
        clean_feats, feature_cols = self._prepare_features(work_df, fit_medians=False)

        scores = np.zeros(len(work_df), dtype=float)
        flags = np.zeros(len(work_df), dtype=bool)

        cfg = self.config
        if cfg.per_service:
            for service_name, group_indices in work_df.groupby("service", sort=False).indices.items():
                service_str = str(service_name)
                if service_str in self.models_:
                    clf = self.models_[service_str]
                    X_svc = clean_feats.iloc[group_indices].values
                    # -score_samples produces positive score where higher = more anomalous
                    raw_scores = -clf.score_samples(X_svc)
                    preds = clf.predict(X_svc) == -1
                    scores[group_indices] = raw_scores
                    flags[group_indices] = preds
        else:
            if self.global_model_ is not None:
                X = clean_feats.values
                scores = -self.global_model_.score_samples(X)
                flags = self.global_model_.predict(X) == -1

        work_df["anomaly_score"] = np.round(scores, 4)
        work_df["is_anomaly"] = flags

        result_cols = ["timestamp", "service", "anomaly_score", "is_anomaly"]
        return work_df[result_cols].sort_values(["timestamp", "service"]).reset_index(drop=True)

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on provided DataFrame and return anomaly predictions."""
        return self.fit(df).predict(df)

    def save(self, output_path: str | Path) -> Path:
        """Serialize fitted detector to disk."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted IsolationForestDetector.")
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config": self.config,
            "models": self.models_,
            "global_model": self.global_model_,
            "feature_medians": self.feature_medians_,
            "is_fitted": self.is_fitted_,
        }
        joblib.dump(state, p)
        return p

    @classmethod
    def load(cls, input_path: str | Path) -> IsolationForestDetector:
        """Load serialized detector from disk."""
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found at {p}")
        state = joblib.load(p)
        detector = cls(config=state["config"])
        detector.models_ = state["models"]
        detector.global_model_ = state["global_model"]
        detector.feature_medians_ = state["feature_medians"]
        detector.is_fitted_ = state["is_fitted"]
        return detector


def detect_anomalies_iforest(
    df: pd.DataFrame,
    config: IsolationForestConfig | None = None,
    train_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convenience function to fit/predict Isolation Forest anomalies."""
    detector = IsolationForestDetector(config=config)
    if train_df is not None:
        detector.fit(train_df)
        return detector.predict(df)
    return detector.fit_predict(df)
