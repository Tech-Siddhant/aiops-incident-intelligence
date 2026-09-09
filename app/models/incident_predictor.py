"""Incident prediction baseline using Logistic Regression."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

EXCLUDED_COLUMNS = {
    "timestamp",
    "service",
    "is_incident",
    "is_root_cause",
    "incident_id",
    "incident_type",
    "is_incident_in_horizon",
}


@dataclass(frozen=True)
class IncidentPredictionConfig:
    """Configuration for incident prediction baseline model."""
    horizon_seconds: int = 1800  # Default 30 minutes
    service_specific: bool = True
    lead_time_only: bool = False
    decision_threshold: float = 0.5
    class_weight: str | dict[Any, Any] | None = "balanced"
    c_reg: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 1000
    random_state: int = 42
    feature_cols: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PredictionMetrics:
    """Evaluation metrics for incident prediction."""
    precision: float
    recall: float
    f1_score: float
    pr_auc: float
    roc_auc: float
    total_samples: int
    positive_samples: int
    negative_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_horizon_labels(
    df: pd.DataFrame,
    incidents: list[dict[str, Any]],
    horizon_seconds: int = 1800,
    service_specific: bool = True,
    lead_time_only: bool = False,
) -> pd.Series:
    """Generate target labels indicating whether an incident occurs within the future horizon.

    Strictly separates target labels from feature data to prevent leakage.

    Args:
        df: Telemetry DataFrame containing 'timestamp' and optionally 'service'.
        incidents: List of ground-truth incident dictionaries.
        horizon_seconds: Prediction window size in seconds (default 1800s = 30m).
        service_specific: If True, only flags services affected by the incident.
        lead_time_only: If True, flags only pre-incident window [start - horizon, start].

    Returns:
        Boolean Series aligned with df.index named 'is_incident_in_horizon'.
    """
    if df.empty:
        return pd.Series(dtype=bool, index=df.index, name="is_incident_in_horizon")

    ts = pd.to_datetime(df["timestamp"], utc=True)
    services = df["service"].astype(str) if "service" in df.columns else pd.Series(["all"] * len(df))
    horizon_td = pd.Timedelta(seconds=horizon_seconds)

    labels = np.zeros(len(df), dtype=bool)

    for inc in incidents:
        inc_start = pd.to_datetime(inc["start_time"], utc=True)
        inc_end = pd.to_datetime(inc["end_time"], utc=True)
        root_svc = inc.get("root_cause_service")
        affected = set(inc.get("affected_services", [root_svc] if root_svc else []))

        if lead_time_only:
            # Future incident starts within (t, t + horizon]
            time_mask = (ts < inc_start) & (ts + horizon_td >= inc_start)
        else:
            # Incident is either starting or active within [t, t + horizon]
            time_mask = (ts + horizon_td >= inc_start) & (ts <= inc_end)

        if service_specific and affected:
            svc_mask = services.isin(affected)
            labels = labels | (time_mask & svc_mask).to_numpy()
        else:
            labels = labels | time_mask.to_numpy()

    return pd.Series(labels, index=df.index, name="is_incident_in_horizon")


def chronological_train_val_test_split(
    df: pd.DataFrame,
    y: pd.Series | np.ndarray | None = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Split telemetry deterministically and chronologically into train, validation, and test splits.

    Ensures zero temporal leakage across splits.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0.")

    if df.empty:
        empty_df = df.copy()
        return empty_df, empty_df, empty_df, None, None, None

    sorted_indices = pd.to_datetime(df["timestamp"], utc=True).argsort()
    df_sorted = df.iloc[sorted_indices].reset_index(drop=True)
    y_sorted = np.asarray(y)[sorted_indices] if y is not None else None

    unique_timestamps = np.sort(pd.to_datetime(df_sorted["timestamp"], utc=True).unique())
    n_ts = len(unique_timestamps)

    n_train_ts = max(1, int(n_ts * train_ratio))
    n_val_ts = max(1, int(n_ts * val_ratio)) if val_ratio > 0 else 0

    train_cutoff = unique_timestamps[n_train_ts - 1]
    val_cutoff = unique_timestamps[n_train_ts + n_val_ts - 1] if n_val_ts > 0 else train_cutoff

    ts_series = pd.to_datetime(df_sorted["timestamp"], utc=True)
    train_mask = ts_series <= train_cutoff
    val_mask = (ts_series > train_cutoff) & (ts_series <= val_cutoff)
    test_mask = ts_series > val_cutoff

    train_df = df_sorted[train_mask].reset_index(drop=True)
    val_df = df_sorted[val_mask].reset_index(drop=True)
    test_df = df_sorted[test_mask].reset_index(drop=True)

    y_train = y_sorted[train_mask] if y_sorted is not None else None
    y_val = y_sorted[val_mask] if y_sorted is not None else None
    y_test = y_sorted[test_mask] if y_sorted is not None else None

    return train_df, val_df, test_df, y_train, y_val, y_test


def calculate_prediction_metrics(
    y_true: Sequence[bool | int] | np.ndarray | pd.Series,
    y_pred: Sequence[bool | int] | np.ndarray | pd.Series,
    y_prob: Sequence[float] | np.ndarray | pd.Series,
) -> PredictionMetrics:
    """Calculate precision, recall, F1, PR-AUC, and ROC-AUC metrics."""
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    ypr = np.asarray(y_prob, dtype=float)

    total = len(yt)
    pos = int(np.sum(yt))
    neg = total - pos

    p = float(precision_score(yt, yp, zero_division=0))
    r = float(recall_score(yt, yp, zero_division=0))
    f1 = float(f1_score(yt, yp, zero_division=0))

    # ROC-AUC / PR-AUC requires at least one positive and one negative sample
    if pos > 0 and neg > 0:
        pr_auc = float(average_precision_score(yt, ypr))
        roc_auc = float(roc_auc_score(yt, ypr))
    elif pos > 0:
        pr_auc = 1.0
        roc_auc = 1.0
    else:
        pr_auc = 0.0
        roc_auc = 0.0

    return PredictionMetrics(
        precision=round(p, 4),
        recall=round(r, 4),
        f1_score=round(f1, 4),
        pr_auc=round(pr_auc, 4),
        roc_auc=round(roc_auc, 4),
        total_samples=total,
        positive_samples=pos,
        negative_samples=neg,
    )


class IncidentPredictorBaseline:
    """Logistic Regression baseline for predicting future incidents within a horizon."""

    def __init__(self, config: IncidentPredictionConfig | None = None) -> None:
        self.config = config or IncidentPredictionConfig()
        self.feature_cols_: list[str] = list(self.config.feature_cols) if self.config.feature_cols else []
        self.feature_medians_: dict[str, float] = {}
        self.scaler_: StandardScaler = StandardScaler()
        self.model_: LogisticRegression = LogisticRegression(
            class_weight=self.config.class_weight,
            C=self.config.c_reg,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        self.is_fitted_: bool = False

    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Extract, impute, and scale numeric feature matrix."""
        if fit:
            if not self.config.feature_cols:
                numeric_cols = [
                    c for c in df.columns
                    if c not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
                ]
                self.feature_cols_ = sorted(numeric_cols)
            else:
                self.feature_cols_ = [c for c in self.config.feature_cols if c in df.columns]

            if not self.feature_cols_:
                raise ValueError("No valid numeric feature columns available for prediction model.")

        feat_df = df[self.feature_cols_].copy()
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

        if fit:
            self.feature_medians_ = feat_df.median().fillna(0.0).to_dict()

        for c in self.feature_cols_:
            feat_df[c] = feat_df[c].fillna(self.feature_medians_.get(c, 0.0))

        X = feat_df.to_numpy(dtype=float)

        if fit:
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = self.scaler_.transform(X)

        return X_scaled

    def fit(self, X: pd.DataFrame, y: Sequence[bool | int] | np.ndarray | pd.Series) -> IncidentPredictorBaseline:
        """Fit scaler and Logistic Regression baseline on training data."""
        if X.empty or len(y) == 0:
            raise ValueError("Cannot fit IncidentPredictorBaseline on empty dataset.")

        y_arr = np.asarray(y, dtype=int)
        if len(X) != len(y_arr):
            raise ValueError(f"Length mismatch: X has {len(X)} rows, y has {len(y_arr)} elements.")

        X_scaled = self._prepare_features(X, fit=True)

        # Handle edge case where y has only 1 class in training split
        if len(np.unique(y_arr)) < 2:
            self.model_.classes_ = np.array([0, 1])
            self.model_.coef_ = np.zeros((1, X_scaled.shape[1]))
            self.model_.intercept_ = np.array([10.0 if y_arr[0] == 1 else -10.0])
        else:
            self.model_.fit(X_scaled, y_arr)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of an incident occurring within the horizon."""
        if not self.is_fitted_:
            raise RuntimeError("IncidentPredictorBaseline must be fitted before predict_proba.")
        if X.empty:
            return np.array([], dtype=float)

        X_scaled = self._prepare_features(X, fit=False)
        # Probabilities for class 1
        if hasattr(self.model_, "classes_") and len(self.model_.classes_) == 2:
            idx = 1 if self.model_.classes_[1] == 1 else 0
            return self.model_.predict_proba(X_scaled)[:, idx]
        return self.model_.predict_proba(X_scaled)[:, -1]

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Predict boolean incident flags using decision threshold."""
        thresh = self.config.decision_threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return probs >= thresh

    def predict_dataframe(self, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        """Generate structured DataFrame with prediction probabilities and flags."""
        probs = self.predict_proba(df)
        flags = self.predict(df, threshold=threshold)

        res = df[["timestamp", "service"]].copy() if "service" in df.columns else df[["timestamp"]].copy()
        res["incident_probability"] = np.round(probs, 4)
        res["is_predicted_incident"] = flags
        return res

    def evaluate(
        self,
        X: pd.DataFrame,
        y: Sequence[bool | int] | np.ndarray | pd.Series,
        threshold: float | None = None,
    ) -> PredictionMetrics:
        """Evaluate model against ground truth target labels."""
        probs = self.predict_proba(X)
        preds = self.predict(X, threshold=threshold)
        return calculate_prediction_metrics(y, preds, probs)

    def save(self, output_path: str | Path) -> Path:
        """Serialize model to disk."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted IncidentPredictorBaseline.")
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config": self.config,
            "feature_cols": self.feature_cols_,
            "feature_medians": self.feature_medians_,
            "scaler": self.scaler_,
            "model": self.model_,
            "is_fitted": self.is_fitted_,
        }
        joblib.dump(state, p)
        return p

    @classmethod
    def load(cls, input_path: str | Path) -> IncidentPredictorBaseline:
        """Load serialized model from disk."""
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found at {p}")
        state = joblib.load(p)
        predictor = cls(config=state["config"])
        predictor.feature_cols_ = state["feature_cols"]
        predictor.feature_medians_ = state["feature_medians"]
        predictor.scaler_ = state["scaler"]
        predictor.model_ = state["model"]
        predictor.is_fitted_ = state["is_fitted"]
        return predictor
