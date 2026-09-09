"""Incident severity classification baseline using Logistic Regression."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

SEVERITY_CLASSES = ("low", "medium", "high", "critical")

EXCLUDED_COLUMNS = {
    "timestamp",
    "service",
    "severity",
    "incident_id",
    "incident_type",
    "description",
    "start_time",
    "detection_time",
    "end_time",
    "is_incident",
    "is_root_cause",
    "is_incident_in_horizon",
}


@dataclass(frozen=True)
class SeverityClassificationConfig:
    """Configuration for severity classification model."""
    class_weight: str | dict[Any, Any] | None = "balanced"
    c_reg: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 1000
    random_state: int = 42
    feature_cols: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SeverityMetrics:
    """Evaluation metrics for incident severity classification."""
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    classes: list[str]
    total_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_severity_labels(
    df: pd.DataFrame,
    incidents: list[dict[str, Any]],
) -> pd.Series:
    """Extract incident severity labels aligned with telemetry timestamps and services.

    Only uses pre-incident and onset time windows. Ignores post-incident resolution fields.

    Returns:
        pd.Series containing severity strings ('low', 'medium', 'high', 'critical')
        or None for non-incident time points.
    """
    if df.empty:
        return pd.Series(dtype=object, index=df.index, name="severity")

    ts = pd.to_datetime(df["timestamp"], utc=True)
    services = df["service"].astype(str) if "service" in df.columns else pd.Series(["all"] * len(df))
    severities: list[str | None] = [None] * len(df)

    for inc in incidents:
        inc_start = pd.to_datetime(inc["start_time"], utc=True)
        inc_end = pd.to_datetime(inc["end_time"], utc=True)
        sev = str(inc.get("severity", "medium")).lower()
        root_svc = inc.get("root_cause_service")
        affected = set(inc.get("affected_services", [root_svc] if root_svc else []))

        time_mask = (ts >= inc_start) & (ts <= inc_end)

        for i in np.where(time_mask)[0]:
            svc = services.iloc[i]
            if not affected or svc in affected:
                severities[i] = sev

    return pd.Series(severities, index=df.index, dtype=object, name="severity")


def calculate_severity_metrics(
    y_true: Sequence[str] | np.ndarray | pd.Series,
    y_pred: Sequence[str] | np.ndarray | pd.Series,
    labels: list[str] | None = None,
) -> SeverityMetrics:
    """Compute macro F1, per-class precision/recall/F1, and confusion matrix."""
    yt = np.asarray(y_true, dtype=str)
    yp = np.asarray(y_pred, dtype=str)

    if labels is not None:
        target_labels = list(labels)
        if len(target_labels) == 1 and target_labels[0] in SEVERITY_CLASSES:
            target_labels = list(SEVERITY_CLASSES)
    else:
        unique_present = sorted(list(set(yt) | set(yp)))
        target_labels = list(SEVERITY_CLASSES) if set(unique_present).issubset(set(SEVERITY_CLASSES)) else unique_present

    if not target_labels:
        return SeverityMetrics(
            macro_f1=0.0,
            per_class_precision={},
            per_class_recall={},
            per_class_f1={},
            confusion_matrix=[],
            classes=[],
            total_samples=0,
        )

    prec, rec, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=target_labels, zero_division=0
    )
    macro_f = float(f1_score(yt, yp, average="macro", zero_division=0))
    cm = confusion_matrix(yt, yp, labels=target_labels).tolist()

    return SeverityMetrics(
        macro_f1=round(macro_f, 4),
        per_class_precision={lbl: round(float(p), 4) for lbl, p in zip(target_labels, prec)},
        per_class_recall={lbl: round(float(r), 4) for lbl, r in zip(target_labels, rec)},
        per_class_f1={lbl: round(float(f), 4) for lbl, f in zip(target_labels, f1)},
        confusion_matrix=cm,
        classes=target_labels,
        total_samples=len(yt),
    )


class SeverityClassifierBaseline:
    """Logistic Regression baseline for incident severity classification."""

    def __init__(self, config: SeverityClassificationConfig | None = None) -> None:
        self.config = config or SeverityClassificationConfig()
        self.feature_cols_: list[str] = list(self.config.feature_cols) if self.config.feature_cols else []
        self.feature_medians_: dict[str, float] = {}
        self.scaler_: StandardScaler = StandardScaler()
        self.classes_: np.ndarray = np.array([])
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
                raise ValueError("No valid numeric feature columns found for severity classification.")

        feat_df = df[self.feature_cols_].copy()
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

        if fit:
            self.feature_medians_ = feat_df.median().fillna(0.0).to_dict()

        for c in self.feature_cols_:
            feat_df[c] = feat_df[c].fillna(self.feature_medians_.get(c, 0.0))

        X = feat_df.to_numpy(dtype=float)
        return self.scaler_.fit_transform(X) if fit else self.scaler_.transform(X)

    def fit(self, X: pd.DataFrame, y: Sequence[str] | np.ndarray | pd.Series) -> SeverityClassifierBaseline:
        """Fit scaler and classifier on incident training data."""
        if X.empty or len(y) == 0:
            raise ValueError("Cannot fit SeverityClassifierBaseline on empty dataset.")

        y_arr = np.asarray(y, dtype=str)
        if len(X) != len(y_arr):
            raise ValueError(f"Length mismatch: X has {len(X)} rows, y has {len(y_arr)} elements.")

        X_scaled = self._prepare_features(X, fit=True)
        unique_classes = np.unique(y_arr)
        self.classes_ = unique_classes

        # Handle edge case with single class in training split
        if len(unique_classes) < 2:
            self.model_.classes_ = unique_classes
        else:
            self.model_.fit(X_scaled, y_arr)
            self.classes_ = self.model_.classes_

        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict severity class labels."""
        if not self.is_fitted_:
            raise RuntimeError("SeverityClassifierBaseline must be fitted before predict.")
        if X.empty:
            return np.array([], dtype=str)

        if len(self.classes_) < 2:
            return np.array([str(self.classes_[0])] * len(X))

        X_scaled = self._prepare_features(X, fit=False)
        return self.model_.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise RuntimeError("SeverityClassifierBaseline must be fitted before predict_proba.")
        if X.empty:
            return np.empty((0, len(self.classes_)), dtype=float)

        if len(self.classes_) < 2:
            return np.ones((len(X), 1), dtype=float)

        X_scaled = self._prepare_features(X, fit=False)
        return self.model_.predict_proba(X_scaled)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate structured DataFrame with predicted severity and class probabilities."""
        preds = self.predict(df)
        probs = self.predict_proba(df)

        res = df[["timestamp", "service"]].copy() if "service" in df.columns else df[["timestamp"]].copy()
        res["predicted_severity"] = preds

        for idx, cls_name in enumerate(self.classes_):
            res[f"prob_{cls_name}"] = np.round(probs[:, idx], 4)

        return res

    def evaluate(self, X: pd.DataFrame, y: Sequence[str] | np.ndarray | pd.Series) -> SeverityMetrics:
        """Evaluate classifier on ground truth severity labels."""
        preds = self.predict(X)
        return calculate_severity_metrics(y, preds, labels=list(self.classes_))

    def save(self, output_path: str | Path) -> Path:
        """Serialize severity classifier to disk."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted SeverityClassifierBaseline.")
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config": self.config,
            "feature_cols": self.feature_cols_,
            "feature_medians": self.feature_medians_,
            "scaler": self.scaler_,
            "classes": self.classes_,
            "model": self.model_,
            "is_fitted": self.is_fitted_,
        }
        joblib.dump(state, p)
        return p

    @classmethod
    def load(cls, input_path: str | Path) -> SeverityClassifierBaseline:
        """Load serialized severity classifier from disk."""
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found at {p}")
        state = joblib.load(p)
        classifier = cls(config=state["config"])
        classifier.feature_cols_ = state["feature_cols"]
        classifier.feature_medians_ = state["feature_medians"]
        classifier.scaler_ = state["scaler"]
        classifier.classes_ = state["classes"]
        classifier.model_ = state["model"]
        classifier.is_fitted_ = state["is_fitted"]
        return classifier
