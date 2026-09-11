#!/usr/bin/env python3
"""CLI script to train and serialize baseline & ML models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import (
    DEFAULT_DURATION_SECONDS,
    DEFAULT_FAILURE_SCENARIO,
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    DEFAULT_SEED,
    DEFAULT_START_TIME,
    INCIDENTS_JSON,
    SYNTHETIC_DATA_DIR,
    TELEMETRY_PARQUET,
)
from app.data.preprocess import preprocess_telemetry
from app.data.synthetic import (
    SyntheticConfig,
    generate_synthetic_telemetry,
    save_synthetic_data,
)
from app.models.anomaly_isolation_forest import (
    IsolationForestConfig,
    IsolationForestDetector,
)
from app.models.incident_predictor import (
    IncidentPredictionConfig,
    IncidentPredictorBaseline,
)
from app.models.severity_classifier import (
    SeverityClassificationConfig,
    SeverityClassifierBaseline,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and save all AIOps incident intelligence models."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Directory to save trained model artifacts (default: data/models)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for training (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Force regeneration of synthetic training telemetry dataset.",
    )
    return parser.parse_args()


def load_or_create_data(regenerate: bool = False, seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, list[dict]]:
    """Load existing synthetic telemetry or generate a new batch."""
    parquet_path = SYNTHETIC_DATA_DIR / TELEMETRY_PARQUET
    json_path = SYNTHETIC_DATA_DIR / INCIDENTS_JSON

    if regenerate or not parquet_path.exists() or not json_path.exists():
        print("Generating training telemetry dataset...")
        config = SyntheticConfig(
            start_time=DEFAULT_START_TIME,
            duration_seconds=DEFAULT_DURATION_SECONDS,
            sampling_interval_seconds=DEFAULT_SAMPLING_INTERVAL_SECONDS,
            seed=seed,
            failure_scenario=DEFAULT_FAILURE_SCENARIO,
            failure_start_seconds=900,
            failure_duration_seconds=900,
        )
        df, incidents = generate_synthetic_telemetry(config)
        save_synthetic_data(df, incidents, SYNTHETIC_DATA_DIR)
        return df, incidents

    df = pd.read_parquet(parquet_path)
    with open(json_path, "r", encoding="utf-8") as f:
        incidents = json.load(f)
    return df, incidents


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data (seed={args.seed})...")
    raw_df, incidents = load_or_create_data(regenerate=args.regenerate_data, seed=args.seed)
    clean_df, _ = preprocess_telemetry(raw_df)
    print(f"Preprocessed {len(clean_df)} telemetry rows across services: {clean_df['service'].unique().tolist()}")

    # 1. Train Isolation Forest Anomaly Detector
    print("\n[1/3] Training Isolation Forest Anomaly Detector...")
    if_config = IsolationForestConfig(
        n_estimators=50,
        contamination=0.05,
        random_state=args.seed,
    )
    if_detector = IsolationForestDetector(config=if_config)
    if_detector.fit(clean_df)
    if_path = output_dir / "isolation_forest_latest.joblib"
    if_detector.save(if_path)
    print(f"  -> Saved Isolation Forest artifact to: {if_path} ({if_path.stat().st_size / 1024:.2f} KB)")

    # 2. Train Incident Predictor Baseline
    print("\n[2/3] Training Incident Predictor Baseline...")
    pred_config = IncidentPredictionConfig(
        random_state=args.seed,
        horizon_seconds=1800,
    )
    pred_model = IncidentPredictorBaseline(config=pred_config)
    y_pred = np.zeros(len(clean_df), dtype=int)
    for inc in incidents:
        mask = (clean_df["timestamp"] >= inc["start_time"]) & (clean_df["timestamp"] <= inc["end_time"])
        y_pred[mask] = 1

    pred_model.fit(clean_df, y_pred)
    pred_path = output_dir / "incident_predictor_latest.joblib"
    pred_model.save(pred_path)
    print(f"  -> Saved Incident Predictor artifact to: {pred_path} ({pred_path.stat().st_size / 1024:.2f} KB)")

    # 3. Train Severity Classifier Baseline
    print("\n[3/3] Training Severity Classifier Baseline...")
    sev_config = SeverityClassificationConfig(
        random_state=args.seed,
    )
    sev_model = SeverityClassifierBaseline(config=sev_config)
    y_sev = np.array(["low"] * len(clean_df), dtype=object)
    for inc in incidents:
        mask = (clean_df["timestamp"] >= inc["start_time"]) & (clean_df["timestamp"] <= inc["end_time"])
        y_sev[mask] = inc.get("severity", "critical")

    sev_model.fit(clean_df, y_sev)
    sev_path = output_dir / "severity_classifier_latest.joblib"
    sev_model.save(sev_path)
    print(f"  -> Saved Severity Classifier artifact to: {sev_path} ({sev_path.stat().st_size / 1024:.2f} KB)")

    total_size = (if_path.stat().st_size + pred_path.stat().st_size + sev_path.stat().st_size) / 1024
    print(f"\nModel training complete! Total artifacts footprint: {total_size:.2f} KB")


if __name__ == "__main__":
    main()
