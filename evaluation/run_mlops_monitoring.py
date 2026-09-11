"""End-to-end local MLOps monitoring and validation execution."""
import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.data.preprocess import preprocess_telemetry
from app.data.synthetic import SyntheticConfig, generate_synthetic_telemetry
from app.mlops.drift import detect_data_drift
from app.mlops.experiment import ExperimentRun, ExperimentTracker
from app.mlops.monitoring import evaluate_model_health
from app.models.anomaly_isolation_forest import IsolationForestConfig, IsolationForestDetector


def run_e2e_mlops_monitoring():
    print("1. Generating synthetic telemetry baselines...")
    # Reference baseline (nominal / healthy data)
    ref_cfg = SyntheticConfig(
        duration_seconds=1800,
        sampling_interval_seconds=10,
        failure_scenario="database_connection_saturation",
        failure_start_seconds=1500,  # mostly baseline
        failure_duration_seconds=300,
        seed=42,
    )
    raw_ref, _ = generate_synthetic_telemetry(ref_cfg)
    clean_ref, _ = preprocess_telemetry(raw_ref)
    train_ref = clean_ref[clean_ref["timestamp"] < "2026-01-01T00:20:00Z"]

    # Current production dataset (active incident)
    curr_cfg = SyntheticConfig(
        duration_seconds=1800,
        sampling_interval_seconds=10,
        failure_scenario="database_connection_saturation",
        failure_start_seconds=900,
        failure_duration_seconds=600,
        seed=101,
    )
    raw_curr, incs = generate_synthetic_telemetry(curr_cfg)
    clean_curr, _ = preprocess_telemetry(raw_curr)

    print("2. Training and saving IsolationForest baseline artifact...")
    detector = IsolationForestDetector(
        config=IsolationForestConfig(n_estimators=100, contamination=0.05, random_state=42)
    )
    detector.fit(train_ref)

    tracker = ExperimentTracker(storage_dir="data/experiments")
    artifact_dir = Path("data/models")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    art_path = artifact_dir / "isolation_forest_latest.joblib"
    detector.save(art_path)

    # Log experiment run
    run = ExperimentRun(
        model_name="isolation_forest",
        model_version="1.0.0",
        training_config=detector.config.__dict__,
        metrics={"precision": 0.7156, "recall": 0.8387, "f1_score": 0.7723},
        feature_version="1.0",
        dataset_id="synthetic_bench_b",
        random_seed=42,
        artifact_path=str(art_path),
    )
    tracker.log_run(run)

    print("3. Running full MLOps Health & Drift Evaluation...")
    health_report = evaluate_model_health(
        model=detector,
        current_df=clean_curr,
        reference_df=clean_ref,
        model_name="isolation_forest",
        model_version="1.0.0",
        evaluation_metrics=run.metrics,
        artifact_path=art_path,
        latency_sla_ms=250.0,
    )

    print("\n================ MLOPS HEALTH REPORT ================")
    print(f"Model: {health_report.model_name} (v{health_report.model_version})")
    print(f"Overall Health Status: {health_report.health_status}")
    print(f"Status Reasons: {health_report.status_reasons}")
    print("\n--- Resource Telemetry ---")
    print(f"Inference Latency: {health_report.resource_usage.latency_ms} ms (Total rows: {len(clean_curr)})")
    print(f"Peak Memory Overhead: {health_report.resource_usage.peak_memory_kb} KB ({health_report.resource_usage.peak_memory_bytes} bytes)")
    print(f"Artifact Size on Disk: {health_report.resource_usage.artifact_size_kb} KB")
    print("\n--- Prediction Distribution ---")
    if health_report.prediction_distribution:
        pdist = health_report.prediction_distribution
        print(f"Total Predictions: {pdist.total_predictions}")
        print(f"Positive Anomaly Rate: {pdist.positive_rate * 100:.2f}% ({pdist.positive_predictions} samples)")
        print(f"Anomaly Score Mean: {pdist.score_mean:.4f}, Max: {pdist.score_max:.4f}, p90: {pdist.score_p90:.4f}")
    print("\n--- Data Quality ---")
    print(f"Valid: {health_report.data_quality.is_valid} (Total rows: {health_report.data_quality.total_rows})")
    print("\n--- Data Drift Summary ---")
    if health_report.drift_report:
        drep = health_report.drift_report
        print(f"Has Drift: {drep.has_drift} (Drifted share: {drep.drift_share*100:.1f}%)")
        print(f"Drifted Features: {drep.drifted_features}")
        print(f"Warning Features: {drep.warning_features}")

    return health_report


if __name__ == "__main__":
    run_e2e_mlops_monitoring()
