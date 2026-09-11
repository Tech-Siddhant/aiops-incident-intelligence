"""Run RCA evaluation over synthetic data."""
import json
from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.data.synthetic import SyntheticConfig, generate_synthetic_telemetry
from app.data.preprocess import preprocess_telemetry
from app.models.anomaly_isolation_forest import IsolationForestConfig, detect_anomalies_iforest
from app.models.rca_engine import rank_root_causes
from app.models.rca_explainer import generate_rca_explanation
from evaluation.rca_evaluation import evaluate_rca_rankings

def main():
    print("Generating synthetic data...")
    cfg = SyntheticConfig(
        duration_seconds=1800,
        sampling_interval_seconds=10,
        failure_scenario="database_connection_saturation",
        failure_start_seconds=900,
        failure_duration_seconds=600,
        seed=42,
    )
    raw_df, incs = generate_synthetic_telemetry(cfg)
    clean_df, _ = preprocess_telemetry(raw_df)
    
    print("Training Isolation Forest baseline...")
    # Train on pre-incident baseline data [0, 900s]
    train_df = clean_df[clean_df["timestamp"] < "2026-01-01T00:15:00Z"]
    anom_df = detect_anomalies_iforest(
        clean_df,
        config=IsolationForestConfig(contamination=0.05, random_state=42),
        train_df=train_df,
    )
    
    print("Ranking candidates...")
    results = []
    explanations = []
    
    for inc in incs:
        rca_res = rank_root_causes(
            clean_df,
            anomalies_df=anom_df,
            incident=inc,
        )
        results.append((rca_res, inc))
        
        expl = generate_rca_explanation(
            telemetry_df=clean_df,
            anomalies_df=anom_df,
            incident=inc
        )
        explanations.append(expl.to_dict())
        
    metrics = evaluate_rca_rankings(results)
    
    print("\n--- RCA Evaluation Metrics ---")
    print(json.dumps(metrics.to_dict(), indent=2))
    
    print("\n--- Top Explanation Snippet ---")
    if explanations:
        cand_b = explanations[0].get("candidates", [])
        if cand_b:
            print("Rank 1:", cand_b[0].get("candidate_service"))
            print("Classification:", cand_b[0].get("classification"))
            print("Score Break:", cand_b[0].get("inference"))
            print("Summary:", cand_b[0].get("summary"))

if __name__ == "__main__":
    main()
