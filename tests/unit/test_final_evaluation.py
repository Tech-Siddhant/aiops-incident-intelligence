"""Unit tests for the final comprehensive evaluation benchmark pipeline."""
import pytest
from evaluation.run_final_evaluation import (
    evaluate_anomaly_layer,
    evaluate_prediction_and_severity_layers,
    evaluate_rca_layer,
    evaluate_system_and_performance,
)


def test_evaluate_anomaly_layer():
    results = evaluate_anomaly_layer()
    assert "statistical_baseline" in results
    assert "isolation_forest" in results

    base = results["statistical_baseline"]
    iforest = results["isolation_forest"]

    assert 0.0 <= base["f1_score"] <= 1.0
    assert 0.0 <= iforest["f1_score"] <= 1.0
    assert iforest["recall"] >= base["recall"]
    assert base["total_samples"] == 360


def test_evaluate_prediction_and_severity_layers():
    results = evaluate_prediction_and_severity_layers()
    assert "prediction_train" in results
    assert "prediction_val" in results
    assert "prediction_test" in results
    assert "severity_metrics" in results

    train_m = results["prediction_train"]
    test_m = results["prediction_test"]
    assert train_m["total_samples"] > 0
    assert test_m["total_samples"] > 0
    assert 0.0 <= test_m["f1_score"] <= 1.0
    assert 0.0 <= test_m["roc_auc"] <= 1.0
    assert 0.0 <= test_m["pr_auc"] <= 1.0


def test_evaluate_rca_layer():
    results = evaluate_rca_layer()
    assert "metrics" in results
    assert "top_ranked_candidates" in results
    metrics = results["metrics"]
    assert metrics["top_1_accuracy"] == 1.0
    assert metrics["top_3_accuracy"] == 1.0
    assert metrics["mean_reciprocal_rank"] == 1.0


def test_evaluate_system_and_performance():
    results = evaluate_system_and_performance()
    assert "artifact_sizes" in results
    assert "inference_metrics" in results
    assert "api_latencies" in results

    arts = results["artifact_sizes"]
    assert arts["total_models_size_kb"] > 0
    assert arts["isolation_forest_kb"] > 0

    inf = results["inference_metrics"]
    assert inf["batch_size_records"] > 0
    assert inf["isolation_forest_batch_ms"] > 0
    assert inf["peak_inference_memory_kb"] > 0

    api = results["api_latencies"]
    assert api["get_health_ms"] > 0
    assert api["post_anomalies_detect_ms"] > 0
