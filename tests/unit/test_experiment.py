"""Unit tests for lightweight MLOps experiment tracker."""
import tempfile
import uuid
from pathlib import Path

import pytest

from app.mlops.experiment import ExperimentRun, ExperimentTracker


def test_experiment_run_serialization():
    """Verify ExperimentRun accurately serializes to and parses from a dict."""
    run = ExperimentRun(
        model_name="isolation_forest",
        model_version="1.1.0",
        training_config={"contamination": 0.05, "random_state": 42},
        metrics={"f1_score": 0.77, "precision": 0.71},
        feature_version="1.0",
        dataset_id="synthetic_baseline_2026",
        random_seed=42,
    )
    
    d = run.to_dict()
    assert d["model_name"] == "isolation_forest"
    assert d["metrics"]["f1_score"] == 0.77
    assert "run_id" in d
    assert "training_timestamp" in d
    
    run_parsed = ExperimentRun.from_dict(d)
    assert run_parsed.run_id == run.run_id
    assert run_parsed.training_timestamp == run.training_timestamp
    assert run_parsed.metrics == run.metrics


def test_experiment_tracker_basics():
    """Verify saving, loading, and listing of experiments over file system."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = ExperimentTracker(storage_dir=tmp_dir)
        
        run = ExperimentRun(
            model_name="test_model",
            training_config={"epochs": 10},
            metrics={"accuracy": 0.95},
        )
        
        # Save run
        file_path = tracker.log_run(run)
        assert file_path.exists()
        assert file_path.suffix == ".json"
        
        # Load run
        loaded_run = tracker.load_run(run.run_id)
        assert loaded_run.model_name == "test_model"
        assert loaded_run.metrics["accuracy"] == 0.95
        
        # List runs
        runs = tracker.list_runs("test_model")
        assert len(runs) == 1
        assert runs[0].run_id == run.run_id


def test_experiment_compare_runs():
    """Verify metric-based comparison effectively ranks different versions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = ExperimentTracker(storage_dir=tmp_dir)
        
        # Baseline model
        run1 = ExperimentRun(
            model_name="baseline_zscore",
            training_config={"window": 10},
            metrics={"f1_score": 0.28},
        )
        # Better model
        run2 = ExperimentRun(
            model_name="isolation_forest",
            training_config={"contamination": 0.05},
            metrics={"f1_score": 0.77},
        )
        # Even better model variant
        run3 = ExperimentRun(
            model_name="isolation_forest",
            training_config={"contamination": 0.02},
            metrics={"f1_score": 0.85},
        )
        
        tracker.log_run(run1)
        tracker.log_run(run2)
        tracker.log_run(run3)
        
        # Compare isolation_forest by f1_score descending
        best_iforests = tracker.compare_runs(
            model_name="isolation_forest", 
            metric="f1_score", 
            reverse=True
        )
        
        assert len(best_iforests) == 2
        assert best_iforests[0].run_id == run3.run_id # highest f1
        assert best_iforests[1].run_id == run2.run_id
        
        # Missing metric fallback (baseline has no 'fpr' recorded)
        best_base = tracker.compare_runs(
            model_name="baseline_zscore", 
            metric="missing_metric"
        )
        assert len(best_base) == 1
        
def test_missing_run():
    """Verify missing run loading throws FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = ExperimentTracker(storage_dir=tmp_dir)
        with pytest.raises(FileNotFoundError):
            tracker.load_run(str(uuid.uuid4()))
