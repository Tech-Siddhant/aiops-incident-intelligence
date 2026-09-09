# Phase 6: MLOps & Model Tracking

## Architecture: Lightweight Experiment Tracker
Omniroute utilizes a strictly file-backed lightweight component for experiment and version tracking (`ExperimentTracker`), implemented purely using native Python `dataclasses` and standard library `json` serialization. 

### Why skipped MLflow? (**Ponytail Justification**)
- **Operational Bloat**: Launching and maintaining an MLflow tracking server and SQL database inherently violates the lightweight local agent constraints unless model versioning scales to >10,000 runs per day across remote distributed teams.
- **YAGNI**: The current architecture trains static baselines and deterministic Isolation Forests on synthetic datasets locally. Everything fits cleanly into plain-text JSON files on the local filesystem (`data/experiments/`).
- **Upgrade Path**: If the system footprint expands to multi-container clusters, external dashboarding, or hyperparameter sweep distributions, this JSON API interface acts exactly like MLflow's local client backend natively; it can be replaced by `mlflow.log_param()` gracefully behind the existing `ExperimentTracker.log_run()` abstraction.

## Tracked Metadata Schema
Each training execution records the following fixed schema without exposing API secrets or credentials:
- `run_id`: Unique deterministic UUID per generated model iteration.
- `model_name` & `model_version`: Tracks algorithm (e.g., `isolation_forest`) and semantic release iteration.
- `training_config`: Full JSON serialization of the model hyperparameters (e.g., `IsolationForestConfig`).
- `feature_version` & `dataset_id`: Links the executed metric scores squarely against the preprocessed telemetry pipelines.
- `random_seed`: Guarantee reproducible bounds.
- `training_timestamp`: Chronological history parsing.
- `metrics`: Evaluation accuracy signals (`precision`, `recall`, `f1_score`).
- `artifact_path`: (Optional) URL or filepath directing where the final compiled `.pkl` or `.onnx` binaries preside.

## Component Operations
- **`log_run(run: ExperimentRun)`**: Dumps to disk via standard fast file I/O.
- **`compare_runs(model_name: str, metric: str)`**: Rapid chronological linear scan allowing for sorting/promotion between different executed candidate versions (i.e. finding the best `f1_score` threshold for `isolation_forest`).

## Readiness
The local filesystem tracking infrastructure perfectly satisfies Phase 6.1 Requirements allowing us to record and evaluate future MLOps pipelines reliably.
