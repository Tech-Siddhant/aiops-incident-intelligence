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

## Data & Model Drift Detection (Phase 6.2)

### Methods & Metrics
To monitor production telemetry features and detected anomaly distributions without deploying heavyweight streaming infrastructure (Evidently/Alibi/Spark):
1. **Population Stability Index (PSI)**:
   - Uses reference quantile binning (default 10 bins) with epsilon floor protections.
   - Thresholds:
     - `PSI < 0.1`: **NO_DRIFT** (Stable distribution).
     - `0.1 <= PSI < 0.2`: **WARNING** (Moderate distribution shift).
     - `PSI >= 0.2`: **DRIFT** (Significant distribution shift requiring operational review).
2. **Two-Sample Kolmogorov-Smirnov Test (KS Test)**:
   - Evaluates whether current feature values originate from the reference empirical cumulative distribution.
   - Combines p-value significance (`p < 0.05`) with KS statistic distance magnitude (`stat >= 0.2` for significant drift, `stat >= 0.1` for warning).
3. **Robust Edge Handling**:
   - **Constant Features**: Identical constant values return `NO_DRIFT` (PSI=0.0, KS=0.0); constant value shifts return `DRIFT`.
   - **Missing Features**: Dropped or missing columns are surfaced as `MISSING` and counted toward `drifted_features`.
   - **NaN / Infinite Handling**: Filtered safely per column before quantile calculations.

### Model Performance Degradation
When post-incident ground-truth labels become available:
- `detect_performance_degradation()` checks baseline vs current metrics (`f1_score`, `precision`, `recall`).
- Relative or absolute drop exceeding `max_allowed_drop` (default 5%) flags the model as degraded (`is_degraded = True`).

