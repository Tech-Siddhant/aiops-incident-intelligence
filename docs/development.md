# Developer Guide & Workflow Standards

## 1. Supported Environment
- **Python Version**: `>=3.10` (tested on Python 3.12 / Linux Codespaces).
- **Package Management**: Standard `pip` and PEP 517/621 `pyproject.toml`.

## 2. Environment Setup

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install in editable mode with all dependencies
pip install -e ".[test]"
# Or: pip install -r requirements.txt
```

## 3. Telemetry & Data Workflows

### Generating Synthetic Data
To generate deterministic synthetic telemetry simulating the database connection saturation cascade:
```bash
python scripts/generate_telemetry.py --duration 3600 --interval 10 --scenario database_connection_saturation
```
Generated artifacts are saved to `data/synthetic/`:
- `data/synthetic/telemetry.parquet` (columnar time-series dataset)
- `data/synthetic/incidents.json` (incident metadata & ground-truth labels)

### Validating Telemetry
To verify that generated datasets strictly adhere to data contract bounds and temporal monotonicity:
```bash
python scripts/validate_telemetry.py
```

### Training & Serializing Models
To train the baseline and ML models (Isolation Forest, Incident Predictor, Severity Classifier) and output compressed artifacts to `data/models/`:
```bash
python scripts/train_models.py
```

## 4. Serving & Web Dashboard

Start the FastAPI backend with the interactive frontend dashboard:
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Documentation (Swagger UI): `http://localhost:8000/docs`
- Interactive Operations Dashboard: `http://localhost:8000/`

## 5. Running Evaluations & MLOps Monitoring

### Comprehensive Final Benchmark Suite
Execute the multi-layer evaluation pipeline across Anomaly Detection, Prediction, Severity Classification, RCA, and System Latency/Memory:
```bash
python evaluation/run_final_evaluation.py
```

### Continuous Drift & Health Evaluation
Run Kolmogorov-Smirnov distribution drift testing, missing rate checks, and model latency profiling:
```bash
python evaluation/run_mlops_monitoring.py
```

## 6. Configuration & Determinism
Central configuration resides in `app/config.py`. Key runtime parameters:
- `ROOT_DIR`: Root repository path resolved via `pathlib.Path`.
- `DATA_DIR`: Base data directory (configurable via `AIOPS_DATA_DIR` environment variable).
- `SYNTHETIC_DATA_DIR`: `DATA_DIR / "synthetic"`.
- `DEFAULT_SEED`: Base random seed (`42`, configurable via `AIOPS_SEED`).
- `DEFAULT_SAMPLING_INTERVAL_SECONDS`: Metric collection interval (`10` seconds).
- `DEFAULT_DURATION_SECONDS`: Simulation window length (`3600` seconds).

## 7. Running Tests
Run the automated test suite locally:
```bash
pytest -q
```
All tests execute offline without external network or database dependencies.

## 8. Code & Contribution Conventions
- **Minimal Abstraction**: Adhere to YAGNI. Favor standard library primitives over custom boilerplate.
- **Type Annotations**: Provide explicit type hints for function signatures and data structures.
- **Pure & Deterministic**: Ensure all data generation and feature transformations accept an explicit random seed or configuration object.
- **Relative Path Resolution**: Never hardcode absolute filesystem paths. Always resolve paths relative to `ROOT_DIR` or use `pathlib.Path`.

## 9. Pre-Submission Checklist
Before committing or opening a pull request, run:
```bash
# 1. Run all unit & integration tests
pytest -q

# 2. Regenerate standard dataset
python scripts/generate_telemetry.py --duration 3600 --interval 10

# 3. Verify quality and contract compliance
python scripts/validate_telemetry.py

# 4. Retrain model artifacts
python scripts/train_models.py

# 5. Execute benchmark evaluation
python evaluation/run_final_evaluation.py
```

