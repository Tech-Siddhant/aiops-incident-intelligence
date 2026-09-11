# OmniRoute: AIOps Incident Intelligence Platform

A lightweight, deterministic, and modular AIOps platform for microservice telemetry analysis, multivariate anomaly detection, early-warning incident prediction, automated root cause analysis (RCA), and operational remediation recommendations.

---

## 1. Overview & Problem Statement

In modern distributed microservice architectures, single-point failures rapidly cascade across service dependencies. For example, database connection pool exhaustion manifests not only at the database tier but cascades upstream into request queuing, elevated p95/p99 latencies, thread exhaustion, and HTTP 504 gateway timeouts.

Diagnosing these incidents manually is slow and error-prone due to alert storms and noisy cross-service correlations. **OmniRoute** provides an end-to-end incident intelligence pipeline that:
1. **Detects Anomalies**: Identifies metric deviations in real time using both statistical baselines and multivariate Isolation Forests.
2. **Predicts Incidents**: Provides an early-warning failure risk score over a 30-minute predictive horizon before SLAs breach.
3. **Classifies Severity**: Categorizes incident impact (`low`, `medium`, `high`, `critical`) based on degradation intensity and blast radius.
4. **Isolates Root Causes**: Accurately pinpoints the root-cause service and causal metrics using dependency topology and temporal onset sequence.
5. **Generates Explainable Remediation**: Translates graph and metric evidence into natural-language runbooks and actionable remediation steps for on-call engineers.
6. **Monitors Model Health & Drift**: Continuously tracks data quality, schema integrity, and statistical distribution drift via two-sample Kolmogorov-Smirnov tests.

---

## 2. System Architecture & Topology

### Microservice Dependency Topology
```
      [ Clients / Users ]
               │
               ▼
     ┌───────────────────┐
     │    api_gateway    │
     └─────────┬─────────┘
               ├───► [ auth_service ] (Independent Path)
               │
               ▼
     ┌───────────────────┐
     │  orders_service   │
     └─────────┬─────────┘
               │
               ▼
     ┌───────────────────┐
     │     database      │
     └───────────────────┘
```

### Cascading Failure Dynamics (`database_connection_saturation`)
1. **$t_0$ (Database Layer)**: Connection pool exhausts (`connection_utilization` $\to 1.0$), query wait queue spikes, database `latency_ms` rises.
2. **$t_0 + 20\text{s}$ (Orders Service Layer)**: Worker threads block waiting for database connections; query timeouts induce rising `error_rate` and `latency_ms`.
3. **$t_0 + 40\text{s}$ (API Gateway Layer)**: Upstream orders requests stall; gateway latency spikes and HTTP 504 Gateway Timeout rates surge.
4. **Auth Service**: Operates independently with baseline metric distributions, serving as a negative-control blast radius boundary.

### End-to-End System Pipeline
```
[ Synthetic Telemetry Engine ] ──► [ Data Quality Validator ]
               │
               ▼
[ Preprocessing & Feature Extraction ] (Rolling windows, EWMA, Z-scores)
               │
       ┌───────┴──────────────────────────┬─────────────────────────┐
       ▼                                  ▼                         ▼
[ Anomaly Detection ]           [ Incident Prediction ]   [ Severity Classifier ]
(Isolation Forest / Z-score)    (Logistic Regression)     (Multi-Class Baseline)
       │                                  │                         │
       └──────────────────────────┬───────┴─────────────────────────┘
                                  ▼
                     [ Graph & Temporal RCA Engine ]
                                  │
                                  ▼
                 [ Natural Language Explainer & Runbooks ]
                                  │
                                  ▼
       ┌──────────────────────────┴─────────────────────────┐
       ▼                                                    ▼
[ FastAPI REST Endpoints ]                       [ MLOps & Drift Monitor ]
       │                                                    │
       └──────────────────────────┬─────────────────────────┘
                                  ▼
               [ Interactive Operations Web Dashboard ]
```

---

## 3. Empirical Benchmark & Evaluation Results

All metrics were rigorously evaluated using chronological partitions (zero future-leakage) on multi-service telemetry datasets. Results are documented in detail in `docs/final-evaluation.md`.

| Operational Domain | Evaluation Metric | Planned Target | Actual Measured Result | Target Met? |
| :--- | :--- | :--- | :--- | :--- |
| **Anomaly Detection** | Precision | $\ge 70.0\%$ | **71.56%** | **PASSED** |
| | Recall | $\ge 80.0\%$ | **83.87%** | **PASSED** |
| | F1 Score | $\ge 0.7500$ | **0.7723** | **PASSED** |
| | False Positive Rate (FPR) | $\le 15.0\%$ | **11.61%** | **PASSED** |
| | Detection Delay | $\le 30.0\text{s}$ | **10.0s** | **PASSED** |
| **Incident Prediction** | PR-AUC (Precision-Recall AUC) | $\ge 0.7500$ | **0.7599** | **PASSED** |
| | ROC-AUC | $\ge 0.8000$ | **0.8130** | **PASSED** |
| | F1 Score (Test Split) | $\ge 0.6500$ | **0.6547** | **PASSED** |
| | Precision (Test Split) | $\ge 60.0\%$ | **63.08%** | **PASSED** |
| | Recall (Test Split) | $\ge 65.0\%$ | **68.05%** | **PASSED** |
| | Early Warning Lead Time | $\ge 900.0\text{s}$ (15 min) | **1,632.5s (~27.2 min)** | **PASSED** |
| **Severity Classification** | Macro F1 Score | $\ge 0.7000$ | **1.0000** | **PASSED** |
| **Root Cause Analysis (RCA)** | Top-1 Accuracy | $\ge 90.0\%$ | **100.0% (1.0)** | **PASSED** |
| | Top-3 Accuracy | $\ge 95.0\%$ | **100.0% (1.0)** | **PASSED** |
| | Mean Reciprocal Rank (MRR) | $\ge 0.9000$ | **1.0000** | **PASSED** |
| **System & Latency** | Max Batch API Latency ($N=720$) | $\le 500\text{ms}$ | **144.45 ms** | **PASSED** |
| | Model Artifact Disk Size | $\le 25\text{MB}$ | **2.20 MB (2,248 KB)** | **PASSED** |
| | Peak Inference Memory Overhead | $\le 10\text{MB}$ | **306.28 KB** | **PASSED** |

---

## 4. Key Features & Modules

- **Synthetic Telemetry Engine (`app.data.synthetic`)**: Generates multivariate time-series across services (`latency_ms`, `error_rate`, `cpu_usage`, `memory_usage`, `request_rate`, `active_connections`, `connection_utilization`) with deterministic noise and configurable cascade injections.
- **Data Quality Validator (`app.data.validation`)**: Validates schema compliance, type stability, metric range boundaries, monotonic timestamps, and ground-truth metadata invariants.
- **Preprocessing & Feature Engineering (`app.data.preprocess`, `app.data.features`)**: Handles missing value imputations, rolling window metrics (mean, std, min, max, delta), EWMA smoothing, and service-relative z-score standardization without future leakage.
- **Anomaly Detectors (`app.models.anomaly_baseline`, `app.models.anomaly_isolation_forest`)**:
  - Trailing window Z-score statistical baseline (Precision: 45.24%, Recall: 20.43%, F1: 0.2815).
  - Multivariate Isolation Forest ensemble (Precision: 71.56%, Recall: 83.87%, F1: 0.7723).
- **Incident Prediction Baseline (`app.models.incident_predictor`)**: Scikit-Learn logistic pipeline predicting incident probabilities within a forward-looking 30-minute window with a mean lead time of ~27.2 minutes.
- **Severity Classifier (`app.models.severity_classifier`)**: Calibrated multi-class classification model categorizing system degradations into `low`, `medium`, `high`, and `critical`.
- **RCA Reasoning Engine (`app.models.rca_engine`)**: Scores root cause candidates by combining temporal anomaly onset order ($40\%$), metric anomaly amplitude ($40\%$), and topological dependency graph depth ($20\%$).
- **Natural Language Explainer (`app.models.rca_explainer`)**: Generates deterministic, human-readable post-mortem summaries, blast radius breakdowns, causal metric traces, and prioritized remediation actions.
- **MLOps & Continuous Monitoring (`app.mlops`)**: Automated drift detection (Kolmogorov-Smirnov test), missing rate alerts, schema validation, latency profiling, and artifact inventory tracking.
- **REST API (`app.api`)**: High-performance FastAPI server with structured Pydantic schemas, error handling, and CORS support.
- **Interactive Operations UI (`frontend/static`)**: Web dashboard for inspecting live telemetry, running on-demand anomaly detection, evaluating incident risks, reviewing RCA graphs, and monitoring MLOps system health.

---

## 5. Repository Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI pipeline
├── app/
│   ├── __init__.py
│   ├── config.py                      # Central configuration & path constants
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI application & static route mounting
│   │   ├── routes.py                  # API endpoints (telemetry, models, RCA, MLOps)
│   │   └── schemas.py                 # Pydantic request/response data contracts
│   ├── data/
│   │   ├── __init__.py
│   │   ├── features.py                # Rolling feature engineering & transformations
│   │   ├── loader.py                  # Telemetry loading and batch chunking
│   │   ├── preprocess.py              # Cleaning, imputation, and time-sorting
│   │   ├── synthetic.py               # Deterministic telemetry & failure generator
│   │   └── validation.py              # Telemetry schema & quality validation
│   ├── mlops/
│   │   ├── __init__.py
│   │   ├── drift.py                   # Kolmogorov-Smirnov distribution drift testing
│   │   ├── experiment.py              # Run tracking and artifact metrics logger
│   │   └── monitoring.py              # Model health, data quality, & latency checks
│   └── models/
│       ├── __init__.py
│       ├── anomaly_baseline.py        # Statistical trailing Z-score detector
│       ├── anomaly_isolation_forest.py # Multivariate Isolation Forest detector
│       ├── incident_predictor.py      # Logistic regression incident risk predictor
│       ├── rca_engine.py              # Graph & temporal heuristic RCA engine
│       ├── rca_explainer.py           # Natural-language explanations & runbooks
│       └── severity_classifier.py     # Multi-class severity classification baseline
├── data/
│   ├── experiments/                   # Final evaluation metric outputs (JSON)
│   ├── models/                        # Serialized model artifacts (.joblib)
│   └── synthetic/                     # Generated telemetry (.parquet) & incidents (.json)
├── docs/
│   ├── anomaly-results.md             # Anomaly detection evaluation report
│   ├── architecture.md                # System topology & failure cascade architecture
│   ├── data-contract.md               # Telemetry and incident JSON schema contracts
│   ├── data-quality-report.md         # Baseline dataset verification report
│   ├── demo-guide.md                  # Interactive demo & step-by-step walkthrough
│   ├── development.md                 # Developer guide and workflow standards
│   ├── evaluation-plan.md             # Evaluation metrics and benchmark targets
│   ├── final-evaluation.md            # Comprehensive final benchmark report
│   ├── incident-prediction-results.md # Incident prediction & severity report
│   ├── integration-results.md         # End-to-end integration test report
│   ├── mlops.md                       # MLOps architecture & monitoring design
│   ├── mlops-results.md               # Drift and health monitoring results
│   ├── portfolio-evidence.md          # Technical portfolio artifacts & decisions
│   ├── problem-statement.md           # Problem framing and scenario definition
│   ├── rca-results.md                 # Root cause analysis benchmark report
│   └── security-reliability.md        # Security posture, bounds, & integrity safeguards
├── evaluation/
│   ├── anomaly_evaluation.py          # Anomaly benchmark harness
│   ├── incident_evaluation.py         # Prediction & severity benchmark harness
│   ├── rca_evaluation.py              # RCA ranking benchmark harness
│   ├── run_final_evaluation.py        # Comprehensive multi-layer evaluation runner
│   ├── run_mlops_monitoring.py        # MLOps drift & health runner
│   └── run_rca_eval.py                # Standalone RCA evaluation CLI
├── frontend/
│   └── static/
│       ├── index.html                 # Single-page operations dashboard UI
│       ├── app.js                     # Dashboard interaction & API client logic
│       └── style.css                  # Dark-mode dashboard styling
├── scripts/
│   ├── generate_telemetry.py          # CLI to generate synthetic telemetry
│   ├── train_models.py                # CLI to train & serialize all models
│   └── validate_telemetry.py          # CLI to validate dataset quality
├── tests/
│   ├── conftest.py                    # Pytest fixtures and mock telemetry
│   ├── api/
│   │   └── test_routes.py             # FastAPI endpoint integration tests
│   └── unit/
│       ├── test_anomaly_baseline.py   # Statistical detector unit tests
│       ├── test_anomaly_evaluation.py # Anomaly evaluation harness tests
│       ├── test_anomaly_isolation_forest.py # Isolation Forest unit tests
│       ├── test_config.py             # Configuration tests
│       ├── test_drift.py              # Drift detection unit tests
│       ├── test_experiment.py         # MLOps experiment tracking tests
│       ├── test_features.py           # Feature engineering tests
│       ├── test_final_evaluation.py   # Final evaluation runner unit tests
│       ├── test_incident_evaluation.py # Incident evaluation unit tests
│       ├── test_incident_predictor.py # Incident predictor unit tests
│       ├── test_integration_e2e.py    # End-to-end pipeline integration tests
│       ├── test_loader.py             # Telemetry loader unit tests
│       ├── test_monitoring.py         # MLOps monitoring unit tests
│       ├── test_preprocess.py         # Preprocessing unit tests
│       ├── test_rca_engine.py         # RCA engine unit tests
│       ├── test_rca_evaluation.py     # RCA evaluation unit tests
│       ├── test_rca_explainer.py      # RCA explainer unit tests
│       ├── test_scripts.py            # CLI scripts unit tests
│       ├── test_severity_classifier.py # Severity classifier unit tests
│       ├── test_synthetic.py          # Synthetic generator unit tests
│       └── test_validation.py         # Data validator unit tests
├── .gitignore
├── pyproject.toml                     # PEP 517/621 project configuration & dependencies
└── requirements.txt                   # Production & test dependencies
```

---

## 6. Quick Start Guide

### Prerequisites
- Python `3.10`, `3.11`, or `3.12`.

### Step 1: Environment Setup & Installation
```bash
# Clone the repository
git clone https://github.com/example/aiops-incident-intelligence.git
cd aiops-incident-intelligence

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode with dependencies
pip install -e ".[test]"
# Or via requirements.txt:
# pip install -r requirements.txt
```

### Step 2: Generate Synthetic Telemetry
Generate 1 hour of synthetic telemetry with a database connection pool saturation incident injected:
```bash
python scripts/generate_telemetry.py --duration 3600 --interval 10 --scenario database_connection_saturation
```

### Step 3: Validate Telemetry Quality
Verify schema invariants, temporal monotonicity, and metric range constraints:
```bash
python scripts/validate_telemetry.py
```

### Step 4: Train Machine Learning Models
Fit the Isolation Forest detector, Incident Predictor, and Severity Classifier, exporting artifacts to `data/models/`:
```bash
python scripts/train_models.py
```

### Step 5: Start the REST API & Web Dashboard
Launch the FastAPI server:
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Operations Dashboard**: Open [http://localhost:8000](http://localhost:8000) in your browser.
- **Interactive Swagger API Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs).
- **Alternative ReDoc Docs**: Open [http://localhost:8000/redoc](http://localhost:8000/redoc).
- **Interactive Demo Walkthrough**: See [`docs/demo-guide.md`](docs/demo-guide.md) for a step-by-step cURL and dashboard guide.
- **Portfolio Evidence & Engineering Decisions**: See [`docs/portfolio-evidence.md`](docs/portfolio-evidence.md).
- **Security & Reliability Posture**: See [`docs/security-reliability.md`](docs/security-reliability.md).

### Step 6: Run Comprehensive Benchmark Evaluation
Run the full evaluation harness and output formal metrics:
```bash
python evaluation/run_final_evaluation.py
```

### Step 7: Run Automated Test Suite
Execute the comprehensive test suite (137 unit and API integration tests):
```bash
pytest -v
```

---

## 7. REST API Reference

The FastAPI service exposes structured endpoints under the `/api/v1` namespace:

| Endpoint | Method | Description | Example Input / Output |
| :--- | :--- | :--- | :--- |
| `/api/v1/health` | `GET` | Service liveness probe | `{"status": "ok", "service": "omniroute-aiops"}` |
| `/api/v1/telemetry/latest` | `GET` | Retrieve latest telemetry records | Query `limit=500` $\to$ List of telemetry rows |
| `/api/v1/telemetry/summary` | `POST` | Aggregated statistical summary per service | Telemetry records batch $\to$ `record_count`, `services`, `metrics` |
| `/api/v1/anomalies/detect` | `POST` | Multivariate anomaly detection | Telemetry batch $\to$ `anomaly_count`, per-row `is_anomaly`, `anomaly_score` |
| `/api/v1/incidents/predict` | `POST` | Early-warning incident probability | Telemetry batch $\to$ `incident_probability`, `predicted_incident` |
| `/api/v1/incidents/severity` | `POST` | Incident severity classification | Telemetry batch $\to$ `predicted_severity` (`low`, `medium`, `high`, `critical`) |
| `/api/v1/rca/rank` | `POST` | Ranked root cause candidates | Telemetry batch + `incident_start_time` $\to$ Ranked services with scores |
| `/api/v1/rca/explain` | `POST` | Natural language explanation & runbook | Telemetry batch + incident $\to$ Root cause, blast radius, actions |
| `/api/v1/mlops/status` | `GET` | Artifact inventory and sizes | `{"artifacts": {"isolation_forest_latest.joblib": {...}}}` |
| `/api/v1/mlops/health` | `GET` | Live data quality & drift check | `{"status": "healthy", "drift_detected": false, "metrics": {...}}` |

---

## 8. Interactive Operations Dashboard

The built-in web dashboard (`/static/index.html`) provides operations teams with a unified single-pane interface:
- **Telemetry Monitor**: Live view of streaming microservice metrics (latency, error rate, CPU, active connections).
- **Anomaly Detection Panel**: Real-time identification of metric outliers and anomaly scores.
- **Incident Prediction Radar**: Early-warning probability gauge alerting operators ahead of SLA breaches.
- **RCA & Incident Explainer**: Top-ranked root-cause service diagnosis, causal metric contributions, blast radius breakdown, and actionable remediation runbooks.
- **MLOps Health & Drift Center**: Model status, dataset health checks, distribution drift warnings, and latency monitoring.

---

## 9. Project Roadmap & Milestone Status

| Milestone | Description | Status |
| :--- | :--- | :--- |
| **Phase 0** | Problem framing, data contracts, architecture design, evaluation plan, synthetic data generator | **Completed** |
| **Phase 1** | Repository structure, project configuration, CI pipeline, developer guide | **Completed** |
| **Phase 2** | Preprocessing pipeline, feature engineering, windowing, and causal validation | **Completed** |
| **Phase 3** | Anomaly detection engines (Z-score baseline & multivariate Isolation Forest) | **Completed** |
| **Phase 4** | Incident prediction (early warning) & multi-class severity classification | **Completed** |
| **Phase 5** | Graph topology & temporal RCA engine + Natural Language Explainer & runbooks | **Completed** |
| **Phase 6** | MLOps drift detection (KS-test), experiment tracking, and automated model monitoring | **Completed** |
| **Phase 7** | FastAPI backend integration, REST contract alignment, and interactive web dashboard | **Completed** |
| **Phase 8** | Final benchmark evaluation, comprehensive reporting, codebase cleanup, and handover | **Completed** |

---

## 10. License

This project is licensed under the MIT License. See `LICENSE` for details.
