# AIOps Incident Intelligence

A lightweight, reproducible platform for microservice telemetry analysis, anomaly detection, incident prediction, and root-cause analysis (RCA).

---

## Overview & Problem Statement

Modern microservice architectures experience complex cascading failures where an issue in a deep downstream dependency (e.g., database connection pool exhaustion) manifests as elevated latencies, request queuing, and HTTP 5xx errors in upstream services.

This project builds an end-to-end incident intelligence system to:
1. **Detect Anomalies**: Continuously identify abnormal telemetry metrics across services.
2. **Predict Incidents**: Provide early warning before critical SLA breaches occur.
3. **Classify Severity**: Categorize incident impact based on degradation and blast radius.
4. **Isolate Root Causes**: Pinpoint the originating service and causal metrics.
5. **Recommend Remediation**: Generate actionable mitigation steps for on-call engineers.

---

## Current Status & Capabilities

- **Status**: **Phase 1 Complete (Foundation & Runtime Baseline)**.
- **Current Capabilities**:
  - Deterministic synthetic telemetry generator with cascading failure injection (`database_connection_saturation`).
  - Columnar data export (`telemetry.parquet`) and ground-truth incident metadata (`incidents.json`).
  - Automated telemetry data-quality validation layer enforcing schema, typing, range bounds, monotonicity, and temporal invariants.
  - Standardized project configuration (`pyproject.toml`) and unified unit testing suite.
- **Current Limitations**:
  - Initial telemetry is synthetic and models a 4-service reference topology.
  - ML anomaly detection, classification, and RCA models are scheduled for subsequent phases.

---

## High-Level Architecture & Topology

```
      [ Clients ]
           │
           ▼
    ┌─────────────┐
    │ api_gateway │
    └──────┬──────┘
           ├───► [ auth_service ]
           │
           ▼
    ┌────────────────┐
    │ orders_service │
    └──────┬─────────┘
           │
           ▼
    ┌──────────┐
    │ database │
    └──────────┘
```

**Failure Propagation Path (`database_connection_saturation`)**:
1. **$t_0$**: Database connection pool exhausts (`connection_utilization` $\to 1.0$), query latency spikes.
2. **$t_0 + 20\text{s}$**: Orders Service blocks on database queries; latency and error rates rise.
3. **$t_0 + 40\text{s}$**: API Gateway stalls on upstream orders requests; 504 timeouts and error rates spike.
4. **Auth Service**: Operates unaffected on an independent path.

---

## Repository Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml             # Lightweight GitHub Actions CI pipeline
├── app/
│   ├── __init__.py
│   ├── config.py              # Central path constants & deterministic defaults
│   └── data/
│       ├── __init__.py
│       ├── synthetic.py       # Deterministic telemetry & incident generation
│       └── validation.py      # Telemetry schema and quality validation
├── data/
│   └── synthetic/             # Telemetry (parquet) & incidents (json)
├── docs/
│   ├── architecture.md        # Topology & failure cascade architecture
│   ├── data-contract.md       # Telemetry and incident schemas
│   ├── data-quality-report.md # Data quality verification report
│   ├── development.md         # Developer guide and workflow standards
│   ├── evaluation-plan.md     # Evaluation metrics & benchmark targets
│   └── problem-statement.md   # Problem statement & failure scenario
├── scripts/
│   ├── generate_telemetry.py  # CLI to generate synthetic telemetry
│   └── validate_telemetry.py  # CLI to validate dataset quality
├── tests/
│   ├── conftest.py
│   └── unit/
│       ├── test_config.py     # Configuration unit tests
│       ├── test_synthetic.py  # Generator unit tests
│       └── test_validation.py # Data quality validation unit tests
├── .gitignore
├── pyproject.toml             # PEP 517/621 project configuration & pytest options
└── requirements.txt           # Minimal runtime & test requirements
```

---

## Quick Start

### 1. Installation
Requires Python `>=3.10` (recommended: `3.11` or `3.12`):
```bash
# Clone the repository
git clone https://github.com/example/aiops-incident-intelligence.git
cd aiops-incident-intelligence

# Install in editable mode with test dependencies
pip install -e ".[test]"
# Or via requirements.txt:
# pip install -r requirements.txt
```

### 2. Generate Synthetic Telemetry
```bash
python scripts/generate_telemetry.py --duration 3600 --interval 10 --scenario database_connection_saturation
```

### 3. Validate Telemetry Dataset
```bash
python scripts/validate_telemetry.py
```

### 4. Run Unit Tests
```bash
pytest -q
```

---

## Project Roadmap

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Problem framing, data contract, architecture, evaluation plan, synthetic data generator | **Completed** |
| **Phase 1** | Repository structure, project configuration, CI workflow, developer documentation | **Completed** |
| **Phase 2** | Telemetry data pipeline, feature engineering, windowing, preprocessing | *Upcoming* |
| **Phase 3** | Anomaly detection engine (multivariate metric anomaly identification) | *Upcoming* |
| **Phase 4** | Root Cause Analysis (RCA) & incident intelligence reasoning | *Upcoming* |
| **Phase 5** | End-to-end evaluation, benchmarking, and operational readiness | *Upcoming* |

