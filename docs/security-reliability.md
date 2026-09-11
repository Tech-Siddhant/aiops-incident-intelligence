# Security, Reliability & Data Integrity Architecture

This document details the security posture, operational reliability guarantees, and data integrity safeguards implemented across the **OmniRoute AIOps Incident Intelligence** platform.

---

## 1. Security Architecture & Threat Model

### 1.1 Input Boundary Validation & Schema Enforcement
All ingress data paths (REST endpoints and CLI entry points) enforce strict structural and type validation using **Pydantic v2**:
- **Timestamp Validation**: Strict parsing of ISO-8601 UTC timestamps (`YYYY-MM-DDTHH:MM:SSZ` or ISO format). Malformed or unparseable timestamps trigger HTTP 422 immediately.
- **Metric Domain Bounds**: Telemetry features are guarded with non-negative constraints and percentage ceilings:
  - `latency_ms >= 0.0`
  - `error_rate` bounded within $[0.0, 1.0]$
  - `cpu_usage_pct`, `memory_usage_pct`, `disk_usage_pct` bounded within $[0.0, 100.0]$
  - `connection_utilization` bounded within $[0.0, 1.0]$
  - `request_rate_rps >= 0.0`, `active_connections >= 0`
- **Payload Size Guards**: Batch requests reject empty payloads (`records=[]`) with `HTTP 400 Bad Request` and cap payload processing to bounded batch sizes to prevent memory exhaustion DoS vectors.

### 1.2 Zero Remote Data Exfiltration & Offline Execution
- **Self-Contained Local Runtime**: The platform requires zero external cloud network connections, SaaS telemetry backends, or third-party LLM API endpoints during runtime.
- **Deterministic Deserialization**: Model artifact loading uses scoped `joblib.load()` guarded within local trusted artifact directories (`data/models/`). Dynamic code execution (`eval`, `exec`) is strictly forbidden across all modules.

---

## 2. Zero Future-Data Leakage & Temporal Integrity

In AIOps time-series and incident prediction systems, data leakage across time splits creates unrealistically optimistic evaluation scores that fail in production. OmniRoute enforces three strict leakage guarantees:

```
Telemetry Stream: [ t_0 --------------------------> t_N ]
                  [=== Train (60%) ===][= Val (20%) =][= Test (20%) =]
                                       ▲              ▲
                                    Split 1        Split 2
```

1. **Chronological Splitting**: Continuous telemetry streams are partitioned strictly chronologically (60% Train, 20% Validation, 20% Test). No randomized $K$-fold cross-validation or shuffle-splitting is used anywhere in the pipeline.
2. **Causal Rolling Windows**: All feature engineering (trailing moving averages, EWMA, rolling max, rolling standard deviation) uses strictly causal, trailing windows (`closed="right"` in pandas). No centering or backward lookups (`shift(-k)`) are permitted in feature extraction.
3. **Scaler & Baseline Isolation**: `StandardScaler` transformations and statistical normalization parameters are fitted exclusively on the training partition ($0\% - 60\%$) and applied forward to validation and test partitions without refitting.

---

## 3. Fault Isolation & Blast Radius Verification

The reference microservice architecture explicitly separates coupled business paths from decoupled negative-control services:

```
      [ Clients / Users ]
               │
               ▼
     ┌───────────────────┐
     │    api_gateway    │
     └─────────┬─────────┘
               ├───► [ auth_service ] (Negative-Control / Isolated Boundary)
               │
               ▼
     ┌───────────────────┐
     │  orders_service   │
     └─────────┬─────────┘
               │
               ▼
     ┌───────────────────┐
     │     database      │ (Root Cause / Saturation Source)
     └───────────────────┘
```

- **Cascading Path**: `database` $\to$ `orders_service` $\to$ `api_gateway`.
- **Negative Control**: `auth_service` executes independently. During a simulated `database_connection_saturation` incident, `auth_service` remains within nominal baseline tolerances.
- **Blast Radius Assertion**: The Root Cause Analysis (RCA) engine tests whether non-impacted nodes like `auth_service` are correctly excluded from the blast radius and assigned minimal candidate ranking weights (measured: candidate score `0.1135`, confidence `LOW`).

---

## 4. Error Handling & Graceful Degradation

| Scenario | System Behavior | HTTP Status Code | Fallback Strategy |
| :--- | :--- | :--- | :--- |
| **Empty Telemetry Batch** | Immediate rejection | `400 Bad Request` | Returns informative error payload: `"Telemetry records list cannot be empty"` |
| **Malformed JSON / Field Typo** | Pydantic validation intercept | `422 Unprocessable Entity` | Details exact missing or malformed field name |
| **Missing Trained ML Artifact** | Fallback to heuristic baseline | `200 OK` (Logged Warning) | RCA Engine falls back to statistical peak & topology heuristics if Isolation Forest artifact is missing |
| **Insufficient Rolling Window** | Causal backfill / min_periods | `200 OK` | `min_periods=1` ensures valid feature generation even on single-record queries |
| **Drift Trigger During Surge** | Health monitoring warning | `200 OK` | `/api/v1/mlops/health` transitions status to `WARNING` with per-feature KS-test p-values without interrupting serving |

---

## 5. Operational Reliability & Resource Bounds

Empirical measurements gathered on standard single-core execution environments confirm predictable, lightweight resource utilization:

- **Peak Memory Overhead**: **306.28 KB** during batch feature extraction and multi-model inference (measured via Python `tracemalloc`).
- **Disk Footprint**: Total model storage is **2.20 MB (2,248 KB)** across all 3 trained models (`isolation_forest`, `incident_predictor`, `severity_classifier`).
- **Batch Processing Latency**: **144.45 ms** to process a full 720-record batch (4 services over 30 minutes) through the end-to-end pipeline.
- **Zero Daemon Dependencies**: Runs without requiring background daemon agents, external database servers, or message brokers for standard operation.
- **Deterministic Reproducibility**: All synthetic generation, dataset partitioning, and model training enforce fixed random seeds (`seed=42`).
