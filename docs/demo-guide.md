# Interactive Demonstration & Operational Walkthrough Guide

This guide provides a step-by-step walkthrough for demonstrating the **OmniRoute AIOps Incident Intelligence** platform to engineering teams, stakeholders, and interviewers.

---

## 1. Demo Flow Overview

The platform demonstrates the complete lifecycle of microservice telemetry intelligence across seven stages:

```
[ Telemetry Stream ]
         │
         ▼
[ 1. Anomaly Detection ] ──► (Multivariate Isolation Forest detects metric deviations)
         │
         ▼
[ 2. Incident Prediction ] ──► (Logistic Regression forecasts failure risk 27m ahead)
         │
         ▼
[ 3. Severity Classifier ] ──► (Categorizes blast radius & degradation intensity)
         │
         ▼
[ 4. Root Cause Analysis ] ──► (Topology DAG & temporal onset isolates root service)
         │
         ▼
[ 5. Natural Language Explainer ] ──► (Produces actionable runbooks & mitigation steps)
         │
         ▼
[ 6. Single-Pane Dashboard ] ──► (Visualizes live telemetry, radar, and diagnosis)
```

---

## 2. Quickstart Environment Setup

### 1. Launch the FastAPI Service
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Verify Service Health
```bash
curl -s http://localhost:8000/api/v1/health | jq .
```
**Expected Output:**
```json
{
  "status": "ok",
  "service": "omniroute-aiops",
  "version": "0.1.0"
}
```

---

## 3. End-to-End Walkthrough Steps

### Step 1: Telemetry Generation & Quality Validation

Generate a deterministic cascading failure scenario (`database_connection_saturation`, 15-minute window, 4 services):
```bash
python -m scripts.generate_telemetry --scenario database_connection_saturation --duration 900
```
Validate the telemetry schema and distribution constraints:
```bash
python -m scripts.validate_telemetry
```
**Expected Output:**
```
=== Validation Report ===
Status: PASSED
Total Rows: 1440
Missing Columns: 0
Null Count: 0
```

---

### Step 2: Anomaly Detection (`/api/v1/anomalies/detect`)

Send a telemetry batch to evaluate real-time multi-dimensional metric anomalies.

```bash
curl -X POST http://localhost:8000/api/v1/anomalies/detect \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "timestamp": "2026-01-01T00:15:30Z",
        "service_name": "database",
        "cpu_usage_pct": 88.5,
        "memory_usage_pct": 74.2,
        "disk_usage_pct": 55.0,
        "network_in_mbps": 12.4,
        "network_out_mbps": 45.1,
        "request_rate_rps": 120.0,
        "error_rate": 0.045,
        "latency_ms": 340.5,
        "active_connections": 100,
        "connection_utilization": 1.0
      }
    ]
  }' | jq .
```

**Key Response Fields:**
- `anomaly_count`: Number of records flagged as anomalous.
- `anomalies[].is_anomaly`: Boolean indicator (`true`).
- `anomalies[].anomaly_score`: Continuous anomaly score $[0.0, 1.0]$.

---

### Step 3: Early-Warning Incident Prediction (`/api/v1/incidents/predict`)

Evaluate whether telemetry patterns signal an impending service failure within the next 30-minute predictive horizon.

```bash
curl -X POST http://localhost:8000/api/v1/incidents/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "timestamp": "2026-01-01T00:10:00Z",
        "service_name": "database",
        "cpu_usage_pct": 65.0,
        "memory_usage_pct": 60.0,
        "disk_usage_pct": 45.0,
        "network_in_mbps": 10.0,
        "network_out_mbps": 30.0,
        "request_rate_rps": 95.0,
        "error_rate": 0.005,
        "latency_ms": 48.0,
        "active_connections": 82,
        "connection_utilization": 0.82
      }
    ]
  }' | jq .
```

**Key Response Fields:**
- `predictions[].incident_probability`: Continuous failure probability (e.g. `0.784`).
- `predictions[].is_predicted_incident`: Early-warning flag (`true`).
- **Operational Benefit**: Provides an average of **27.2 minutes of lead time** before SLA breach.

---

### Step 4: Incident Severity Classification (`/api/v1/incidents/severity`)

Categorize the degradation level into operational tiers (`low`, `medium`, `high`, `critical`).

```bash
curl -X POST http://localhost:8000/api/v1/incidents/severity \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "timestamp": "2026-01-01T00:16:00Z",
        "service_name": "orders_service",
        "cpu_usage_pct": 92.0,
        "memory_usage_pct": 80.0,
        "disk_usage_pct": 40.0,
        "network_in_mbps": 25.0,
        "network_out_mbps": 50.0,
        "request_rate_rps": 80.0,
        "error_rate": 0.08,
        "latency_ms": 620.0,
        "active_connections": 95,
        "connection_utilization": 0.95
      }
    ]
  }' | jq .
```

**Key Response Fields:**
- `severities[].predicted_severity`: Classified severity tier (`"critical"`).

---

### Step 5: Root Cause Analysis Ranking (`/api/v1/rca/rank`)

Send a multi-service incident telemetry batch to identify and rank candidate root-cause services.

```bash
curl -X POST http://localhost:8000/api/v1/rca/rank \
  -H "Content-Type: application/json" \
  -d '{
    "incident_start_time": "2026-01-01T00:15:00Z",
    "records": [
      {
        "timestamp": "2026-01-01T00:15:10Z",
        "service_name": "database",
        "cpu_usage_pct": 85.0, "memory_usage_pct": 70.0, "disk_usage_pct": 50.0,
        "network_in_mbps": 10.0, "network_out_mbps": 40.0, "request_rate_rps": 100.0,
        "error_rate": 0.04, "latency_ms": 320.0, "active_connections": 100, "connection_utilization": 1.0
      },
      {
        "timestamp": "2026-01-01T00:15:30Z",
        "service_name": "orders_service",
        "cpu_usage_pct": 75.0, "memory_usage_pct": 65.0, "disk_usage_pct": 40.0,
        "network_in_mbps": 15.0, "network_out_mbps": 30.0, "request_rate_rps": 90.0,
        "error_rate": 0.06, "latency_ms": 480.0, "active_connections": 70, "connection_utilization": 0.70
      },
      {
        "timestamp": "2026-01-01T00:15:50Z",
        "service_name": "api_gateway",
        "cpu_usage_pct": 60.0, "memory_usage_pct": 50.0, "disk_usage_pct": 30.0,
        "network_in_mbps": 20.0, "network_out_mbps": 20.0, "request_rate_rps": 110.0,
        "error_rate": 0.05, "latency_ms": 510.0, "active_connections": 50, "connection_utilization": 0.50
      },
      {
        "timestamp": "2026-01-01T00:15:10Z",
        "service_name": "auth_service",
        "cpu_usage_pct": 25.0, "memory_usage_pct": 30.0, "disk_usage_pct": 20.0,
        "network_in_mbps": 5.0, "network_out_mbps": 5.0, "request_rate_rps": 50.0,
        "error_rate": 0.001, "latency_ms": 12.0, "active_connections": 10, "connection_utilization": 0.10
      }
    ]
  }' | jq .
```

**Key Response Fields:**
- `top_1.service`: `"database"`
- `top_1.score`: `0.3881`
- `top_1.confidence`: `"MEDIUM"`
- `ranked_candidates`: Full list of services ranked by composite causal weight.

---

### Step 6: Natural Language Explanation & Runbook (`/api/v1/rca/explain`)

Generate a human-readable engineering post-incident brief and remediation runbook.

```bash
curl -X POST http://localhost:8000/api/v1/rca/explain \
  -H "Content-Type: application/json" \
  -d '{
    "incident_start_time": "2026-01-01T00:15:00Z",
    "records": [
      {
        "timestamp": "2026-01-01T00:15:10Z",
        "service_name": "database",
        "cpu_usage_pct": 85.0, "memory_usage_pct": 70.0, "disk_usage_pct": 50.0,
        "network_in_mbps": 10.0, "network_out_mbps": 40.0, "request_rate_rps": 100.0,
        "error_rate": 0.04, "latency_ms": 320.0, "active_connections": 100, "connection_utilization": 1.0
      },
      {
        "timestamp": "2026-01-01T00:15:30Z",
        "service_name": "orders_service",
        "cpu_usage_pct": 75.0, "memory_usage_pct": 65.0, "disk_usage_pct": 40.0,
        "network_in_mbps": 15.0, "network_out_mbps": 30.0, "request_rate_rps": 90.0,
        "error_rate": 0.06, "latency_ms": 480.0, "active_connections": 70, "connection_utilization": 0.70
      }
    ]
  }' | jq .
```

**Sample Output Structure:**
```json
{
  "probable_root_cause": "database",
  "confidence": "MEDIUM",
  "summary": "Root cause identified as database (Score: 0.3881, Confidence: MEDIUM).",
  "top_1": {
    "candidate_service": "database",
    "dependency_evidence": "Topology role: Leaf downstream dependency; callers: ['orders_service']",
    "contributing_metrics": [
      "Severe connection saturation: peak 100.0%",
      "Elevated error rate: peak 4.00%"
    ],
    "temporal_evidence": "First detected anomaly at t0+10s"
  },
  "recommended_actions": [
    "Scale database connection pool or terminate idle transactions.",
    "Inspect slow queries locking connection slots on database.",
    "Verify circuit breaker thresholds between orders_service and database."
  ]
}
```

---

### Step 7: Single-Pane Interactive Dashboard Walkthrough

Navigate to `http://localhost:8000/static/index.html` in your web browser:

1. **System Status Header**: Displays live API liveness, active service count (4), and latest telemetry timestamp.
2. **Telemetry Streaming Chart**: Visualizes real-time metric traces (`latency_ms`, `error_rate`, `connection_utilization`) across all services.
3. **Anomaly Monitor**: Highlights flagged outliers with individual row scores.
4. **Early-Warning Radar**: Displays continuous probability meter forecasting failures 27 minutes in advance.
5. **RCA Diagnosis & Runbook Panel**: Displays the ranked root-cause hierarchy, evidence breakdown, and actionable on-call remediation steps.
6. **MLOps Drift Center**: Displays model artifact sizes and automated Kolmogorov-Smirnov drift indicators.

---

### Step 8: Live MLOps Health & Drift Check (`/api/v1/mlops/health`)

Verify live data quality, distribution stability, and latency tracking:
```bash
curl -s http://localhost:8000/api/v1/mlops/health | jq .
```
**Expected Response:**
```json
{
  "status": "healthy",
  "drift_detected": false,
  "metrics": {
    "active_connections_ks_stat": 0.042,
    "latency_ms_ks_stat": 0.038
  }
}
```
