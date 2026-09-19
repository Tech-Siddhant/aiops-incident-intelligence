# System Architecture: AIOps Incident Intelligence

## 1. Overview

AIOps Incident Intelligence is an end-to-end operational intelligence system designed to ingest continuous distributed microservice telemetry, detect anomalous degradation, forecast impending service failures, classify incident severity, and rank probable root-cause contributors with deterministic evidence.

```
+-----------------------------------------------------------------------------------+
|                            INGESTION & DATA LAYER                                 |
|                                                                                   |
|  Synthetic / Real-world Telemetry Stream (Parquet / In-Memory JSON)              |
|  [api_gateway, auth_service, orders_service, database]                            |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                    PREPROCESSING & CAUSAL FEATURE PIPELINE                        |
|                                                                                   |
|  - Strict Chronological Splitting (Train: 60%, Val: 20%, Test: 20%)              |
|  - Causal Trailing Rolling Features (closed="right", zero future leakage)        |
|  - Rolling Stats (1m, 2m, 5m, 10m windows: mean, max, std, EWMA)                 |
|  - Cross-Service Correlation & Propagation Deltas                                |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                            AIOps ML & CAUSAL ENGINE                               |
|                                                                                   |
|  +--------------------+  +----------------------+  +---------------------------+  |
|  | Anomaly Detection  |  | Incident Prediction  |  | Severity Classification   |  |
|  | (Isolation Forest  |  | (Temporal Logistic   |  | (Multi-class Classifier   |  |
|  |  vs Rolling Z-Score|  |  Regression / GBDT)  |  |  Low, Med, High, Critical)|  |
|  +--------------------+  +----------------------+  +---------------------------+  |
|                                     |                                             |
|                                     v                                             |
|                         +-----------------------+                                 |
|                         |      RCA Engine       |                                 |
|                         | (Heuristic Topology   |                                 |
|                         |  Graph & Anomaly Rank)|                                 |
|                         +-----------------------+                                 |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                      EVIDENCE & EXPLANATION GENERATOR                             |
|                                                                                   |
|  - Ranked Probable Contributors (Top-1, Top-3 with Confidence Scores)             |
|  - Telemetry Deviation Markers (Z-scores, delta % above baseline)                |
|  - Human-readable Operational Summary + Technical Diagnostic Feature Vectors      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                           SERVING & PRESENTATION LAYER                            |
|                                                                                   |
|  FastAPI Application (Port 8000)                                                  |
|  - REST Endpoints (/health, /telemetry, /anomalies, /incidents, /rca, /mlops)    |
|                                                                                   |
|  Single-Page Application (HTML5 / Vanilla CSS / Modern ES6 JS)                   |
|  - Default View: Clear operational explanations (What, Where, Severity, Action)  |
|  - Technical Toggle: Model scores, raw features, thresholds, drift metrics        |
+-----------------------------------------------------------------------------------+
```

---

## 2. Microservice Reference Topology

The system evaluates telemetry generated across a four-service production topology:

```
      [ Upstream Clients / Users ]
                   │
                   ▼
         ┌───────────────────┐
         │    api_gateway    │ (Routing, rate limiting, SSL termination)
         └─────────┬─────────┘
                   ├───► [ auth_service ] (Token verification / Negative-Control)
                   │
                   ▼
         ┌───────────────────┐
         │  orders_service   │ (Business logic, transaction orchestration)
         └─────────┬─────────┘
                   │
                   ▼
         ┌───────────────────┐
         │     database      │ (PostgreSQL connection pool & storage engine)
         └───────────────────┘
```

### Reference Failure Propagation: Database Connection Saturation
1. **Database Layer ($t_0$)**: Connection pool exhausts (`connection_utilization` $\to$ 1.0, `active_connections` rises), query wait queue builds up, database `latency_ms` spikes.
2. **Orders Service Layer ($t_0 + \Delta_1$)**: Blocked on query responses; service `latency_ms` increases, `error_rate` begins rising due to database query timeouts.
3. **API Gateway Layer ($t_0 + \Delta_2$)**: Upstream order requests stall; gateway `latency_ms` rises, HTTP 504/500 error rates spike.
4. **Auth Service (Negative-Control)**: Operates independently with minimal or normal baseline variation.

---

## 3. Core Subsystems

### 3.1 Ingestion & Preprocessing
- Telemetry format: In-memory streaming JSON or Parquet time-series records.
- Metrics ingested per service: `latency_ms`, `error_rate`, `cpu_usage_pct`, `memory_usage_pct`, `disk_usage_pct`, `request_rate_rps`, `active_connections`, `connection_utilization`.
- Causal feature extraction: strictly trailing rolling aggregations (`closed="right"` in pandas). Scaler transformations fitted exclusively on the training split to guarantee zero future-leakage.

### 3.2 AIOps ML Engine
- **Anomaly Detection**: Scikit-Learn Isolation Forest trained on normal operational baselines with a 5% contamination target, evaluated against a rolling 12-sample Z-Score baseline.
- **Incident Prediction**: Supervised classifier predicting whether a service incident will occur within a forward 30-minute horizon.
- **Severity Classification**: Multi-class model mapping telemetry feature amplitudes and error rates into operational severity tiers (`low`, `medium`, `high`, `critical`).
- **RCA Engine**: Topology-aware heuristic graph engine that traces failure propagation backward from affected downstream nodes to identify the most likely upstream contributor. Framed strictly as *ranked probable contributors*, avoiding causal overclaiming.

### 3.3 API Layer
- Built with **FastAPI** and **Pydantic v2**.
- Delivers sub-100ms response times for all analytical endpoints.
- Endpoints:
  - `GET /health` - System liveness & readiness check.
  - `GET /api/v1/telemetry/latest` - Latest metric snapshots across microservices.
  - `POST /api/v1/anomalies/detect` - Real-time anomaly detection scoring.
  - `POST /api/v1/incidents/predict` - Early-warning failure forecast.
  - `POST /api/v1/incidents/severity` - Multi-class incident severity categorization.
  - `POST /api/v1/rca/rank` - Ranked probable contributor list.
  - `POST /api/v1/rca/explain` - Natural language operational & technical evidence.
  - `GET /api/v1/mlops/health` - Model monitoring, data drift (PSI, KS-test), and pipeline latency.

### 3.4 Presentation Layer
- High-performance, zero-framework Vanilla JS/CSS Single-Page Application.
- Progressive disclosure UX:
  - **Default View**: Actionable, human-readable operational answers ("What is wrong?", "Where?", "Condition?", "Recommended Action?").
  - **Technical Details Toggle**: Reveals underlying model versions, raw anomaly scores, feature weight contributions, thresholds, and statistical diagnostics.
