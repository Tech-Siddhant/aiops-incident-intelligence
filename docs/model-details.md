# Machine Learning & Causal Engine Details

## 1. Pipeline Overview

The AIOps Incident Intelligence engine integrates four complementary analytical models that operate sequentially across continuous microservice telemetry streams:

```
Telemetry Record
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Feature Engineering (Trailing Rolling Windows, EWMA)   │
└─────┬──────────────────────────────────────────────────┘
      │
      ├──────────────────────┐
      ▼                      ▼
┌──────────────────┐   ┌───────────────────────────┐
│ Anomaly Detector │   │ Early Incident Predictor  │
│ (Isolation Forest│   │ (Temporal Classification, │
│  vs Z-Score)     │   │  30-min forward horizon)  │
└─────┬────────────┘   └─────┬─────────────────────┘
      │                      │
      └──────────┬───────────┘
                 ▼
┌───────────────────────────────────┐
│ Severity Classifier               │
│ (Low, Medium, High, Critical)     │
└────────────────┬──────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│ Topology-Aware RCA Graph Engine   │
│ (Ranked Probable Contributors)    │
└────────────────┬──────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│ Evidence & Explanation Generator  │
└───────────────────────────────────┘
```

---

## 2. Model Specifications

### 2.1 Anomaly Detection: Isolation Forest
- **Objective**: Identify multi-dimensional deviation from normal system operation across distributed services.
- **Algorithm**: `sklearn.ensemble.IsolationForest`
- **Hyperparameters**:
  - `n_estimators`: 100
  - `contamination`: 0.05 (5% expected anomaly rate under normal conditions)
  - `max_features`: 1.0
  - `random_state`: 42
- **Input Features**: `latency_ms`, `error_rate`, `cpu_usage_pct`, `memory_usage_pct`, `disk_usage_pct`, `connection_utilization`, `request_rate_rps`, `active_connections`.
- **Artifact Path**: `data/models/isolation_forest_latest.joblib` (Size: ~1.2 MB)
- **Baseline Comparison**: Rolling 12-sample Z-Score baseline ($\mu \pm 3\sigma$). The baseline suffers from normalization of deviance (Recall drops to 20.43% during persistent failures), whereas the Isolation Forest sustains an 83.87% recall across the incident lifecycle.

### 2.2 Incident Prediction: Temporal Early-Warning Model
- **Objective**: Predict whether an operational incident will occur within a 30-minute forward lookahead window ($t \in [t+1, t+180]$ steps at 10s intervals).
- **Algorithm**: `sklearn.linear_model.LogisticRegression` with calibrated probabilities.
- **Training Strategy**: Strict chronological partitioning:
  - Train Split: 0% to 60%
  - Validation Split: 60% to 80% (threshold tuning)
  - Test Split: 80% to 100% (final unseen evaluation)
- **Engineered Features**:
  - Rolling mean, max, and standard deviation over 12 samples (2-minute window) and 60 samples (10-minute window).
  - Cross-service metric deltas: gateway-to-order latency propagation ratio, database connection saturation differential.
- **Artifact Path**: `data/models/incident_predictor_latest.joblib` (Size: ~450 KB)
- **Key Measured Results**: ROC-AUC 0.8130, PR-AUC 0.7599, Precision 63.08%, Recall 68.05%, Mean Lead Time 1,632.5s (~27.2 minutes).

### 2.3 Incident Severity Classifier
- **Objective**: Categorize incoming incident state into four standardized operational tiers: `low`, `medium`, `high`, `critical`.
- **Algorithm**: Multi-class supervised classifier trained on composite telemetry stress indicators (error rate thresholds, SLA breach duration, downstream blast radius).
- **Artifact Path**: `data/models/severity_classifier_latest.joblib` (Size: ~580 KB)
- **Operational Tiers**:
  - `low`: Isolated single-service jitter, error rate $< 1\%$, latency $< 1.5\times$ baseline.
  - `medium`: Moderate service degradation, error rate $1\% - 5\%$, upstream queues beginning to build.
  - `high`: Cross-service SLA violations, error rate $5\% - 15\%$, end-user visible slowness.
  - `critical`: Cascading failure, upstream 5xx spikes $> 15\%$, connection pool exhaustion.

### 2.4 Root Cause Analysis (RCA) Engine
- **Objective**: Isolate the most probable root-cause service and metric driver from a cascading multi-service failure.
- **Architecture**: Heuristic topology-directed dependency graph traversal combined with temporal onset scoring:
  1. **Topology Mapping**: Traverses dependency edges (`api_gateway` $\to$ `orders_service` $\to$ `database`).
  2. **Temporal Onset Tracking**: Evaluates the earliest timestamp of statistically significant anomaly onset ($\tau_{\text{onset}}$). The service exhibiting the earliest sustained degradation along the dependency path receives the highest causal weight.
  3. **Metric Contribution Weighting**: Ranks internal metric deviation (e.g., `connection_utilization` z-score vs `latency_ms` z-score).
- **Output Structure**:
  - `most_likely_contributor`: Identified service node and primary metric driver.
  - `ranked_candidates`: Ordered list of all services with individual confidence percentages.
  - `deterministic_explanation`: Clear natural language description of how failure propagated.
- **Causal Framing**: The output is explicitly framed as **"Ranked Probable Contributors"**. It highlights correlation and temporal precedence, rather than making unverified counterfactual claims.

---

## 3. MLOps Monitoring & Drift Detection

The platform includes embedded statistical drift monitoring (`app/mlops/monitoring.py`):
- **Population Stability Index (PSI)**: Measures distribution shifts between baseline training data and live production telemetry.
  - $\text{PSI} < 0.10$: No significant shift.
  - $0.10 \le \text{PSI} < 0.25$: Moderate drift (warning alert).
  - $\text{PSI} \ge 0.25$: Significant drift (retraining triggered).
- **Two-Sample Kolmogorov-Smirnov (KS) Test**: Evaluates continuous feature distribution similarity non-parametrically.
- **Execution**: Computes on demand via `GET /api/v1/mlops/health` within 180ms without external telemetry dependencies.
