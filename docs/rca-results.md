# Phase 5.3: RCA Evaluation & Failure Analysis Results

## Objective
Evaluate whether the Root Cause Analysis (RCA) Engine correctly ranks the known root cause (`database`) during the synthetic `database_connection_saturation` incident, strictly utilizing information available at diagnosis time.

## Evaluation Setup
- **Data Source**: Deterministic synthetic telemetry generator (`app/data/synthetic.py`).
- **Incident Scenario**: `database_connection_saturation` spanning from `2026-01-01T00:15:00Z` to `2026-01-01T00:25:00Z`.
- **Topological Reality**: 
  - `database` (affected, source)
  - `orders_service` (affected, direct caller cascade)
  - `api_gateway` (affected, indirect caller cascade)
  - `auth_service` (unaffected)
- **Anomaly Signals**: Extracted using the pre-trained `IsolationForestDetector` acting upon data exclusively ahead of the incident horizon (zero future-leakage).

## Actual Ranking Performance (Heuristic)

| Metric | Result |
| :--- | :--- |
| **Top-1 Accuracy** | 100% (1.0) |
| **Top-3 Accuracy** | 100% (1.0) |
| **Mean Reciprocal Rank (MRR)** | 1.0 (Rank 1 hit) |
| **Valid Incident Evaluations** | 1 / 1 |

### Rank Output
1. **`database` (Score: 0.3881, Confidence: MEDIUM)** 
2. **`orders_service` (Score: 0.2520, Confidence: MEDIUM)**
3. **`api_gateway` (Score: 0.2464, Confidence: MEDIUM)**
4. **`auth_service` (Score: 0.1135, Confidence: LOW)**

## Explanation Quality Observations
The generated deterministic explanations accurately surfaced the evidence used by the heuristic ranking engine safely, distinguishing evidence from inference:
- **Language Safety**: The explanation explicitly refrained from claiming absolute resolution, outputting `"Classification: Probable root cause"` and `"Confidence: MEDIUM"` (driven by heuristic margins rather than LLM overconfidence).
- **Temporal Alignment**: Tracked the `orders_service` exhibiting its primary anomaly slightly ahead or simultaneous with the DB severity crest; yet the engine correctly penalized the callers due to topological downstream flow dampening (`"Dependency evidence: Topology role: Leaf downstream dependency"`).
- **Contributing Metrics**: Successfully isolated `"Severe connection saturation: peak 100.0%"` and `"Elevated error rate: peak 4.08%"` on the root candidate as driving weights.

## RCA Ranking Failures & Incorrect Candidate Rankings
- **Rank Margin Tightness**: `database` achieved Top-1 but only with a fractional margin (+0.136 over `orders_service`), earning a `MEDIUM` confidence rather than `HIGH`. Incorrect candidates (`orders_service`) ranked highly because they exhibited extremely severe temporal delays and massive error-rate saturation almost instantaneously after the database connection pool filled. If network latency delayed the DB anomaly metrics by just a single sampling tick (10s), the heuristic temporal weight could potentially flip the ranking, dropping DB to Rank 2.
- **Topological Rigidity Limitation**: Had `auth_service` somehow cascaded due to a shared infrastructure failure not encoded in the DAG topology graph, it would be improperly ranked due to lacking structural linkage.

## Limitations
1. **Heuristic Margins**: A composite score of 0.388 vs 0.252 works safely on this synthetic baseline, but real-world telemetry noise may scramble these weights (especially Temporal onset).
2. **Missing Causal Proof**: The results signify a "highest correlated cascading node," not a mathematical proof of causality.
3. **Single Incident**: Evaluated over 1 incident type; multiple concurrent disjoint incidents would diffuse scores significantly unless segmented.

## Readiness for Phase 6
The lightweight `RCAEngine` securely, predictably, and deterministically ranks failure cascades utilizing temporal, topological, and severity inferences without heavy unexplainable inference layers. It is fully ready for Phase 6 (MLOps, Drift & Model Monitoring).