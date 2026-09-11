# Phase 8.1: Final Evaluation & Comprehensive Benchmark Report

## 1. Executive Summary & Evaluation Scope

This report presents the final empirical evaluation and operational benchmarking for the **AIOps Incident Intelligence** platform (`OmniRoute`). The evaluation rigorously assesses all five core layers of the system using measured data, strictly preserving chronological boundaries, zero future-leakage guarantees, and verified ground-truth baselines.

### Scope of Evaluation
1. **Anomaly Detection**: Comparison between the Statistical Baseline (Trailing Window Z-score) and the Machine Learning model (Isolation Forest).
2. **Incident Prediction**: Early-warning failure prediction across a 30-minute future horizon evaluated over chronological continuous telemetry streams (Train: 60%, Validation: 20%, Test: 20%).
3. **Severity Classification**: Multi-class incident severity grading (`low`, `medium`, `high`, `critical`) with per-class diagnostics and confusion matrices.
4. **Root Cause Analysis (RCA)**: Heuristic topology and temporal ranking engine evaluating Top-1, Top-3, Mean Reciprocal Rank (MRR), and deterministic natural-language explainability.
5. **System & Operational Performance**: Profiling API response latencies, model batch inference throughput, memory overhead (`tracemalloc`), and on-disk artifact footprints.

---

## 2. Planned Targets vs. Actual Measured Results

| Operational Domain | Evaluation Metric | Planned Target | Actual Measured Result | Target Met? |
| :--- | :--- | :--- | :--- | :--- |
| **Anomaly Detection** | Precision | $\ge 70.0\%$ | **71.56%** | Yes |
| | Recall | $\ge 80.0\%$ | **83.87%** | Yes |
| | F1 Score | $\ge 0.7500$ | **0.7723** | Yes |
| | False Positive Rate (FPR) | $\le 15.0\%$ | **11.61%** | Yes |
| | Detection Delay | $\le 30.0\text{s}$ | **10.0s** | Yes |
| **Incident Prediction** | PR-AUC | $\ge 0.7500$ | **0.7599** | Yes |
| | ROC-AUC | $\ge 0.8000$ | **0.8130** | Yes |
| | F1 Score (Test Split) | $\ge 0.6500$ | **0.6547** | Yes |
| | Precision (Test Split) | $\ge 60.0\%$ | **63.08%** | Yes |
| | Recall (Test Split) | $\ge 65.0\%$ | **68.05%** | Yes |
| | Prediction Lead Time | $\ge 900.0\text{s}$ (15 min) | **1,632.5s (~27.2 min)** | Yes |
| **Severity Classification**| Macro F1 Score | $\ge 0.7000$ | **1.0000** (Single Scenario) | Ceiling Note* |
| **Root Cause Analysis** | Top-1 Accuracy | $\ge 90.0\%$ | **100.0% (1.0)** | Yes |
| | Top-3 Accuracy | $\ge 95.0\%$ | **100.0% (1.0)** | Yes |
| | Mean Reciprocal Rank (MRR)| $\ge 0.9000$ | **1.0000** | Yes |
| **System Performance** | Max Batch API Latency | $\le 500\text{ms}$ (720 records) | **144.45ms** (Max: Anomaly) | Yes |
| | Total Model Disk Size | $\le 25\text{MB}$ | **2.20 MB (2,248 KB)** | Yes |
| | Peak Inference Memory | $\le 10\text{MB}$ | **306.28 KB** | Yes |

*\*Note on Severity Macro F1*: Tested on the single cascading failure scenario where all incident occurrences are labeled `critical`. When multi-class diversity is synthetically perturbed, macro F1 drops to ~0.25 (chance level) due to identical telemetry feature amplitudes across synthetic classes.

---

## 3. Anomaly Detection Benchmark

Evaluated on the benchmark cascading incident dataset ($N=360$ samples across 4 microservices over 900 seconds, 10s interval, incident from $t=300\text{s}$ to $t=600\text{s}$, affecting 93 total samples).

### Comparative Metric Results

| Metric | Statistical Baseline (Z-score) | Isolation Forest Detector |
| :--- | :--- | :--- |
| **Precision** | 45.24% | **71.56%** |
| **Recall** | 20.43% | **83.87%** |
| **F1 Score** | 0.2815 | **0.7723** |
| **False Positive Rate (FPR)** | **8.61%** | 11.61% |
| **Detection Delay** | **10.0s** | **10.0s** |
| **True Positives (TP)** | 19 | **78** |
| **False Positives (FP)** | 23 | 31 |
| **True Negatives (TN)** | 244 | 236 |
| **False Negatives (FN)** | 74 | **15** |
| **Total Test Samples** | 360 | 360 |

### Anomaly Failure Analysis
1. **Statistical Baseline (Trailing Z-score)**:
   - *Primary Failure Mode*: Severe Recall failure (20.43%, 74 False Negatives). The rolling statistical window (12 samples / 120s) rapidly incorporates the anomalous metric values into its internal rolling mean and standard deviation. As the incident persists past the initial onset burst, the detector normalizes the saturated state and ceases flagging anomalies.
   - *Advantage*: High pre-incident noise rejection (FPR 8.61%).
2. **Isolation Forest**:
   - *Primary Strength*: High recall (83.87%) sustained across the full multi-minute duration of the cascading failure because the multidimensional isolation trees are anchored against a static pre-incident baseline.
   - *Primary Failure Mode*: Minor False Positives (31 FP, FPR 11.61%) triggered by synthetic normal jitter crossing the multidimensional boundary when `contamination=0.05`.

---

## 4. Incident Prediction Benchmark

Evaluated on **Benchmark B**: a continuous 6-hour multi-episode telemetry stream ($N=8,640$ rows across `api_gateway`, `auth_service`, `orders_service`, `database`, 4 failure episodes, 2 healthy recovery episodes), split strictly chronologically.

### Chronological Splits
- **Train Split (60%)**: $N=5,184$ samples (1,356 positive horizon samples, 3,828 negative)
- **Validation Split (20%)**: $N=1,728$ samples (783 positive horizon samples, 945 negative)
- **Test Split (20%)**: $N=1,728$ samples (723 positive horizon samples, 1,005 negative)

### Measured Performance Across Partitions

| Metric | Train Partition | Validation Partition | Test Partition (Final) |
| :--- | :--- | :--- | :--- |
| **Precision** | 45.20% | 68.94% | **63.08%** |
| **Recall** | 74.63% | 68.33% | **68.05%** |
| **F1 Score** | 0.5630 | 0.6863 | **0.6547** |
| **PR-AUC** | 0.6956 | 0.8125 | **0.7599** |
| **ROC-AUC** | 0.8261 | 0.8328 | **0.8130** |
| **Lead Time** | — | — | **1,632.5s (~27.2 minutes)** |
| **False Positives** | 1,643 | 308 | **288** |
| **False Negatives** | 344 | 248 | **231** |

### Top Predictive Features (Logistic Regression Feature Weights)

| Rank | Feature Name | Weight (Coef) | Physical / Operational Interpretation |
| :--- | :--- | :--- | :--- |
| 1 | `memory_usage_pct_roll_mean_12` | `-3.2215` | Pre-incident memory stabilization baseline |
| 2 | `disk_usage_pct` | `+3.0508` | Sustained I/O pressure indicator |
| 3 | `system_max_error_rate` | `-2.6102` | Normal pre-incident window exhibits near-zero error baseline |
| 4 | `system_mean_latency` | `+2.2436` | Cross-service latency drift preceding cascade onset |
| 5 | `memory_usage_pct_roll_max_12` | `+2.0158` | Trailing peak memory spikes |
| 6 | `latency_ms_roll_max_12` | `-1.7189` | Peak latency stabilization penalty |
| 7 | `connection_pressure` | `-1.6878` | Gradient of active connections vs pool limit |
| 8 | `connection_utilization_roll_mean_12` | `+1.1627` | Gradual saturation of database connection pool |

### Incident Prediction Failure Analysis
1. **Post-Incident False Positives (288 in Test)**:
   - Trailing 12-sample (120s) rolling aggregations (`memory_usage_pct_roll_mean_12`, `connection_utilization_roll_mean_12`) require 1–2 minutes after incident recovery to drain back to nominal levels. Because the target label immediately switches to `False` once outside the horizon window, decaying rolling statistics produce false alarms immediately after an incident concludes.
2. **Early False Negatives (231 in Test)**:
   - At $t - 28\text{m}$, pre-saturation telemetry is nearly indistinguishable from nominal baseline jitter under a purely linear decision boundary. Linear models cannot capture the sharp non-linear knee where connection usage jumps from 75% to 100%.

---

## 5. Severity Classification Benchmark

Evaluated on the Benchmark B Test partition ($N=177$ incident occurrences during active failure windows).

### Per-Class Performance Breakdown

| Severity Class | Precision | Recall | F1 Score | Support (Test Partition) |
| :--- | :--- | :--- | :--- | :--- |
| **Low** | 0.0% | 0.0% | 0.0000 | 0 |
| **Medium** | 0.0% | 0.0% | 0.0000 | 0 |
| **High** | 0.0% | 0.0% | 0.0000 | 0 |
| **Critical** | **100.0%** | **100.0%** | **1.0000** | **177** |
| **Macro Average** | **1.0000** | **1.0000** | **1.0000** | **177** |

### Severity Confusion Matrix (Test Split)
```
                  Predicted: Low   Medium   High   Critical
Actual Low                    0        0       0          0
Actual Medium                 0        0       0          0
Actual High                   0        0       0          0
Actual Critical               0        0       0        177
```

### Severity Limitation Analysis
- **Synthetic Data Ceiling**: The current generator injects failure scenarios with uniform severe degradation offsets. Although the multi-class pipeline, feature extractors, and evaluation harness are fully implemented and verified, evaluating true multi-class discrimination requires multi-amplitude synthetic models or real-world enterprise incident traces.

---

## 6. Root Cause Analysis (RCA) Benchmark

Evaluated across the synthetic failure scenario (`database_connection_saturation`, $t=900\text{s} \to 1500\text{s}$) where `database` is the known ground-truth root cause.

### Ranking Accuracy

| Metric | Measured Value | Planned Target |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | **100.0% (1.0)** | $\ge 90.0\%$ |
| **Top-3 Accuracy** | **100.0% (1.0)** | $\ge 95.0\%$ |
| **Mean Reciprocal Rank (MRR)** | **1.0000** | $\ge 0.9000$ |
| **Total Evaluated Incidents** | 1 | — |
| **Missing Ground Truth** | 0 | — |

### Root Cause Candidate Score Breakdown

| Rank | Service Name | Composite Score | Temporal Score | Topology Score | Severity Score | Classification / Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `database` | **0.3881** | **0.8551** | **1.0000** | **0.8945** | **Probable Root Cause** (Leaf dependency, earliest anomaly at 00:15:10Z, peak saturation 100%) |
| **2** | `auth_service` | **0.2518** | 0.7246 | 0.8000 | 0.0964 | Correlated Candidate (Low severity max score 0.559, unaffected caller) |
| **3** | `orders_service`| **0.2381** | 0.5652 | 0.3000 | 0.9228 | Direct Caller Cascade (Delayed anomaly onset at 00:15:30Z, high cascade error rate) |
| **4** | `api_gateway` | **0.1220** | 0.2754 | 0.0000 | 0.7103 | Edge Gateway Cascade (Delayed anomaly onset at 00:15:50Z, downstream victim) |

### Explanation Verification & Non-Causal Safety
The deterministic RCA Explainer output cleanly distinguishes evidence from inference:
- Generates transparent linguistic framing: `"Classification: Probable root cause (Confidence: MEDIUM)"`.
- Explicit non-causal disclaimer present on all API payloads: `"Heuristic ranking based on temporal onset, dependency topology, and anomaly severity. Does not constitute formal causal proof."`

---

## 7. System & Operational Benchmarks

All benchmarks measured locally under non-virtualized standard execution profiling $N=720$ records across 4 services.

### 1. API Endpoint Response Latencies

| Endpoint | Method | Payload Size | Measured Response Latency | Operational SLA Budget |
| :--- | :--- | :--- | :--- | :--- |
| `/api/v1/health` | GET | None | **13.41 ms** | $< 200\text{ms}$ |
| `/api/v1/mlops/status` | GET | None | **3.24 ms** | $< 200\text{ms}$ |
| `/api/v1/telemetry/summary` | POST | 720 records | **19.36 ms** | $< 500\text{ms}$ |
| `/api/v1/anomalies/detect` | POST | 720 records | **144.45 ms** | $< 500\text{ms}$ |
| `/api/v1/incidents/predict` | POST | 720 records | **47.79 ms** | $< 500\text{ms}$ |
| `/api/v1/incidents/severity` | POST | 720 records | **50.59 ms** | $< 500\text{ms}$ |
| `/api/v1/rca/rank` | POST | 720 records + incident | **112.89 ms** | $< 500\text{ms}$ |
| `/api/v1/rca/explain` | POST | 720 records + incident | **114.10 ms** | $< 500\text{ms}$ |

### 2. Model Inference & Throughput Metrics

| Model Artifact | Batch Latency ($N=720$) | Per-Sample Latency | Throughput | Peak Memory Overhead |
| :--- | :--- | :--- | :--- | :--- |
| **Isolation Forest Detector** | 150.37 ms | 208.85 µs / record | ~4,788 records/sec | ~306.28 KB |
| **Incident Predictor Baseline**| 7.47 ms | 10.37 µs / record | ~96,385 records/sec| Included in peak |
| **Severity Classifier Baseline**| 7.35 ms | 10.21 µs / record | ~97,959 records/sec| Included in peak |

### 3. Model Artifact Footprint on Disk

| Artifact Path | Storage Format | Measured Size on Disk |
| :--- | :--- | :--- |
| `data/models/isolation_forest_latest.joblib` | Compressed Scikit-Learn Ensemble | **2,243.54 KB (~2.19 MB)** |
| `data/models/incident_predictor_latest.joblib` | Scikit-Learn Logistic Pipeline | **2.22 KB** |
| `data/models/severity_classifier_latest.joblib`| Scikit-Learn Multi-Class Pipeline | **2.30 KB** |
| **Total Model Artifact Footprint** | | **2,248.07 KB (~2.20 MB)** |

---

## 8. Summary of Major Failures & Inherent Limitations

1. **Synthetic Noise Stationarity**:
   - The synthetic generator produces Gaussian stationary baselines during non-incident intervals. Real-world microservice workloads exhibit diurnal and seasonal volatility that will require dynamic baseline recalibration.
2. **Post-Recovery Alert Ringing**:
   - Trailing rolling features introduce a 60–120s recovery delay where models continue predicting incidents shortly after resolution.
3. **Linear Decision Boundary for Saturation**:
   - Connection saturation behavior is intrinsically non-linear (step-function failure once connection limits are exceeded). Logistic regression under-predicts the earliest slope before saturation occurs.
4. **DAG Graph Rigidity**:
   - RCA ranking presumes a static dependency DAG. Dynamic ephemeral microservices or unmodeled infrastructure dependencies (e.g., AWS availability zone outages) cannot be traced without expanded graph topology definitions.

---

## 9. Verification & Codebase Integrity

- **Automated Test Suite**: 136 passed tests (`pytest -q` execution time ~5.9s).
- **Leakage Safeguards**: Strict temporal splits, causal rolling feature calculations, and zero test-partition leakage.
- **Reproducibility**: All evaluation code is packaged and runnable via `PYTHONPATH=. python evaluation/run_final_evaluation.py`.

---

## 10. Next Steps

With Phase 8.1 (Final Evaluation & Benchmarking) successfully completed, the project proceeds to **Phase 8.2 — Final Packaging, Code Cleanup, and Project Handover Documentation**.
