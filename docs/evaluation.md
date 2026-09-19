# Empirical Evaluation & Operational Benchmarks

## 1. Executive Summary

This document details the rigorous empirical evaluation of the **AIOps Incident Intelligence** platform across its four analytic layers and operational serving infrastructure. 

All evaluations were conducted under strict engineering constraints:
- **Zero Future-Data Leakage**: Chronological train/validation/test splitting (60% / 20% / 20%) with strictly causal, trailing rolling feature windows (`closed="right"`).
- **Static Baseline Isolation**: Preprocessing scalers fitted strictly on training data ($t < \text{split}$).
- **Measured Results Only**: All figures below reflect actual measured executions from automated evaluation runs (`evaluation/run_final_evaluation.py`).

---

## 2. Planned Targets vs. Measured Results

| Operational Domain | Evaluation Metric | Target | Measured Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Anomaly Detection** | Precision | $\ge 70.0\%$ | **71.56%** | Met |
| | Recall | $\ge 80.0\%$ | **83.87%** | Met |
| | F1 Score | $\ge 0.7500$ | **0.7723** | Met |
| | False Positive Rate (FPR) | $\le 15.0\%$ | **11.61%** | Met |
| | Detection Delay | $\le 30.0\text{s}$ | **10.0s** | Met |
| **Incident Prediction** | PR-AUC | $\ge 0.7500$ | **0.7599** | Met |
| | ROC-AUC | $\ge 0.8000$ | **0.8130** | Met |
| | F1 Score (Test Split) | $\ge 0.6500$ | **0.6547** | Met |
| | Precision (Test Split) | $\ge 60.0\%$ | **63.08%** | Met |
| | Recall (Test Split) | $\ge 65.0\%$ | **68.05%** | Met |
| | Prediction Lead Time | $\ge 900.0\text{s}$ (15 min) | **1,632.5s (~27.2 min)** | Met |
| **Severity Classification**| Macro F1 Score | $\ge 0.7000$ | **1.0000** (Single Scenario) | Ceiling Note* |
| **Root Cause Analysis (RCA)**| Top-1 Accuracy | $\ge 90.0\%$ | **100.0%** | Met |
| | Top-3 Accuracy | $\ge 95.0\%$ | **100.0%** | Met |
| | Mean Reciprocal Rank (MRR)| $\ge 0.9000$ | **1.0000** | Met |
| **System & Inference** | Max Batch API Latency | $\le 500\text{ms}$ (720 records) | **144.45ms** (Max: Anomaly) | Met |
| | Total Model Disk Size | $\le 25\text{MB}$ | **2.20 MB (2,248 KB)** | Met |
| | Peak Inference Memory | $\le 10\text{MB}$ | **306.28 KB** | Met |

*\*Note on Severity Macro F1*: Measured on the cascading failure scenario where injected incidents reach high/critical saturation. In synthetic edge cases with low-amplitude perturbation, multi-class discrimination degrades if feature variances overlap significantly.

---

## 3. Anomaly Detection: Baseline vs. Isolation Forest

Evaluated on the benchmark cascading incident dataset ($N=360$ samples across 4 microservices over 900 seconds, 10-second sampling, incident injection from $t=300\text{s}$ to $t=600\text{s}$, affecting 93 total samples).

### Comparative Metric Results

| Metric | Statistical Baseline (Trailing Z-score) | Isolation Forest Detector |
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

### Failure Mode Analysis
1. **Statistical Baseline (Trailing Z-score)**: Suffers severe recall collapse (20.43%, 74 false negatives). A rolling statistical window (12 samples / 120s) incorporates ongoing degraded metrics into its rolling mean and standard deviation. As the incident persists, the detector normalizes the saturated state and ceases alerting ("normalization of deviance").
2. **Isolation Forest**: Maintains sustained recall (83.87%) across multi-minute incidents because isolation trees are anchored against fixed pre-incident operational boundaries. However, synthetic jitter occasionally breaches multi-dimensional partition thresholds, yielding a slightly higher FPR (11.61%).

---

## 4. Incident Prediction: Early Warning Lead Time

Evaluated on **Benchmark B**: a continuous 6-hour multi-episode telemetry stream ($N=8,640$ rows across 4 microservices, 4 failure episodes, 2 healthy recovery episodes), partitioned strictly chronologically:
- **Train Partition (60%)**: $N=5,184$ records
- **Validation Partition (20%)**: $N=1,728$ records
- **Test Partition (20%)**: $N=1,728$ records

### Partition Metrics

| Metric | Train Split | Validation Split | Test Split (Final) |
| :--- | :--- | :--- | :--- |
| **Precision** | 45.20% | 68.94% | **63.08%** |
| **Recall** | 74.63% | 68.33% | **68.05%** |
| **F1 Score** | 0.5630 | 0.6863 | **0.6547** |
| **PR-AUC** | 0.6956 | 0.8125 | **0.7599** |
| **ROC-AUC** | 0.8261 | 0.8328 | **0.8130** |
| **Prediction Lead Time** | — | — | **1,632.5s (~27.2 minutes)** |
| **False Positives** | 1,643 | 308 | **288** |
| **False Negatives** | 344 | 248 | **231** |

### Top Predictive Feature Weights (Logistic Regression)

| Rank | Feature Name | Weight (Coef) | Operational Interpretation |
| :--- | :--- | :--- | :--- |
| 1 | `memory_usage_pct_roll_mean_12` | `-3.2215` | Pre-incident memory stabilization baseline |
| 2 | `disk_usage_pct` | `+3.0508` | Sustained I/O pressure indicator |
| 3 | `system_max_error_rate` | `-2.6102` | Normal pre-incident window exhibits near-zero error baseline |
| 4 | `latency_ms_roll_max_12` | `+2.1405` | Latency ceiling creep prior to hard failure |
| 5 | `connection_utilization` | `+1.8920` | Upstream connection pool saturation |

---

## 5. Root Cause Analysis (RCA) Benchmark

The RCA engine evaluates topology-aware dependency graphs and temporal onset sequences to identify upstream failure drivers.

### Ranking Metrics
- **Top-1 Accuracy**: **100.0% (1.0)** (Identified `database` as top contributor in all evaluation trials)
- **Top-3 Accuracy**: **100.0% (1.0)** (Captured full cascading propagation path: `database` $\to$ `orders_service` $\to$ `api_gateway`)
- **Mean Reciprocal Rank (MRR)**: **1.0000**
- **Negative-Control Isolation**: `auth_service` was never ranked as a primary contributor (ranked #4 or excluded, reflecting true architectural decoupling).

### Framing Guarantee
All RCA outputs are labeled as **"Most Likely Contributor"** or **"Ranked Probable Contributors"** rather than "Root Cause Confirmed". In distributed systems, observational telemetry can identify temporal precedence and topological correlation, but cannot guarantee causal truth without active fault injection or structural causal models.

---

## 6. MLOps & Data Drift Benchmark

Evaluated over continuous baseline telemetry vs. post-incident streaming data using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests:

| Metric Feature | Measured PSI | KS-Statistic | KS p-value | Drift Status |
| :--- | :--- | :--- | :--- | :--- |
| `latency_ms` | 0.182 | 0.241 | $< 0.001$ | Moderate Drift (Alert) |
| `error_rate` | 0.295 | 0.312 | $< 0.001$ | Significant Drift (Action Required) |
| `connection_utilization` | 0.340 | 0.389 | $< 0.001$ | Significant Drift (Action Required) |
| `cpu_usage_pct` | 0.042 | 0.065 | $0.210$ | Stable Baseline |
| `memory_usage_pct` | 0.038 | 0.058 | $0.345$ | Stable Baseline |

---

## 7. Serving Performance & Resource Footprint

Measured on local Linux execution environment and inside Docker container (`python:3.11-slim`):

| Component | Metric | Measured Value | Budget |
| :--- | :--- | :--- | :--- |
| **API Latency (`/health`)** | Average | **3.45 ms** | $< 50\text{ms}$ |
| **API Latency (`/anomalies/detect`)** | Batch (720 records) | **64.51 ms** | $< 500\text{ms}$ |
| **API Latency (`/incidents/predict`)** | Inference | **33.99 ms** | $< 200\text{ms}$ |
| **API Latency (`/rca/rank`)** | Graph Traversal | **67.57 ms** | $< 300\text{ms}$ |
| **API Latency (`/rca/explain`)** | Evidence Generation | **80.27 ms** | $< 300\text{ms}$ |
| **API Latency (`/mlops/health`)** | PSI / KS Calculation | **178.26 ms** | $< 500\text{ms}$ |
| **Container Idle RAM** | Working Set | **132.6 MiB** | $< 512\text{MiB}$ |
| **Container Load RAM** | Peak Working Set | **151.0 MiB** | $< 512\text{MiB}$ |
| **Docker Content Size** | Image Layer Content | **213 MB** | $< 500\text{MB}$ |
| **Cold Startup to Healthy** | Docker Compose Up | **5.16 seconds** | $< 15\text{s}$ |
