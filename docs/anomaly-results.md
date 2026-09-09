# Phase 3.3: Anomaly Detection Evaluation & Results

## Objective
Evaluate and compare the performance of the **Statistical Baseline** (Trailing Window Z-score) and the **Isolation Forest** ML model on synthetic incident telemetry for the `database_connection_saturation` cascading failure scenario. 

## Dataset Description
- **Duration**: 900 seconds (15 minutes) at 10-second sampling intervals (360 total samples across 4 services).
- **Incident Injection**: Database saturation from $t=300\text{s}$ to $t=600\text{s}$.
- **Ground Truth**: The timeline ($[300, 600]$) marks `database`, `orders_service`, and `api_gateway` as strictly anomalous (affecting 93 samples total). `auth_service` remains unaffected.

## Measured Metrics

| Metric | Statistical Baseline | Isolation Forest |
| :--- | :--- | :--- |
| **Precision** | 45.24% | **71.56%** |
| **Recall** | 20.43% | **83.87%** |
| **F1 Score** | 0.2815 | **0.7723** |
| **False Positive Rate (FPR)** | **8.61%** | 11.61% |
| **Detection Delay** | **10.0s** | **10.0s** |
| **True Positives** | 19 | 78 |
| **False Negatives** | 74 | 15 |

## Analysis of Results

### 1. Statistical Baseline (Z-Score)
* **Strengths**: Extremely rapid initial detection (10.0s) and excellent noise rejection pre-incident (FPR < 9%).
* **Major Failure Cases**: Abysmal recall (20.43%). Trailing window algorithms rapidly drift their internal mean/std upwards to "absorb" the failure when an incident spans continuously for several minutes. As a result, the detector only flags the initial gradient spike and completely ignores the ongoing steady-state saturation, yielding enormous numbers of false negatives.
* **Signals Used**: `latency_ms` and `error_rate` spikes caused the short sequence of true positives.

### 2. Isolation Forest
* **Strengths**: High recall (83.87%) throughout the entire long-running failure duration. Because the ML model leverages a static baseline trained firmly upon pre-incident data (or normal historical conditions), it does not falsely condition itself to the new saturated status quo. 
* **Major Failure Cases**: Slightly higher false positive rate (11.61%) resulting from minor non-anomalous jitter triggering the `contamination` ceiling when scoring multi-dimensional thresholds.
* **Signals Used**: Detected multidimensional structural shifts in `connection_utilization` acting orthogonally to `cpu_usage_pct`.

## Limitations of the Synthetic Evaluation
1. **Unrealistic Data Cleanliness**: The synthetic data generator produces perfectly stationary multivariate normality prior to the incident, heavily favoring the Isolation Forest over real-life highly non-stationary microservice traffic.
2. **Fixed Contamination Rates**: Fixing the ML threshold at 5% strictly bounds worst-case FP distribution, whereas dynamic alerting thresholds often depend on seasonality not encoded in the synthetic stream.

## Recommendation for Next Phase
The **Isolation Forest** model is decisively recommended for deployment into Phase 4 (Incident Prediction & Root Cause Analysis). Its structural capability to sustain incident alerting across the entire lifecycle of a complex cascade outweighs the lightweight Statistical Baseline, which suffers from severe "normalization of deviance" blind spots after just a few minutes of failure. 
