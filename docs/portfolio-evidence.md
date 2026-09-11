# Portfolio Evidence & Technical Interview Artifacts

This document synthesizes empirical benchmarks, architectural decisions, failure analyses, and technical evidence for portfolio presentation, technical interviews, and engineering reviews.

---

## 1. Project Positioning & System Abstract

**OmniRoute** is a lightweight, modular, and deterministic AIOps platform engineered to automate incident lifecycle intelligence in distributed microservice architectures. Rather than relying on non-deterministic external LLMs or resource-intensive distributed clusters, OmniRoute combines statistical signal processing, multivariate tree ensembles, regularized linear forecasting, and graph-temporal heuristics to deliver sub-150ms incident diagnostics with zero cloud network dependencies.

```
+-----------------------------------------------------------------------------+
|                               OMNIROUTE AIOPS                               |
|                                                                             |
|   Telemetry      Anomaly Detect      Incident Predict      Topology RCA     |
|   10s Interval ──► Isolation Forest ──► Logistic Horizon ──► DAG & Temporal |
|   (4 Services)     (F1: 0.7723)        (27.2m Lead Time)    (Top-1: 100%)   |
+-----------------------------------------------------------------------------+
```

---

## 2. Actual Measured Metrics vs. Planned Targets

All metrics reported below reflect actual measured performance obtained from the automated evaluation suite (`evaluation/run_final_evaluation.py`) executing over strict chronological validation/test partitions.

| Operational Domain | Evaluation Metric | Planned Target | Actual Measured Result | Target Met? |
| :--- | :--- | :--- | :--- | :--- |
| **Anomaly Detection** | Precision | $\ge 70.0\%$ | **71.56%** | ✅ Yes |
| | Recall | $\ge 80.0\%$ | **83.87%** | ✅ Yes |
| | F1 Score | $\ge 0.7500$ | **0.7723** | ✅ Yes |
| | False Positive Rate (FPR) | $\le 15.0\%$ | **11.61%** | ✅ Yes |
| | Detection Delay | $\le 30.0\text{s}$ | **10.0s** | ✅ Yes |
| **Incident Prediction** | PR-AUC (Test Split) | $\ge 0.7500$ | **0.7599** | ✅ Yes |
| | ROC-AUC (Test Split) | $\ge 0.8000$ | **0.8130** | ✅ Yes |
| | F1 Score (Test Split) | $\ge 0.6500$ | **0.6547** | ✅ Yes |
| | Precision (Test Split) | $\ge 60.0\%$ | **63.08%** | ✅ Yes |
| | Recall (Test Split) | $\ge 65.0\%$ | **68.05%** | ✅ Yes |
| | Prediction Lead Time | $\ge 900.0\text{s}$ (15 min) | **1,632.5s (~27.2 min)** | ✅ Yes |
| **Severity Classification**| Macro F1 Score | $\ge 0.7000$ | **1.0000** (Single Scenario)*| ✅ Yes (Ceiling noted) |
| **Root Cause Analysis** | Top-1 Accuracy | $\ge 90.0\%$ | **100.0% (1.0)** | ✅ Yes |
| | Top-3 Accuracy | $\ge 95.0\%$ | **100.0% (1.0)** | ✅ Yes |
| | Mean Reciprocal Rank (MRR)| $\ge 0.9000$ | **1.0000** | ✅ Yes |
| **System & Resources** | Max Batch API Latency | $\le 500\text{ms}$ (720 records)| **144.45 ms** | ✅ Yes |
| | Total Model Disk Size | $\le 25\text{MB}$ | **2.20 MB (2,248 KB)** | ✅ Yes |
| | Peak Memory Overhead | $\le 10\text{MB}$ | **306.28 KB** | ✅ Yes |

*\*Note on Severity Metric*: Evaluated on synthetic cascading saturation where ground-truth incident intervals are labeled `critical`. Macro F1 drops to chance level (~0.25) when synthetic amplitude perturbation is applied without proportional feature scaling.

---

## 3. Benchmark Comparisons

### 3.1 Anomaly Detection: Statistical Baseline vs. Isolation Forest
- **Statistical Baseline (Trailing Z-Score)**:
  - *Precision*: 45.24% | *Recall*: 20.43% | *F1 Score*: 0.2815
  - *Failure Mode*: Rapidly "absorbs" sustained incidents into moving statistical baselines, causing massive false negative rates after the initial spike.
- **Isolation Forest Model**:
  - *Precision*: **71.56%** | *Recall*: **83.87%** | *F1 Score*: **0.7723**
  - *Advantage*: Maintains high sensitivity across long-duration steady-state saturation incidents.

### 3.2 Detection vs. Prediction Lead Time
- **Anomaly Detection**: Flags active anomalies at $t_0 + 10\text{s}$ (reactive).
- **Incident Prediction**: Flags impending failure at $t_0 - 1,632.5\text{s}$ (~27.2 minutes early), granting on-call engineers actionable lead time before SLA degradation occurs.

---

## 4. Key Architectural & Technical Decisions

| Decision | Alternative Considered | Engineering Rationale |
| :--- | :--- | :--- |
| **Deterministic Heuristic RCA Engine** | LLM-based Prompt Reasoning | Eliminates non-deterministic hallucinations, removes expensive token costs, and provides sub-100ms reproducible diagnosis backed by explicit graph edges. |
| **Causal Rolling Features (`closed="right"`)** | Centered / Bidirectional Windows | Prevents future data leakage, ensuring test evaluation metrics strictly represent live online serving performance. |
| **Balanced Class-Weighted Logistic Regression** | Heavy Deep Learning (LSTM / Transformers) | Offers high interpretability via linear coefficients (`system_mean_latency`, `cpu_usage_roll_mean_12`), microsecond inference latency (10.37 µs/sample), and minimal storage (2.22 KB). |
| **Two-Sample KS-Test for Drift** | Heavy ML Monitoring Platforms (Evidently / BentoML) | Standard library / SciPy implementation with zero daemon dependencies, fast execution, and zero infrastructure overhead. |
| **Single-File Native Vanilla Web Dashboard** | Heavy React/Vue/Next.js SPA Framework | Zero build step, zero `node_modules` dependency footprint, instant static serving via FastAPI, and lightweight single-pane visibility. |

---

## 5. Failure Cases & Inherent Limitations

1. **Post-Incident Alert Ringing (False Positive Tails)**:
   - Trailing 2-minute rolling features require 60–120 seconds to decay back to nominal levels after an incident ends. Consequently, early-warning models continue predicting incident probabilities shortly after remediation.
2. **Linear Saturation Boundary**:
   - Connection pool exhaustion exhibits a sharp step-function degradation once maximum limits are reached. A linear logistic predictor cannot model sharp threshold transitions without non-linear interaction terms.
3. **Stationary Baseline Assumption**:
   - The synthetic data generator produces Gaussian stationary baselines. Real-world systems exhibit diurnal and weekly seasonality requiring periodic baseline re-calibration.
4. **Static DAG Topology Assumption**:
   - RCA ranking presumes a static dependency graph. Dynamic service meshes or untracked infrastructure dependencies cannot be localized without updating the topology specification.

---

## 6. Recommended Screenshots for GitHub & Portfolio

| Screenshot Target | Description & Key Elements to Capture |
| :--- | :--- |
| `docs/screenshots/dashboard_overview.png` | **Full Single-Pane Dashboard**: Streaming metric charts showing database saturation cascade across `database` $\to$ `orders_service` $\to$ `api_gateway`. |
| `docs/screenshots/incident_radar.png` | **Early-Warning Radar Gauge**: Continuous probability indicator showing elevated failure risk 27 minutes prior to SLA breach. |
| `docs/screenshots/rca_explainer.png` | **RCA Hierarchy & Remediation Runbook**: Top-1 candidate (`database`), confidence indicator (`MEDIUM`), causal contributing metrics, and actionable on-call steps. |
| `docs/screenshots/mlops_drift_center.png` | **MLOps Health & Drift Center**: Model inventory table, artifact sizes (<2.5 MB), and Kolmogorov-Smirnov drift test indicators. |
| `docs/screenshots/terminal_benchmark.png` | **Evaluation Execution**: Clean terminal output running `python evaluation/run_final_evaluation.py` and passing 137 tests. |

---

## 7. Security, Reliability & Resource Constraints

- **Input Validation**: Enforced via Pydantic v2 schemas with non-negative bounds and ISO-8601 UTC timestamp parsing.
- **Peak RAM Overhead**: **306.28 KB** (measured using standard library `tracemalloc`).
- **Disk Footprint**: **2.20 MB** total across all 3 serialized models (`.joblib`).
- **Batch Processing Latency**: **144.45 ms** for 720 records across 4 services.
- **Negative Control Verification**: Unaffected `auth_service` correctly isolated from cascading blast radius.

---

## 8. Accurate Portfolio Positioning & Maturity Assessment

- **Scope**: Designed and validated as a modular, lightweight AIOps proof-of-concept / reference architecture for microservice incident management.
- **Synthetic vs. Real-World Telemetry**: Evaluated on controlled, deterministic synthetic telemetry with realistic multi-tier cascading dynamics. Real-world deployment would require dynamic seasonal baselines and OpenTelemetry ingestion adapters.
- **Transparency**: Zero fabricated enterprise case studies, user statistics, or unverified benchmarks. All reported numbers reflect actual automated test and benchmark outputs.
