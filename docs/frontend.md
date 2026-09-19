# Frontend Architecture & UX Design

## 1. Philosophy: Progressive Disclosure

AIOps Incident Intelligence adopts a strict **progressive disclosure** user experience designed for two distinct operational audiences:

1. **Default Operational View (Incident Responders / SREs / On-Call Engineers)**:
   - Immediately communicates:
     - **What is wrong?** (e.g., "Database connection pool saturated")
     - **Where?** (e.g., `database` node affecting `orders_service` and `api_gateway`)
     - **Current condition?** (e.g., "Active connections at 100%, query wait times degraded")
     - **Severity?** (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`)
     - **Evidence?** (e.g., "Connection utilization 98% (normal: 42%), earliest onset t=300s")
     - **Likely contributor?** (Identified upstream dependency)
     - **What to investigate?** (Actionable mitigation step)
   - Zero machine learning jargon, zero raw feature tensors, zero confusing probability mathematics.

2. **Technical Details Toggle (Machine Learning Engineers / Data Scientists / Senior SREs)**:
   - When toggled, reveals the complete diagnostic underpinnings:
     - Model identity and version (`IsolationForest-v1`, `LogisticRegression-v2`)
     - Multidimensional anomaly scores and decision thresholds ($\mu \pm 3\sigma$, contamination = 0.05)
     - Feature contribution weights (e.g., `memory_usage_pct_roll_mean_12: -3.22`)
     - Inference latency (ms) and processing overhead
     - Drift diagnostics (PSI and KS-test statistics)

---

## 2. Technical Stack

- **Technology**: Modern HTML5, Vanilla ES6 JavaScript, and Vanilla CSS.
- **Rationale**:
  - **Zero Build Step**: No Node.js runtime, npm installs, Webpack/Vite bundlers, or hydration delays required.
  - **Instant Serving**: Single static asset folder (`frontend/static/`) served directly by FastAPI via `StaticFiles`.
  - **Lightweight & Fast**: Initial page load under 50ms locally, sub-200ms inside Docker.
  - **Resilient**: Fully self-contained offline; requires zero CDN dependencies or external fonts.

---

## 3. UI Layout & View Structure

```
+----------------------------------------------------------------------------------+
|  [Logo Mark]  AIOps Intelligence               [Status: Healthy]  [Run Analysis] |
+----------------------------------------------------------------------------------+
|  [Overview]    [Incidents]    [Anomaly Explorer]    [Root Cause]    [MLOps Health]|
+----------------------------------------------------------------------------------+
|                                                                                  |
|  ACTIVE INCIDENT BANNER                                                          |
|  Critical Failure Detected: database connection pool saturated                   |
|                                                                                  |
|  +----------------------------------------------------------------------------+  |
|  | MOST LIKELY CONTRIBUTOR                                                    |  |
|  | Node: database | Confidence: 94.2% | Impact: Upstream orders_service stall |  |
|  | Evidence: Connection utilization at 98% (normal: 42%)                      |  |
|  | Recommended Action: Check connection pool leaks and slow query locks.       |  |
|  |                                                                            |  |
|  | [Toggle Technical Details]                                                |  |
|  | +------------------------------------------------------------------------+ |  |
|  | | Model: Isolation Forest v1 | Anomaly Score: -0.284 (threshold: -0.150) | |  |
|  | | Top Features: connection_utilization (+3.05), latency_ms (+2.14)       | |  |
|  | +------------------------------------------------------------------------+ |  |
|  +----------------------------------------------------------------------------+  |
|                                                                                  |
|  OTHER POSSIBLE CONTRIBUTORS (RANKED)                                            |
|  1. orders_service (Confidence: 78.4%) [Toggle Technical Details]                |
|  2. api_gateway (Confidence: 62.1%)    [Toggle Technical Details]                |
|  3. auth_service (Confidence: 8.3%)   [Isolated / Negative-Control]              |
|                                                                                  |
+----------------------------------------------------------------------------------+
```

---

## 4. Design System & Theming

The UI employs a dark technical aesthetic styled with standard CSS variables:

| Variable | Value | Purpose |
| :--- | :--- | :--- |
| `--bg-canvas` | `#090d16` | Main background |
| `--bg-surface` | `#111827` | Card surfaces |
| `--bg-surface-elevated` | `#1a2234` | Elevated dialogs & active cards |
| `--border-subtle` | `#1e293d` | Dividers & card outlines |
| `--color-primary` | `#3b82f6` | Brand accent & primary actions |
| `--status-normal` | `#10b981` | Healthy system status |
| `--status-warning` | `#f59e0b` | Medium/High warning state |
| `--status-critical` | `#ef4444` | Critical failure alert |

---

## 5. Assets & Branding

- **Favicon**: Vector SVG at `frontend/static/favicon.svg` (64x64 technical mark depicting telemetry waves, alert nodes, and interconnected service vertices).
- **Header Logo**: Inline vector mark with crisp typography (`AIOps Intelligence`).
- **Responsive**: Fully fluid responsive grid adapting seamlessly from 1920px desktop displays to narrow tablet/mobile viewports.
