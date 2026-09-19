# Local Docker Deployment & Reproducibility Guide

This guide documents the local containerized deployment for the **Omniroute AIOps Incident Intelligence** platform on a standard developer workstation or laptop.

> **Positioning**: *Containerized and reproducibly runnable on a developer machine.* (Not cloud-deployed or distributed enterprise cluster).

---

## 1. Architecture of the Docker Setup

The deployment is intentionally structured as a lean, single-container service to respect local laptop resource constraints and eliminate unnecessary microservice overhead:

```
[ Developer Browser / Client ]
             │
      Port 8000:8000
             ▼
┌────────────────────────────────────────────────────────┐
│  Container: aiops-incident-intelligence                │
│  (Base: python:3.11-slim, User: appuser 1000)          │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Uvicorn ASGI Server                              │  │
│  │                                                  │  │
│  │  • Static Dashboard Route:                       │  │
│  │    /            -> Redirect to /static/index.html│  │
│  │    /static/*    -> frontend/static/index.html    │  │
│  │                                                  │  │
│  │  • REST API Namespace (/api/v1):                 │  │
│  │    /health                  -> Liveness probe    │  │
│  │    /telemetry/latest        -> Parquet loader    │  │
│  │    /telemetry/summary       -> Feature stats     │  │
│  │    /anomalies/detect        -> Isolation Forest  │  │
│  │    /incidents/predict       -> LogReg Predictor  │  │
│  │    /incidents/severity      -> Multiclass Class. │  │
│  │    /rca/rank                -> Heuristic Graph   │  │
│  │    /rca/explain             -> Runbook Explainer │  │
│  │    /mlops/health            -> KS Drift & Health │  │
│  │    /mlops/status            -> Artifact status   │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  Persistent Volume Mount:                              │
│  ./data  ──►  /app/data (Models, Parquet, Experiments) │
└────────────────────────────────────────────────────────┘
```

### Architectural Decisions & Rationale
1. **Unified Application Container**: The frontend dashboard is a zero-dependency static web application (`frontend/static/index.html`) mounted directly by FastAPI at `/static`. Running a separate Nginx container would add unnecessary process overhead without functional benefit.
2. **Zero Database Infrastructure**: The system operates on columnar Parquet telemetry and serialized `.joblib` model artifacts. PostgreSQL, Redis, or Celery are not required by the core design and were omitted to preserve developer memory limits.
3. **Non-Root Execution**: Runs under unprivileged user `appuser` (UID 1000) for container hygiene.
4. **Standard Library Health Check**: Health verification uses Python stdlib `urllib` rather than adding `curl` to the image.

---

## 2. Services

Defined in [`docker-compose.yml`](../docker-compose.yml):

| Service | Container Name | Image | Port Mapping | Healthcheck |
| :--- | :--- | :--- | :--- | :--- |
| `backend` | `aiops-incident-intelligence` | `aiops-backend:latest` | `8000:8000` | `GET /api/v1/health` every 15s |

---

## 3. Environment Variables

Configured via [`.env`](../.env.example):

| Variable | Description | Container Default | Local Default |
| :--- | :--- | :--- | :--- |
| `PORT` | HTTP port exposed by Uvicorn | `8000` | `8000` |
| `HOST` | Bind host address | `0.0.0.0` | `0.0.0.0` |
| `AIOPS_DATA_DIR` | Absolute path to datasets and models | `/app/data` | `./data` |
| `AIOPS_SEED` | Pseudo-random seed for determinism | `42` | `42` |

---

## 4. Startup Flow

1. **Build Context**: Reads `requirements.txt`, application source (`app/`), static dashboard (`frontend/`), and initial models/datasets (`data/`). Excludes virtual environments and temporary test files via `.dockerignore`.
2. **Layer Packaging**: Installs pre-compiled binary wheels (`pandas`, `pyarrow`, `scikit-learn`, `scipy`, `fastapi`, `uvicorn`).
3. **Execution**: Starts Uvicorn ASGI server binding `0.0.0.0:${PORT}`.
4. **Health Probe**: Docker daemon triggers healthcheck `python -c "import urllib.request..."` to verify `/api/v1/health` responds with `{"status":"ok"}`. Container marks `healthy` within ~5 seconds.

---

## 5. Measured Resource Observations

*Hardware Environment: Linux x86_64, 8 vCPUs, 7.1 GiB RAM (Standard Developer Laptop Target)*

### Container Footprint (Measured Results)
- **Base Image**: `python:3.11-slim`
- **Docker Image Content Size**: `213 MB` (compressed layers: ~213 MB, uncompressed virtual disk size: ~923 MB)
- **Container Idle Memory (RAM)**: `132.6 MiB` (~1.8% of 7.1 GiB total)
- **Container Post-Load Memory (RAM)**: `151.0 MiB` (~2.1% of 7.1 GiB total)
- **Idle CPU Usage**: `< 0.3%`
- **Cold Startup Time (to healthy)**: `5.16 seconds`

### Representative API Latency (Measured, N=10 iterations)

| Endpoint | Method | Mean Latency | Median (p50) | Target / Spec |
| :--- | :--- | :--- | :--- | :--- |
| `/api/v1/health` | `GET` | **3.45 ms** | 3.42 ms | < 20 ms |
| `/api/v1/telemetry/latest?limit=50` | `GET` | **33.82 ms** | 33.62 ms | < 100 ms |
| `/api/v1/telemetry/summary` | `POST` | **16.02 ms** | 15.90 ms | < 50 ms |
| `/api/v1/anomalies/detect` (50 rows) | `POST` | **64.51 ms** | 58.79 ms | < 200 ms |
| `/api/v1/incidents/predict` (50 rows) | `POST` | **33.99 ms** | 33.32 ms | < 100 ms |
| `/api/v1/incidents/severity` (50 rows) | `POST` | **38.74 ms** | 39.04 ms | < 100 ms |
| `/api/v1/rca/rank` (50 rows) | `POST` | **67.57 ms** | 66.43 ms | < 200 ms |
| `/api/v1/rca/explain` (50 rows) | `POST` | **80.27 ms** | 74.73 ms | < 250 ms |
| `/api/v1/mlops/health` (1000 rows KS-test) | `GET` | **178.26 ms** | 175.90 ms | < 500 ms |

---

## 6. How Another Developer Can Reproduce

To spin up this project on any computer with Docker installed:

```bash
# 1. Clone the repository
git clone https://github.com/example/aiops-incident-intelligence.git
cd aiops-incident-intelligence

# 2. Create environment configuration from template
cp .env.example .env

# 3. Build and launch container in detached mode
docker compose up -d --build

# 4. Confirm container is running and healthy
docker compose ps

# 5. Access the Operations Dashboard
# Navigate to: http://localhost:8000
```

To stop the deployment:
```bash
docker compose down
```

---

## 7. Known Limitations

- **Single Worker Process**: Configured with a single Uvicorn worker process suitable for local development and demos; does not scale horizontally across multiple instances.
- **In-Memory Model Inference**: Scikit-Learn models are lazy-loaded as singletons inside the Python process.
- **Local Parquet Data Store**: Telemetry data is read directly from local filesystem parquet files mounted via volume, rather than an external data warehouse (e.g. Snowflake or BigQuery).
- **Single Host Binding**: Designed for localhost/developer machine access. Network policies, ingress controllers, TLS termination, and distributed secret managers are out of scope.
