"""Application configuration and path constants."""
from __future__ import annotations

import os
from pathlib import Path

# Repository paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("AIOPS_DATA_DIR", str(ROOT_DIR / "data")))
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = DATA_DIR / "models"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
DOCS_DIR = ROOT_DIR / "docs"
FRONTEND_STATIC_DIR = ROOT_DIR / "frontend" / "static"

# Default dataset filenames
TELEMETRY_PARQUET = "telemetry.parquet"
INCIDENTS_JSON = "incidents.json"

# Deterministic runtime defaults
DEFAULT_SEED = int(os.getenv("AIOPS_SEED", "42"))
DEFAULT_START_TIME = "2026-01-01T00:00:00Z"
DEFAULT_SAMPLING_INTERVAL_SECONDS = 10
DEFAULT_DURATION_SECONDS = 3600
DEFAULT_FAILURE_SCENARIO = "database_connection_saturation"

