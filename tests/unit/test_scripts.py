"""Unit tests for CLI scripts."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from app.config import SYNTHETIC_DATA_DIR, TELEMETRY_PARQUET, INCIDENTS_JSON
from scripts.generate_telemetry import main as generate_main
from scripts.validate_telemetry import main as validate_main
from scripts.train_models import main as train_main


def test_scripts_pipeline(monkeypatch, tmp_path):
    """Test executing generate, validate, and train script entrypoints."""
    out_data_dir = tmp_path / "data"
    out_models_dir = tmp_path / "models"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Test generate_telemetry CLI
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_telemetry.py",
            "--duration", "600",
            "--interval", "10",
            "--output-dir", str(out_data_dir),
            "--scenario", "database_connection_saturation",
            "--failure-start", "100",
            "--failure-duration", "200",
        ],
    )
    generate_main()
    assert (out_data_dir / TELEMETRY_PARQUET).exists()
    assert (out_data_dir / INCIDENTS_JSON).exists()

    # 2. Test validate_telemetry CLI
    monkeypatch.setattr("app.config.SYNTHETIC_DATA_DIR", out_data_dir)
    monkeypatch.setattr("scripts.validate_telemetry.SYNTHETIC_DATA_DIR", out_data_dir)
    with pytest.raises(SystemExit) as excinfo:
        validate_main()
    assert excinfo.value.code == 0

    # 3. Test train_models CLI
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_models.py",
            "--output-dir", str(out_models_dir),
            "--seed", "42",
        ],
    )
    monkeypatch.setattr("scripts.train_models.SYNTHETIC_DATA_DIR", out_data_dir)
    train_main()
    assert (out_models_dir / "isolation_forest_latest.joblib").exists()
    assert (out_models_dir / "incident_predictor_latest.joblib").exists()
    assert (out_models_dir / "severity_classifier_latest.joblib").exists()
