#!/usr/bin/env python3
"""CLI script to validate synthetic telemetry data."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import INCIDENTS_JSON, SYNTHETIC_DATA_DIR, TELEMETRY_PARQUET
from app.data.validation import validate_telemetry


def main() -> None:
    parquet_path = SYNTHETIC_DATA_DIR / TELEMETRY_PARQUET
    json_path = SYNTHETIC_DATA_DIR / INCIDENTS_JSON

    if not parquet_path.exists():
        print(f"Error: {parquet_path} does not exist. Run generation first.")
        sys.exit(1)

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows.")

    incidents = []
    if json_path.exists():
        print(f"Loading {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            incidents = json.load(f)
        print(f"Loaded {len(incidents)} incidents.")

    print("Running validation rules...")
    val_res = validate_telemetry(df, incidents)

    print("\n=== Validation Report ===")
    print(f"Status: {'PASSED' if val_res.passed else 'FAILED'}")
    if val_res.errors:
        print("\nErrors:")
        for e in val_res.errors:
            print(f" - [X] {e}")
    if val_res.warnings:
        print("\nWarnings:")
        for w in val_res.warnings:
            print(f" - [!] {w}")

    sys.exit(0 if val_res.passed else 1)


if __name__ == "__main__":
    main()

