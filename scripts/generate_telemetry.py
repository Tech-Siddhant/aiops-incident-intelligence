#!/usr/bin/env python3
"""CLI script to generate synthetic telemetry and incident ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import (
    DEFAULT_DURATION_SECONDS,
    DEFAULT_FAILURE_SCENARIO,
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    DEFAULT_SEED,
    DEFAULT_START_TIME,
    SYNTHETIC_DATA_DIR,
)
from app.data.synthetic import (
    SyntheticConfig,
    generate_synthetic_telemetry,
    save_synthetic_data,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic microservice telemetry dataset."
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=DEFAULT_START_TIME,
        help=f"Start timestamp in ISO 8601 format (default: {DEFAULT_START_TIME})",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
        help=f"Duration in seconds (default: {DEFAULT_DURATION_SECONDS})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_SAMPLING_INTERVAL_SECONDS,
        help=f"Sampling interval in seconds (default: {DEFAULT_SAMPLING_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_FAILURE_SCENARIO,
        choices=[DEFAULT_FAILURE_SCENARIO, "none"],
        help=f"Failure scenario to inject (default: {DEFAULT_FAILURE_SCENARIO})",
    )
    parser.add_argument(
        "--failure-start",
        type=int,
        default=900,
        help="Failure start offset in seconds (default: 900)",
    )
    parser.add_argument(
        "--failure-duration",
        type=int,
        default=900,
        help="Failure duration in seconds (default: 900)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SYNTHETIC_DATA_DIR),
        help=f"Output directory (default: {SYNTHETIC_DATA_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and save telemetry and ground truth."""
    args = parse_args()
    scenario = None if args.scenario == "none" else args.scenario
    failure_start = args.failure_start
    failure_dur = args.failure_duration
    if scenario is not None and failure_start >= args.duration:
        failure_start = args.duration // 4
        failure_dur = min(args.failure_duration, args.duration // 2)

    config = SyntheticConfig(
        start_time=args.start_time,
        duration_seconds=args.duration,
        sampling_interval_seconds=args.interval,
        seed=args.seed,
        failure_scenario=scenario,
        failure_start_seconds=failure_start,
        failure_duration_seconds=failure_dur,
    )

    print(f"Generating synthetic telemetry (seed={config.seed}, scenario={config.failure_scenario})...")
    df, incidents = generate_synthetic_telemetry(config)

    out_dir = Path(args.output_dir)
    parquet_path, json_path = save_synthetic_data(df, incidents, out_dir)

    print(f"Saved telemetry to: {parquet_path}")
    print(f"Saved incidents to: {json_path}")
    print(f"Total rows: {len(df)}")
    print(f"Services: {df['service'].unique().tolist()}")
    print(f"Timestamp range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Incidents generated: {len(incidents)}")


if __name__ == "__main__":
    main()

