# Data Quality Report: Synthetic Telemetry & Incident Ground Truth

## Overview
This report documents the quality assurance standards and validation results for the synthetic telemetry dataset and incident ground-truth metadata generated for the AIOps Incident Intelligence platform.

## Dataset Profile
- **Storage Format**: Apache Parquet (`data/synthetic/telemetry.parquet`) & JSON (`data/synthetic/incidents.json`)
- **Topology**: 4 services (`api_gateway`, `auth_service`, `orders_service`, `database`)
- **Sampling Interval**: 10 seconds
- **Default Duration**: 3,600 seconds (360 samples/service = 1,440 records total)
- **Failure Scenario**: `database_connection_saturation`

## Quality Invariants & Validation Rules
The validation suite (`app/data/validation.py`) enforces the following invariants:

| Category | Check | Rule / Constraint | Result |
|---|---|---|---|
| **Schema** | Column Completeness | All 12 required metrics present | PASSED |
| **Types** | Data Types | Numeric float/int metrics, string service identifiers | PASSED |
| **Completeness** | Null / Missing Values | 0 missing or NaN values across all series | PASSED |
| **Uniqueness** | Duplicate Rows | Zero duplicate `(timestamp, service)` tuples | PASSED |
| **Temporal Integrity** | Monotonicity | Strictly increasing ISO 8601 UTC timestamps per service | PASSED |
| **Domain Integrity** | Service Names | Only known services: `api_gateway`, `auth_service`, `orders_service`, `database` | PASSED |
| **Metric Bounds** | Usage Percentages | `cpu_usage_pct`, `memory_usage_pct`, `disk_usage_pct` in [0.0, 100.0] | PASSED |
| **Metric Bounds** | Ratio Metrics | `error_rate`, `connection_utilization` in [0.0, 1.0] | PASSED |
| **Metric Bounds** | Non-negative Metrics | `network_in_mbps`, `network_out_mbps`, `request_rate_rps`, `latency_ms`, `active_connections` >= 0 | PASSED |
| **Incident Integrity** | Temporal Ordering | `start_time <= detection_time <= end_time` within telemetry window | PASSED |
| **Incident Integrity** | Entity Alignment | Valid `root_cause_service` matching system topology | PASSED |

## Validation Summary
- **Validation Engine**: `scripts/validate_telemetry.py` / `app/data/validation.py`
- **Total Invariants Tested**: 11
- **Hard Errors**: 0
- **Warnings**: 0
- **Status**: PASSED
