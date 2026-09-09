# Data Contract: Telemetry & Incidents

## 1. Telemetry Schema
Telemetry is stored as columnar Parquet files (`data/synthetic/telemetry.parquet`) containing periodic metric snapshots per service.

| Field Name | Type | Range / Constraints | Description |
|------------|------|---------------------|-------------|
| `timestamp` | string (ISO 8601 UTC) / datetime64 | Valid UTC timestamp | Metric snapshot timestamp (e.g., `2026-01-01T00:00:00Z`) |
| `service` | string | `api_gateway`, `auth_service`, `orders_service`, `database` | Service identifier |
| `cpu_usage_pct` | float | [0.0, 100.0] | CPU utilization percentage |
| `memory_usage_pct` | float | [0.0, 100.0] | Memory utilization percentage |
| `disk_usage_pct` | float | [0.0, 100.0] | Disk space utilization percentage |
| `network_in_mbps` | float | >= 0.0 | Inbound network traffic in Mbps |
| `network_out_mbps` | float | >= 0.0 | Outbound network traffic in Mbps |
| `request_rate_rps` | float | >= 0.0 | Requests per second handled by service |
| `latency_ms` | float | >= 0.0 | Mean request/query processing latency in ms |
| `error_rate` | float | [0.0, 1.0] | Ratio of errors / total requests (0.0 to 1.0) |
| `active_connections` | int / float | >= 0 | Number of concurrent active connections |
| `connection_utilization` | float | [0.0, 1.0] | Fraction of max pool capacity utilized (0.0 to 1.0) |

## 2. Incident Ground Truth Schema
Incident records are stored as JSON (`data/synthetic/incidents.json`).

| Field Name | Type | Description |
|------------|------|-------------|
| `incident_id` | string | Unique identifier (e.g., `INC-001`) |
| `incident_type` | string | Failure scenario type (e.g., `database_connection_saturation`) |
| `root_cause_service` | string | Primary service responsible for the failure (`database`) |
| `start_time` | string (ISO 8601 UTC) | Timestamp when root cause failure injection begins |
| `detection_time` | string (ISO 8601 UTC) | Timestamp when downstream symptoms trigger detection |
| `end_time` | string (ISO 8601 UTC) | Timestamp when failure ends / resolves |
| `severity` | string | Incident severity (`low`, `medium`, `high`, `critical`) |
| `description` | string | Human-readable explanation of the failure dynamics |

## 3. Data Integrity & Invariants
- Each `(timestamp, service)` tuple must be unique (no duplicate metrics).
- Timestamps must be monotonically strictly increasing per service.
- No null / missing values allowed in telemetry records.
- For incidents: `start_time <= detection_time <= end_time`.
