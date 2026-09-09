"""FastAPI data validation schemas.

ponytail: Minimal Pydantic models. We accept records as dictionaries or typed records
to easily convert into Pandas DataFrames for the ML backend.
"""
from typing import Any
from pydantic import BaseModel, Field


class TelemetryRecord(BaseModel):
    """Single telemetry snapshot."""
    timestamp: str
    service: str
    cpu_usage_pct: float
    memory_usage_pct: float
    disk_usage_pct: float
    network_in_mbps: float
    network_out_mbps: float
    request_rate_rps: float
    latency_ms: float
    error_rate: float
    active_connections: int | float
    connection_utilization: float

class TelemetryBatchRequest(BaseModel):
    """Batch of telemetry records."""
    records: list[TelemetryRecord]

class RcaRequest(BaseModel):
    """Request for Root Cause Analysis over a time window."""
    records: list[TelemetryRecord]
    incident_start_time: str
    incident_end_time: str
    affected_services: list[str] = Field(default_factory=list)

class HealthResponse(BaseModel):
    """Basic API health."""
    status: str
    service: str = "omniroute-aiops"

class PredictIncidentRequest(BaseModel):
    """Request for predicting future incidents."""
    records: list[TelemetryRecord]
