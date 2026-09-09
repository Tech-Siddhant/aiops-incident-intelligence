"""Deterministic synthetic telemetry generator."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp", "service", "cpu_usage_pct", "memory_usage_pct",
    "disk_usage_pct", "network_in_mbps", "network_out_mbps",
    "request_rate_rps", "latency_ms", "error_rate",
    "active_connections", "connection_utilization"
]
VALID_SCENARIOS = ("database_connection_saturation",)

SERVICES = ("api_gateway", "auth_service", "orders_service", "database")

@dataclass(frozen=True)
class SyntheticConfig:
    start_time: str = "2026-01-01T00:00:00Z"
    duration_seconds: int = 3600
    sampling_interval_seconds: int = 10
    seed: int = 42
    failure_scenario: str | None = "database_connection_saturation"
    failure_start_seconds: int = 900
    failure_duration_seconds: int = 900
    failure_severity: str = "critical"

    def validate(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError(f"duration_seconds must be > 0")
        if self.sampling_interval_seconds <= 0:
            raise ValueError("sampling_interval_seconds must be > 0")
        if self.sampling_interval_seconds > self.duration_seconds:
            raise ValueError("sampling_interval_seconds cannot exceed duration_seconds")
        if self.failure_scenario is not None:
            if self.failure_scenario != "database_connection_saturation":
                raise ValueError("Unknown failure_scenario")
            if self.failure_duration_seconds <= 0:
                raise ValueError("failure_duration_seconds must be > 0")
            if self.failure_start_seconds < 0:
                raise ValueError("failure_start_seconds must be >= 0")
            if self.failure_start_seconds >= self.duration_seconds:
                raise ValueError("failure_start_seconds must be < duration_seconds")

def _iso_utc(dt: datetime) -> str:
    """Format datetime as ISO 8601 UTC string."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _series(srv: str, n: int, ts: np.ndarray, rng: np.random.Generator, cfg: SyntheticConfig) -> dict:
    bp = {
        "api_gateway": {"cpu_usage_pct": (25., 3.), "memory_usage_pct": (45., 1.5), "disk_usage_pct": (30., 0.2), "network_in_mbps": (20., 2.5), "network_out_mbps": (25., 3.), "request_rate_rps": (100., 10.), "latency_ms": (20., 3.), "error_rate": (0.002, 0.0005), "active_connections": (60., 5.), "connection_utilization": (0.3, 0.03)},
        "auth_service": {"cpu_usage_pct": (18., 2.), "memory_usage_pct": (35., 1.), "disk_usage_pct": (20., 0.1), "network_in_mbps": (8., 1.), "network_out_mbps": (8., 1.), "request_rate_rps": (75., 8.), "latency_ms": (8., 1.5), "error_rate": (0.001, 0.0003), "active_connections": (30., 3.), "connection_utilization": (0.2, 0.02)},
        "orders_service": {"cpu_usage_pct": (32., 3.5), "memory_usage_pct": (50., 1.5), "disk_usage_pct": (35., 0.2), "network_in_mbps": (15., 2.), "network_out_mbps": (15., 2.), "request_rate_rps": (65., 7.), "latency_ms": (25., 4.), "error_rate": (0.002, 0.0005), "active_connections": (45., 4.), "connection_utilization": (0.35, 0.03)},
        "database": {"cpu_usage_pct": (28., 3.), "memory_usage_pct": (65., 1.), "disk_usage_pct": (45., 0.2), "network_in_mbps": (14., 1.5), "network_out_mbps": (18., 2.), "request_rate_rps": (90., 8.), "latency_ms": (5., 0.8), "error_rate": (0.0005, 0.0002), "active_connections": (50., 4.), "connection_utilization": (0.45, 0.03)}
    }
    m = {k: rng.normal(mean, std, n) for k, (mean, std) in bp[srv].items()}

    if cfg.failure_scenario == "database_connection_saturation":
        fs = cfg.failure_start_seconds
        fe = min(cfg.duration_seconds, fs + cfg.failure_duration_seconds)
        delay = {"database": 0., "orders_service": cfg.sampling_interval_seconds*2., "api_gateway": cfg.sampling_interval_seconds*4.}.get(srv, float("inf"))
        if delay < float("inf"):
            ss, se = fs + delay, fe + delay
            intens = np.zeros(n)
            rd = cfg.sampling_interval_seconds*4.
            for i, t in enumerate(ts):
                if ss <= t < se: intens[i] = min(1., (t-ss)/max(1., rd))**2 * (3 - 2*min(1., (t-ss)/max(1., rd)))
                elif se <= t < se+rd: intens[i] = max(0., 1.-(t-se)/max(1., rd))**2 * (3 - 2*max(0., 1.-(t-se)/max(1., rd)))
            if srv == "database":
                m["connection_utilization"] += intens*.52; m["active_connections"] += intens*50
                m["cpu_usage_pct"] += intens*50; m["latency_ms"] += intens*650; m["error_rate"] += intens*.04
            elif srv == "orders_service":
                m["latency_ms"] += intens*850; m["error_rate"] += intens*.28
                m["cpu_usage_pct"] += intens*35; m["active_connections"] += intens*35; m["connection_utilization"] += intens*.40
            elif srv == "api_gateway":
                m["latency_ms"] += intens*950; m["error_rate"] += intens*.35
                m["cpu_usage_pct"] += intens*25; m["active_connections"] += intens*40

    m["cpu_usage_pct"] = np.clip(m["cpu_usage_pct"], 0., 100.)
    m["memory_usage_pct"] = np.clip(m["memory_usage_pct"], 0., 100.)
    m["disk_usage_pct"] = np.clip(m["disk_usage_pct"], 0., 100.)
    m["network_in_mbps"] = np.clip(m["network_in_mbps"], 0., None)
    m["network_out_mbps"] = np.clip(m["network_out_mbps"], 0., None)
    m["request_rate_rps"] = np.clip(m["request_rate_rps"], 0., None)
    m["latency_ms"] = np.clip(m["latency_ms"], 0., None)
    m["error_rate"] = np.clip(m["error_rate"], 0., 1.)
    m["active_connections"] = np.clip(np.round(m["active_connections"]), 0, None)

    m["connection_utilization"] = np.clip(m["connection_utilization"], 0., 1.)
    return m


def generate_synthetic_telemetry(config: SyntheticConfig | None = None) -> tuple[pd.DataFrame, list[dict]]:
    cfg = config or SyntheticConfig()
    cfg.validate()
    sdt = datetime.fromisoformat(cfg.start_time.replace("Z", "+00:00"))
    n = cfg.duration_seconds // cfg.sampling_interval_seconds
    rng = np.random.default_rng(cfg.seed)
    ts = np.arange(n) * cfg.sampling_interval_seconds
    t_strs = [_iso_utc(sdt + timedelta(seconds=int(t))) for t in ts]

    dfs = []
    for srv in SERVICES:
        srng = np.random.default_rng(int(rng.integers(0, 2**31-1)))
        dfs.append(pd.DataFrame({"timestamp": t_strs, "service": srv, **_series(srv, n, ts, srng, cfg)}))
    
    df = pd.concat(dfs, ignore_index=True).sort_values(["timestamp", "service"]).reset_index(drop=True)
    incs = []
    if cfg.failure_scenario == "database_connection_saturation":
        fs = sdt + timedelta(seconds=cfg.failure_start_seconds)
        fe = sdt + timedelta(seconds=min(cfg.duration_seconds, cfg.failure_start_seconds + cfg.failure_duration_seconds))
        fd = fs + timedelta(seconds=cfg.sampling_interval_seconds * 4)
        incs.append({
            "incident_id": "INC-001", "incident_type": cfg.failure_scenario, "root_cause_service": "database",
            "affected_services": ["database", "orders_service", "api_gateway"],
            "start_time": _iso_utc(fs), "detection_time": _iso_utc(fd), "end_time": _iso_utc(fe),
            "severity": cfg.failure_severity, "description": "Database connection pool saturation cascade."
        })
    return df, incs

def save_synthetic_data(df: pd.DataFrame, incs: list[dict], output_dir: str | Path = "data/synthetic") -> tuple[Path, Path]:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p/"telemetry.parquet", index=False, engine="pyarrow")
    with open(p/"incidents.json", "w") as f: json.dump(incs, f, indent=2)
    return p/"telemetry.parquet", p/"incidents.json"

