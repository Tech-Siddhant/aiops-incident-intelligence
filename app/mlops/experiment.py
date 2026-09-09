"""Lightweight local experiment and model version tracking.

ponytail: O(N) file system scans for JSON reads. Upgrade to SQLite or MLflow 
only if runs exceed 1000s and file I/O becomes a measurable bottleneck.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRun:
    """Deterministic metadata tracking for a single model training/evaluation run."""
    model_name: str
    training_config: dict[str, Any]
    metrics: dict[str, float]
    model_version: str = "1.0.0"
    feature_version: str = "1.0"
    dataset_id: str = "unknown"
    random_seed: int | None = None
    artifact_path: str | None = None
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    training_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRun:
        return cls(**data)


class ExperimentTracker:
    """File-backed lightweight tracker for model experiments."""
    
    def __init__(self, storage_dir: str | Path = "data/experiments") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def log_run(self, run: ExperimentRun) -> Path:
        """Serialize and save an experiment run to JSON."""
        run_path = self.storage_dir / f"{run.run_id}.json"
        with open(run_path, "w") as f:
            json.dump(run.to_dict(), f, indent=2)
        return run_path

    def load_run(self, run_id: str) -> ExperimentRun:
        """Load an experiment run by its ID."""
        run_path = self.storage_dir / f"{run_id}.json"
        if not run_path.exists():
            raise FileNotFoundError(f"Run {run_id} not found in {self.storage_dir}.")
        with open(run_path, "r") as f:
            return ExperimentRun.from_dict(json.load(f))

    def list_runs(self, model_name: str | None = None) -> list[ExperimentRun]:
        """List runs, optionally filtered by model_name, sorted newest first."""
        runs = []
        for p in self.storage_dir.glob("*.json"):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                    if model_name is None or data.get("model_name") == model_name:
                        runs.append(ExperimentRun.from_dict(data))
            except (json.JSONDecodeError, TypeError):
                continue
        return sorted(runs, key=lambda r: r.training_timestamp, reverse=True)

    def compare_runs(
        self, 
        model_name: str, 
        metric: str, 
        reverse: bool = True
    ) -> list[ExperimentRun]:
        """Rank runs for a specific model based on a chosen metric."""
        runs = self.list_runs(model_name)
        # Default missing metrics to -inf if sorting descending, else inf
        default_val = float("-inf") if reverse else float("inf")
        return sorted(
            runs, 
            key=lambda r: r.metrics.get(metric, default_val), 
            reverse=reverse
        )
