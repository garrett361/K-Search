"""Local filesystem metrics tracker implementation."""
from __future__ import annotations
import json
from pathlib import Path


class LocalMetricsTracker:
    """Metrics tracker that writes JSONL to local filesystem."""

    def __init__(self, output_dir: Path | str, run_config: dict | None = None) -> None:
        self._output_dir = Path(output_dir)
        self._run_config = run_config
        self._run_id = run_config["run_id"] if run_config else None
        self._initialized = False

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        if not self._initialized:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            if self._run_config:
                (self._output_dir / "config.json").write_text(json.dumps(self._run_config, indent=2))
            self._initialized = True

        row: dict = {"step": step, **metrics}
        if self._run_id:
            row = {"run_id": self._run_id, **row}
        with (self._output_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
