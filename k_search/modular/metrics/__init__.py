"""Metrics tracking implementations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from k_search.modular.metrics.noop import NoOpMetricsTracker
from k_search.modular.metrics.wandb import WandbMetricsTracker
from k_search.modular.metrics.local import LocalMetricsTracker

if TYPE_CHECKING:
    from k_search.modular.protocols import MetricsTracker
    from k_search.modular.config import MetricsConfig

__all__ = [
    "NoOpMetricsTracker",
    "WandbMetricsTracker",
    "LocalMetricsTracker",
    "create_metrics_trackers",
]


def create_metrics_trackers(
    config: MetricsConfig | None = None,
    output_dir: Path | str | None = None,
    run_config: dict | None = None,
) -> list[MetricsTracker]:
    from k_search.modular.config import MetricsConfig as _MetricsConfig

    config = config or _MetricsConfig()
    trackers: list[MetricsTracker] = []

    if config.wandb:
        trackers.append(WandbMetricsTracker(config))

    if config.local and output_dir:
        trackers.append(LocalMetricsTracker(output_dir, run_config))

    return trackers or [NoOpMetricsTracker()]
