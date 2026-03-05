"""Metrics tracking implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from k_search.modular.metrics.noop import NoOpMetricsTracker
from k_search.modular.metrics.wandb import WandbMetricsTracker

if TYPE_CHECKING:
    from k_search.modular.protocols import MetricsTracker
    from k_search.search_v2.config import MetricsConfig

__all__ = [
    "NoOpMetricsTracker",
    "WandbMetricsTracker",
    "create_metrics_trackers",
]


def create_metrics_trackers(
    config: MetricsConfig | None = None,
) -> list[MetricsTracker]:
    from k_search.search_v2.config import MetricsConfig as _MetricsConfig

    config = config or _MetricsConfig()
    if config.wandb:
        return [WandbMetricsTracker(config)]
    return [NoOpMetricsTracker()]
