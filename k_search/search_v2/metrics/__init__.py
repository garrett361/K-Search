from k_search.search_v2.config import MetricsConfig

from .noop import NoOpMetricsTracker
from .protocol import MetricsTracker
from .wandb import WandbMetricsTracker

__all__ = [
    "MetricsTracker",
    "NoOpMetricsTracker",
    "WandbMetricsTracker",
    "create_metrics_trackers",
]


def create_metrics_trackers(config: MetricsConfig | None = None) -> list[MetricsTracker]:
    config = config or MetricsConfig()
    if config.wandb:
        return [WandbMetricsTracker(config)]
    return [NoOpMetricsTracker()]
