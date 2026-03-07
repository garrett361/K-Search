"""Wandb metrics tracker implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.modular.config import MetricsConfig


class WandbMetricsTracker:
    """Metrics tracker that logs to Weights & Biases."""

    def __init__(self, config: MetricsConfig) -> None:
        try:
            import wandb
        except ImportError as e:
            raise RuntimeError("wandb configured but not installed") from e

        if wandb.run is None:
            raise RuntimeError(
                "wandb configured but no active run (call wandb.init() first)"
            )

        self._wandb = wandb

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        self._wandb.log(metrics, step=step)
