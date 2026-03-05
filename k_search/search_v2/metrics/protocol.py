"""Protocol definition for metrics tracking."""

from typing_extensions import Protocol


class MetricsTracker(Protocol):
    """Protocol for metrics tracking implementations.

    Defines the interface for logging metrics during search operations.
    Implementations may write to various backends (console, wandb, tensorboard, etc.).
    """

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        """Log metrics to the tracking backend.

        Args:
            metrics: Dictionary mapping metric names to their values.
            step: Optional step/iteration number for time-series tracking.
        """
        ...
