"""MetricsTracker protocol definition."""

from typing import Protocol


class MetricsTracker(Protocol):
    """Protocol for metrics tracking implementations."""

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        """Log metrics to the tracking backend."""
        ...
