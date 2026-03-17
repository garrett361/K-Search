class NoOpMetricsTracker:
    """No-op implementation of MetricsTracker for when metrics aren't needed."""

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        pass
