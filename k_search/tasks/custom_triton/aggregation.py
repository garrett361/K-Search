"""Metric aggregation strategies for multi-configuration evaluation."""

from __future__ import annotations

import math
from typing import Protocol


class AggregationStrategy(Protocol):
    """Protocol for aggregating metrics across multiple workload configurations."""

    def aggregate(
        self,
        latencies: list[float],
        speedups: list[float] | None = None,
    ) -> dict[str, float]:
        """
        Aggregate metrics across multiple workload configurations.

        Args:
            latencies: List of latency values (in ms) for each passed workload
            speedups: Optional list of speedup factors for each passed workload

        Returns:
            Dictionary containing aggregated metrics:
            - mean_latency_ms: Aggregated latency
            - mean_speedup: Aggregated speedup (if speedups provided)
            - score: Comparable scalar score (higher is better)
        """
        ...


class MeanAggregation:
    """Arithmetic mean aggregation (default strategy)."""

    def aggregate(
        self,
        latencies: list[float],
        speedups: list[float] | None = None,
    ) -> dict[str, float]:
        """Compute arithmetic mean of metrics."""
        if not latencies:
            return {
                "mean_latency_ms": float("inf"),
                "mean_speedup": 0.0,
                "score": 0.0,
            }

        mean_latency = sum(latencies) / len(latencies)

        if speedups:
            mean_speedup = sum(speedups) / len(speedups)
            score = mean_speedup
        else:
            mean_speedup = 0.0
            score = 1.0 / mean_latency if mean_latency > 0 else 0.0

        return {
            "mean_latency_ms": mean_latency,
            "mean_speedup": mean_speedup,
            "score": score,
        }


class GeometricMeanAggregation:
    """
    Geometric mean aggregation.

    Better for multiplicative metrics like speedup, as it's less sensitive
    to outliers and gives equal weight to improvements and regressions.
    """

    def aggregate(
        self,
        latencies: list[float],
        speedups: list[float] | None = None,
    ) -> dict[str, float]:
        """Compute geometric mean of metrics."""
        if not latencies:
            return {
                "mean_latency_ms": float("inf"),
                "mean_speedup": 0.0,
                "score": 0.0,
            }

        if any(lat <= 0 for lat in latencies):
            return {
                "mean_latency_ms": float("inf"),
                "mean_speedup": 0.0,
                "score": 0.0,
            }

        log_sum = sum(math.log(lat) for lat in latencies)
        geom_mean_latency = math.exp(log_sum / len(latencies))

        if speedups and all(sp > 0 for sp in speedups):
            log_speedup_sum = sum(math.log(sp) for sp in speedups)
            geom_mean_speedup = math.exp(log_speedup_sum / len(speedups))
            score = geom_mean_speedup
        else:
            geom_mean_speedup = 0.0
            score = 1.0 / geom_mean_latency if geom_mean_latency > 0 else 0.0

        return {
            "mean_latency_ms": geom_mean_latency,
            "mean_speedup": geom_mean_speedup,
            "score": score,
        }
